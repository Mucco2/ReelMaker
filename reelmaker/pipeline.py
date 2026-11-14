"""Runs the agents in order and hands each one's result to the next."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from . import reference
from .animator import animate
from .config import Settings
from .cutter import cut
from .director import plan_edits
from .fetch import resolve_input
from .finisher import finish
from .transcribe import transcribe
from .utils import OUTPUT_DIR, WORK_DIR, PipelineError, Stopwatch, file_fingerprint, log, probe, slugify

AGENT = "ReelMaker"


def make_reel(src: str | list[str], settings: Settings, ref: str | None = None, style_file: Path | None = None,
              explicit: set[str] | None = None) -> list[Path]:
    """Run every agent; returns the path of each finished reel.mp4 (one per --count).

    `src` may be several videos: they are mixed into one montage (Claude follows settings.prompt)."""
    sw = Stopwatch()
    explicit = explicit or set()
    downloads = WORK_DIR / "downloads"
    srcs = [src] if isinstance(src, (str, Path)) else [s for s in src if str(s).strip()]
    seen: set[str] = set()
    uniq = []
    for s in srcs:                                   # the same video twice would just repeat footage
        key = str(s).strip().strip('"').lower()
        key = str(Path(key).resolve()).lower() if not key.startswith(("http://", "https://")) else key
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    srcs = uniq
    if not srcs:
        raise PipelineError("No video given.")

    # 1. Fetcher
    videos: list[Path] = []
    infos = []
    works: list[Path] = []
    for i, s in enumerate(srcs):
        v = resolve_input(str(s), downloads)
        inf = probe(v)
        if not inf.has_video or inf.width == 0:
            raise PipelineError(f"{v.name} has no video stream.")
        tag = f"Video {i + 1}: " if len(srcs) > 1 else "Input: "
        log(AGENT, f"{tag}{v.name} | {inf.width}x{inf.height} | {inf.duration:.1f}s | {inf.fps:.0f} fps")
        w = WORK_DIR / f"{slugify(v.stem)}_{file_fingerprint(v)}"
        w.mkdir(parents=True, exist_ok=True)
        videos.append(v)
        infos.append(inf)
        works.append(w)
    if len(videos) > 1:
        return _make_multi(videos, infos, works, settings, ref, style_file, explicit, sw)
    video, info, work = videos[0], infos[0], works[0]

    # 2. Reference Analyst
    if ref:
        profile = reference.analyze(ref, WORK_DIR / "references", save_to=work / "style.json")
        reference.apply(profile, settings, explicit)
    elif style_file:
        reference.apply(json.loads(Path(style_file).read_text(encoding="utf-8")), settings, explicit)

    # 3. Transcriber
    transcript = transcribe(video, work, settings.whisper_model, settings.language, info.has_audio,
                            duration=info.duration)
    speech_secs = sum(w["end"] - w["start"] for s in transcript.get("segments", []) for w in s["words"])
    mode = settings.mode
    if mode == "auto":
        # (almost) nothing said -> a music montage of the best moments instead of a talking edit
        mode = "montage" if speech_secs < max(4.0, 0.06 * min(info.duration, 600)) else "speech"
    settings.speech_language = (settings.language or transcript.get("language")) if mode == "speech" else None
    if mode == "montage" and not settings.text_language:
        settings.text_language = settings.language or "en"
    settings.resolve_cta(settings.speech_language or settings.text_language)

    # 4. Director
    if mode == "montage":
        from .montage import plan_montage
        log(AGENT, f"Heard only {speech_secs:.1f}s of speech -> montage mode (best moments, cut to the beat)")
        plans = [p for p in plan_montage(info, settings, work, settings.count) if p.segments]
    else:
        plans = [p for p in plan_edits(transcript, info.duration, settings, settings.count) if p.segments]
    if not plans:
        raise PipelineError("Nothing left to keep after editing - try --pacing chill or a longer --max-duration.")

    stamp = f"{slugify(video.stem)}_{datetime.now():%Y%m%d-%H%M%S}"
    return _render_plans(plans, info, transcript, settings, work, stamp, [str(video)], sw)


def _make_multi(videos: list[Path], infos: list, works: list[Path], settings: Settings, ref: str | None,
                style_file: Path | None, explicit: set[str], sw: Stopwatch) -> list[Path]:
    """Several videos -> one montage. Claude watches all of them and follows the creator's instructions."""
    key = hashlib.sha1("|".join(w.name for w in works).encode("utf-8")).hexdigest()[:10]
    work = WORK_DIR / f"multi_{slugify(videos[0].stem, 24)}_{key}"
    work.mkdir(parents=True, exist_ok=True)
    if ref:
        profile = reference.analyze(ref, WORK_DIR / "references", save_to=work / "style.json")
        reference.apply(profile, settings, explicit)
    elif style_file:
        reference.apply(json.loads(Path(style_file).read_text(encoding="utf-8")), settings, explicit)
    if settings.mode == "speech":
        log(AGENT, "Several videos are always edited as one montage to the music")
    if settings.fx == "classic":
        log(AGENT, "Several videos use the effects edit (the calm style works with one video)")
    settings.speech_language = None
    if not settings.text_language:
        settings.text_language = settings.language or "en"
    settings.resolve_cta(settings.text_language)
    log(AGENT, f"{len(videos)} videos -> one montage cut to the music"
               + (" (following your instructions)" if (settings.prompt or "").strip() else ""))
    from .hype import plan_hype
    plans = [p for p in plan_hype(infos, settings, works, settings.count) if p.shots]
    if not plans:
        raise PipelineError("Nothing to edit - the videos seem to be empty.")
    stamp = f"{slugify(videos[0].stem, 30)}_plus{len(videos) - 1}_{datetime.now():%Y%m%d-%H%M%S}"
    return _render_plans(plans, infos[0], {"segments": []}, settings, work, stamp, [str(v) for v in videos], sw)


def _render_plans(plans: list, info, transcript: dict, settings: Settings, work: Path, stamp: str,
                  inputs: list[str], sw: Stopwatch) -> list[Path]:
    """Cutter -> Animator -> Finisher for every planned reel."""
    finals: list[Path] = []
    for k, plan in enumerate(plans, 1):
        many = len(plans) > 1
        if many:
            log(AGENT, f"--- Reel {k} of {len(plans)} ---")
        rwork = work / f"reel{k}"
        rwork.mkdir(parents=True, exist_ok=True)

        # 5. Cutter
        edited, timeline = cut(info, plan, settings, rwork)

        # 6. Animator
        overlay = animate(transcript, plan, timeline, settings, rwork)

        # 7. Finisher
        if settings.out_dir:
            out_dir = Path(settings.out_dir) / (f"reel{k}" if many else "")
        else:
            out_dir = OUTPUT_DIR / (f"{stamp}_reel{k}" if many else stamp)
        final = finish(edited, overlay, plan, settings, out_dir, rwork, timeline.duration)

        (out_dir / "edit_plan.json").write_text(json.dumps({
            "input": inputs[0] if len(inputs) == 1 else inputs, "settings": settings.to_dict(),
            "decided_by": plan.decided_by, "hook": plan.hook, "kept_ranges": plan.segments,
            "final_duration": round(timeline.duration, 3),
            "overlay": overlay.kind, "language": settings.speech_language or settings.text_language,
            "mode": plan.mode, "music_start": plan.music_start,
            "shots": [{key: v for key, v in s.to_dict().items() if key != "events"} for s in plan.shots],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        finals.append(final)
        print(f"RESULT_DIR={out_dir.resolve()}", flush=True)

    log(AGENT, f"[92mFinished in {sw.lap()}![0m  {len(finals)} reel(s) -> "
               + ", ".join(str(f) for f in finals))
    return finals
