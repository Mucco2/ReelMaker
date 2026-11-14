"""Runs the agents in order and hands each one's result to the next."""

from __future__ import annotations

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


def make_reel(src: str, settings: Settings, ref: str | None = None, style_file: Path | None = None,
              explicit: set[str] | None = None) -> list[Path]:
    """Run every agent; returns the path of each finished reel.mp4 (one per --count)."""
    sw = Stopwatch()
    explicit = explicit or set()
    downloads = WORK_DIR / "downloads"

    # 1. Fetcher
    video = resolve_input(src, downloads)
    info = probe(video)
    if not info.has_video or info.width == 0:
        raise PipelineError(f"{video.name} has no video stream.")
    log(AGENT, f"Input: {video.name} | {info.width}x{info.height} | {info.duration:.1f}s | {info.fps:.0f} fps")
    work = WORK_DIR / f"{slugify(video.stem)}_{file_fingerprint(video)}"
    work.mkdir(parents=True, exist_ok=True)

    # 2. Reference Analyst
    if ref:
        profile = reference.analyze(ref, WORK_DIR / "references", save_to=work / "style.json")
        reference.apply(profile, settings, explicit)
    elif style_file:
        reference.apply(json.loads(Path(style_file).read_text(encoding="utf-8")), settings, explicit)

    # 3. Transcriber
    transcript = transcribe(video, work, settings.whisper_model, settings.language, info.has_audio,
                            duration=info.duration)
    settings.speech_language = settings.language or transcript.get("language")
    settings.resolve_cta(settings.speech_language)

    # 4. Director
    plans = [p for p in plan_edits(transcript, info.duration, settings, settings.count) if p.segments]
    if not plans:
        raise PipelineError("Nothing left to keep after editing - try --pacing chill or a longer --max-duration.")

    stamp = f"{slugify(video.stem)}_{datetime.now():%Y%m%d-%H%M%S}"
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
            "input": str(video), "settings": settings.to_dict(), "decided_by": plan.decided_by,
            "hook": plan.hook, "kept_ranges": plan.segments, "final_duration": round(timeline.duration, 3),
            "overlay": overlay.kind, "language": transcript.get("language"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        finals.append(final)
        print(f"RESULT_DIR={out_dir.resolve()}", flush=True)

    log(AGENT, f"\033[92mFinished in {sw.lap()}!\033[0m  {len(finals)} reel(s) -> "
               + ", ".join(str(f) for f in finals))
    return finals
