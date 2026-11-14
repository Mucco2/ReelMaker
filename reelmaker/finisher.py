"""Finisher agent - lays the animation layer over the video, polishes the audio and exports.

Audio: rumble cut, light denoise, compression, optional ducked background music, and two-pass
loudness normalisation to -14 LUFS (what TikTok / Instagram / YouTube play back at).
Video: H.264 High, 1080x1920, 30 fps, yuv420p, faststart - accepted everywhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .animator import OverlayResult, write_srt
from .config import FPS, PLATFORM_LIMITS, Settings
from .director import EditPlan
from .utils import PipelineError, Stopwatch, ffmpeg, log, run, warn

AGENT = "Finisher"

VOICE_CHAIN = ("highpass=f=75,afftdn=nf=-25,"
               "acompressor=threshold=-20dB:ratio=3:attack=5:release=90:makeup=2")


MONTAGE_CHAIN = "acompressor=threshold=-16dB:ratio=2:attack=10:release=150"   # no denoise: keep engine/crowd sound


def _music_input(settings: Settings, plan: EditPlan) -> list:
    """Music input args; a montage starts the song at the chosen beat."""
    montage = plan.mode == "montage"
    # A montage with a beat grid loops a whole number of beats in the filter instead (stays on the beat).
    # No input -ss: seeking into an MP3 drops the first hit (bit reservoir), so the start is trimmed in the
    # filter graph, which is sample-exact.
    args = [] if (montage and plan.music_loop > 0) else ["-stream_loop", "-1"]
    return args + ["-i", Path(settings.music).resolve()]


def _extra_inputs(settings: Settings, plan: EditPlan, first: int) -> tuple[list, int | None, int | None]:
    """Input args for the music and the sound-effect track; returns (args, music index, sfx index)."""
    args: list = []
    music_idx = sfx_idx = None
    idx = first
    if settings.music:
        args += _music_input(settings, plan)
        music_idx = idx
        idx += 1
    if plan.sfx and Path(plan.sfx).exists():
        args += ["-i", Path(plan.sfx).resolve()]
        sfx_idx = idx
    return args, music_idx, sfx_idx


def _audio_graph(settings: Settings, music_idx: int | None, plan: EditPlan | None = None,
                 duration: float = 0.0, sfx_idx: int | None = None) -> str:
    """Filtergraph producing [mix] from [0:a] (+ optional music and sound effects)."""
    montage = plan is not None and plan.mode == "montage"
    fx = (f"[{sfx_idx}:a]aresample=48000,aformat=channel_layouts=stereo[fx];" if sfx_idx is not None else "")
    if music_idx is None:
        if fx:
            return (f"[0:a]{MONTAGE_CHAIN if montage else VOICE_CHAIN}[orig];{fx}"
                    f"[orig][fx]amix=inputs=2:duration=first:normalize=0[mix]")
        return f"[0:a]{MONTAGE_CHAIN if montage else VOICE_CHAIN}[mix]"
    if montage:   # the music leads, the original sound sits underneath
        fade = max(0.0, duration - 1.2)
        s = max(0.0, plan.music_start)
        if plan.music_loop > 0:   # cut the song at the first beat; if it's too short, repeat whole beats
            loop = (f"atrim=start={s:.4f}:end={s + plan.music_loop:.4f},asetpts=PTS-STARTPTS,"
                    f"aloop=loop=-1:size={int(round(plan.music_loop * 48000))},")
        else:
            loop = f"atrim=start={s:.4f},asetpts=PTS-STARTPTS," if s > 0 else ""
        return (f"[0:a]{MONTAGE_CHAIN},volume=0.32[orig];"
                f"[{music_idx}:a]aresample=48000,{loop}aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d=0.005,afade=t=out:st={fade:.3f}:d=1.2[mu];"   # keep the first beat's punch
                + (f"{fx}[orig][mu][fx]amix=inputs=3:duration=first:normalize=0[mix]" if fx else
                   "[orig][mu]amix=inputs=2:duration=first:normalize=0[mix]"))
    return (f"[0:a]{VOICE_CHAIN},asplit=2[vo1][vo2];"
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={settings.music_volume:.3f}[mu];"
            f"[mu][vo2]sidechaincompress=threshold=0.04:ratio=8:attack=15:release=400[duck];"
            f"[vo1][duck]amix=inputs=2:duration=first:normalize=0[mix]")


def _measure_loudness(edited: Path, settings: Settings, plan: EditPlan, duration: float) -> dict | None:
    cmd = [ffmpeg(), "-hide_banner", "-nostats", "-i", edited]
    extra, music_idx, sfx_idx = _extra_inputs(settings, plan, 1)
    cmd += extra
    graph = _audio_graph(settings, music_idx, plan, duration, sfx_idx) + \
        f";[mix]loudnorm=I={settings.loudness}:TP=-1.5:LRA=11:print_format=json[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]", "-t", f"{duration:.3f}", "-f", "null", "-"]
    try:
        proc = run(cmd, what="loudness measurement")
    except PipelineError as e:
        warn(AGENT, f"Could not measure loudness ({e}); using one-pass normalisation")
        return None
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.S)
    if not m:
        return None
    data = json.loads(m[-1])
    try:
        if float(data["input_i"]) < -70:   # basically silent
            return {"silent": True}
    except (ValueError, KeyError):
        return None
    return data


def _loudnorm(settings: Settings, meas: dict | None) -> str:
    base = f"loudnorm=I={settings.loudness}:TP=-1.5:LRA=11"
    if not meas:
        return base
    if meas.get("silent"):
        return "anull"
    return (f"{base}:measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:"
            f"measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:"
            f"offset={meas['target_offset']}:linear=true")


def finish(edited: Path, overlay: OverlayResult, plan: EditPlan, settings: Settings,
           out_dir: Path, work: Path, duration: float) -> Path:
    sw = Stopwatch()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "reel.mp4"

    log(AGENT, "Measuring loudness...")
    meas = _measure_loudness(edited, settings, plan, duration)

    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", edited.resolve()]
    idx = 1
    v_graph = "[0:v]format=yuv420p[v]"
    if overlay.kind == "hyperframes" and overlay.path:
        cmd += ["-c:v", "libvpx-vp9", "-i", overlay.path.resolve()]  # libvpx decoder keeps the alpha channel
        v_graph = f"[0:v][{idx}:v]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[v]"
        idx += 1
    elif overlay.kind == "ass" and overlay.path:
        v_graph = f"[0:v]ass={overlay.path.name},format=yuv420p[v]"   # relative path: run from work dir
    extra, music_idx, sfx_idx = _extra_inputs(settings, plan, idx)
    cmd += extra

    a_graph = _audio_graph(settings, music_idx, plan, duration, sfx_idx) + \
        f";[mix]{_loudnorm(settings, meas)},aresample=48000[a]"
    cmd += ["-filter_complex", f"{v_graph};{a_graph}", "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-profile:v", "high", "-level:v", "4.2",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", str(FPS * 2), "-maxrate", "12M", "-bufsize", "24M",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", "-t", f"{duration:.3f}", final.resolve()]
    log(AGENT, "Exporting final reel...")
    run(cmd, cwd=work, what="final export")

    # cover image (hook is on screen at ~1s)
    cover = out_dir / "cover.jpg"
    try:
        run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{min(1.0, duration / 2):.2f}",
             "-i", final, "-frames:v", "1", "-q:v", "2", cover], what="cover image")
    except PipelineError:
        warn(AGENT, "Could not make the cover image")

    if overlay.chunks:
        write_srt(overlay.chunks, out_dir / "captions.srt")

    ok_for = [name for name, lim in (("TikTok", PLATFORM_LIMITS["tiktok"]), ("Instagram Reels", PLATFORM_LIMITS["reels"]),
                                     ("YouTube Shorts", PLATFORM_LIMITS["shorts"]), ("Facebook Reels", PLATFORM_LIMITS["facebook"]))
              if duration <= lim]
    post = [
        f"TITLE: {plan.title}".rstrip(),
        "",
        "CAPTION:",
        plan.description,
        "",
        " ".join(plan.hashtags),
        "",
        "-" * 50,
        f"Length {duration:.1f}s | 1080x1920 | {FPS} fps | H.264 + AAC | {settings.loudness:.0f} LUFS",
        f"Ready for: {', '.join(ok_for) if ok_for else 'TikTok (long-form)'}",
        f"Edit decided by: {plan.decided_by}" + (f' | Hook: "{plan.hook}"' if plan.hook else ""),
    ]
    (out_dir / "post.txt").write_text("\n".join(post) + "\n", encoding="utf-8")
    log(AGENT, f"Done in {sw.lap()} -> {final}")
    return final
