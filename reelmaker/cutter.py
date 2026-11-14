"""Cutter agent - FFmpeg does the actual cutting, 9:16 reframing, zoom punch-ins and colour.

Every kept piece is snapped to whole frames, rendered to its own file (in parallel), then joined
losslessly. Snapping keeps audio and video perfectly in sync no matter how many cuts there are.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import FPS, HEIGHT, WIDTH, Settings
from .director import EditPlan
from .utils import MediaInfo, Stopwatch, ffmpeg, log, run, warn

AGENT = "Cutter"


@dataclass
class Piece:
    src_start: float
    src_end: float
    out_start: float
    frames: int
    zoom: float = 1.0
    cx: float | None = None
    punch: bool = False          # montage: quick zoom-in "hit" at the start of the clip (lands on the beat)
    flash: bool = False          # montage: short white flash at the start of the clip

    @property
    def duration(self) -> float:
        return self.frames / FPS


class Timeline:
    """Maps source-video times to times in the edited reel."""

    def __init__(self, pieces: list[Piece]):
        self.pieces = pieces
        self.duration = sum(p.duration for p in pieces)

    def map_range(self, s: float, e: float) -> tuple[float, float] | None:
        # need 20ms of overlap, or half the word for very short words (zero-length words: just inside)
        need = min(0.02, max(0.0, e - s) * 0.5)
        best, best_ov = None, need - 1e-9
        for p in self.pieces:
            ov = min(e, p.src_end) - max(s, p.src_start)
            if ov > best_ov:
                best, best_ov = p, ov
        if best is None:
            return None
        ms = best.out_start + (max(s, best.src_start) - best.src_start)
        me = best.out_start + (min(e, best.src_end) - best.src_start)
        return ms, me


# --------------------------------------------------------------------------- filters

def choose_mode(info: MediaInfo, plan: EditPlan, settings: Settings, finder) -> str:
    mode = settings.reframe
    if mode != "auto":
        return mode
    if info.is_vertical:
        return "fit"
    montage = plan.mode == "montage"
    if finder is None:
        return "crop" if montage else "blur"
    if not finder.ok:                   # no face model, but we can still follow the action in a montage
        return "smart" if montage else "blur"
    segs = plan.segments
    step = max(1, len(segs) // 10)
    probes = segs[::step][:10]
    hits = sum(1 for s, e in probes if finder.center_x(s, e, samples=1) is not None)
    if probes and hits / len(probes) >= 0.5:
        mode = "face"
    else:
        mode = "smart" if montage else "blur"
    label = {"face": "face-tracking crop", "smart": "full-screen crop that follows the action",
             "blur": "blurred background fill"}[mode]
    log(AGENT, f"Faces in {hits}/{len(probes)} test frames -> {label}")
    return mode


def reframe_filter(mode: str, info: MediaInfo, cx: float | None, zoom: float) -> str:
    W, H = WIDTH, HEIGHT
    if mode in ("crop", "face", "smart"):
        sw, sh = info.width, info.height
        ar = W / H
        if sw / sh > ar:
            ch = sh / zoom
            cw = ch * ar
        else:
            cw = sw / zoom
            ch = cw / ar
        cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
        cxp = (cx if cx is not None else 0.5) * sw
        x = int(min(max(cxp - cw / 2, 0), sw - cw))
        y = int(min(max(sh * 0.5 - ch / 2, 0), sh - ch))
        return f"crop={cw}:{ch}:{x}:{y},scale={W}:{H}:flags=lanczos"

    if mode == "blur":
        base = (f"split=2[bg][fg];"
                f"[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                f"boxblur=10:2,scale={W}:{H},eq=brightness=-0.07:saturation=1.1[bgb];"
                f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fgs];"
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2")
    else:  # fit: fill the frame, crop the overflow
        base = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"
    if zoom > 1.001:
        zw, zh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
        base += f",scale={zw}:{zh}:flags=lanczos,crop={W}:{H}"
    return base


def _video_chain(mode: str, info: MediaInfo, piece: Piece, settings: Settings) -> str:
    # start_time=0 pins the frame grid to the cut point, so every piece has exactly `frames` frames
    # that line up with its audio (otherwise each cut can add a frame of lag).
    chain = f"fps=fps={FPS}:start_time=0," + reframe_filter(mode, info, piece.cx, piece.zoom)
    if piece.punch:   # 12% zoom that eases out over ~7 frames
        chain += (f",zoompan=z='1+0.12*pow(max(0\\,1-on/7)\\,2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                  f":d=1:s={WIDTH}x{HEIGHT}:fps={FPS}")
    if piece.flash:
        chain += ",eq=brightness='if(lt(t\\,0.1)\\,0.3*(1-t/0.1)\\,0)':eval=frame"
    if settings.enhance:
        chain += ",eq=contrast=1.05:saturation=1.12,unsharp=5:5:0.45:5:5:0.0"
    chain += ",setsar=1,format=yuv420p,tpad=stop_mode=clone:stop_duration=1"
    return chain


def _render_piece(idx: int, piece: Piece, info: MediaInfo, mode: str, settings: Settings, out: Path) -> Path:
    dur = piece.duration
    fade = min(0.012, dur / 4)
    samples = piece.frames * (48000 // FPS)          # exact audio length = exact video length
    # seek half a millisecond early so a frame sitting exactly on the cut point is not dropped
    seek = max(0.0, piece.src_start - 0.0005)
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{seek:.6f}", "-i", info.path]
    if not info.has_audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    cmd += ["-map", "0:V:0", "-map", "0:a:0" if info.has_audio else "1:a:0",   # V = skip cover art
            "-vf", _video_chain(mode, info, piece, settings),
            "-af", f"aresample=48000:async=1:first_pts=0,afade=t=in:st=0:d={fade:.3f},"
                   f"afade=t=out:st={max(0.0, dur - fade):.4f}:d={fade:.3f},"
                   f"apad,atrim=end_sample={samples}",
            "-frames:v", str(piece.frames),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-g", str(FPS),
            "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
            out]
    run(cmd, what=f"cutting piece {idx + 1}")
    return out


# --------------------------------------------------------------------------- main entry

def _cut_fx(info: MediaInfo, plan: EditPlan, settings: Settings, work: Path) -> tuple[Path, Timeline]:
    """Insane montage: every shot rendered by the FX engine, plus the sound-effect track."""
    from .fx import build_sfx, render_shots, write_wav
    try:
        edited, _ = render_shots(info, plan.shots, settings, work)
    except Exception as e:   # never lose the reel: fall back to plain beat-synced cuts
        warn(AGENT, f"Effects failed ({type(e).__name__}: {str(e)[:200]}) - using simple cuts instead")
        plan.segments = [(s0, s0 + sh.frames / FPS) for sh in plan.shots
                         for s0 in [max(0.0, min(sh.src_start, info.duration - sh.frames / FPS - 0.05))]]
        plan.shots, plan.sfx_events, plan.sfx = [], [], None
        return cut(info, plan, settings, work)
    pieces = [Piece(src_start=s, src_end=e, out_start=sh.out_start, frames=sh.frames)
              for sh, (s, e) in zip(plan.shots, plan.segments)]
    tl = Timeline(pieces)
    if plan.sfx_events:
        try:
            plan.sfx = str(write_wav(work / "sfx.wav", build_sfx(plan.sfx_events, tl.duration)))
        except Exception as e:  # sound effects are a bonus
            warn(AGENT, f"Sound effects skipped ({type(e).__name__})")
    return edited, tl


def cut(info: MediaInfo, plan: EditPlan, settings: Settings, work: Path) -> tuple[Path, Timeline]:
    if plan.shots:
        return _cut_fx(info, plan, settings, work)
    sw = Stopwatch()
    finder = None
    if settings.reframe in ("auto", "face", "smart") and not info.is_vertical:
        try:
            from .faces import FaceFinder
            finder = FaceFinder(info.path)
        except Exception as e:  # opencv missing or broken -> blur fallback
            warn(AGENT, f"Face detection unavailable ({e}); using blurred background")
    mode = choose_mode(info, plan, settings, finder)
    if mode == "face" and (finder is None or not finder.ok):
        mode = "crop"

    pieces: list[Piece] = []
    out_t = 0.0
    prev_end = None
    for k, (s, e) in enumerate(plan.segments):
        s = round(s * FPS) / FPS
        frames = max(1, round((e - s) * FPS))
        montage = plan.mode == "montage"
        jump = prev_end is not None and abs(s - prev_end) < 8.0
        zoom = 1.12 if (settings.zoom_cuts and jump and not montage and len(pieces) % 2 == 1) else 1.0
        pieces.append(Piece(src_start=s, src_end=s + frames / FPS, out_start=out_t, frames=frames, zoom=zoom,
                            punch=montage and settings.zoom_cuts,
                            flash=montage and settings.zoom_cuts and k % 4 == 0))
        out_t += frames / FPS
        prev_end = s + frames / FPS

    if mode == "face" and finder is not None:
        xs = [finder.center_x(p.src_start, p.src_end) for p in pieces]
        found = [x for x in xs if x is not None]
        fallback = statistics.median(found) if found else 0.5
        prev = None
        for p, x in zip(pieces, xs):
            x = x if x is not None else (prev if prev is not None else fallback)
            if prev is not None and abs(x - prev) < 0.06:
                x = prev  # ignore tiny jitter so the frame stays steady
            p.cx = prev = x
    if mode == "smart" and finder is not None:
        crop_frac = min(1.0, (info.height * WIDTH / HEIGHT) / max(1, info.width))
        for p in pieces:
            x = finder.center_x(p.src_start, p.src_end, samples=2) if finder.ok else None
            p.cx = x if x is not None else finder.content_x(p.src_start, p.src_end, crop_frac / p.zoom)
    if finder is not None:
        finder.close()

    seg_dir = work / "pieces"
    seg_dir.mkdir(parents=True, exist_ok=True)
    for old in seg_dir.glob("*.mov"):
        old.unlink()
    log(AGENT, f"Cutting {len(pieces)} pieces ({mode} reframe, zoom cuts {'on' if settings.zoom_cuts else 'off'})...")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_render_piece, i, p, info, mode, settings, seg_dir / f"p{i:04d}.mov")
                   for i, p in enumerate(pieces)]
        files = [f.result() for f in futures]

    listfile = seg_dir / "list.txt"
    listfile.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
    edited = work / "edited.mov"
    run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile.name, "-c", "copy", edited.resolve()], cwd=seg_dir, what="joining pieces")
    tl = Timeline(pieces)
    log(AGENT, f"Edited video ready: {tl.duration:.1f}s ({sw.lap()})")
    return edited, tl
