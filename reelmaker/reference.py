"""Reference Analyst agent - studies a reel you like (link or file) and copies its rhythm.

It measures length, how often the shot changes, and how much silence there is, then turns that
into settings (target length, pacing, zoom cuts) for your video.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import PLATFORM_LIMITS, Settings
from .fetch import resolve_input
from .utils import ffmpeg, log, probe, run

AGENT = "Reference"


def analyze(src: str, downloads: Path, save_to: Path | None = None) -> dict:
    path = resolve_input(src, downloads)
    info = probe(path)
    log(AGENT, f"Studying {path.name} ({info.duration:.1f}s, {info.width}x{info.height})")

    cuts = 0
    if info.has_video:
        proc = run([ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-an",
                    "-vf", "scale=240:-2,select='gt(scene,0.32)',showinfo", "-f", "null", "-"],
                   what="shot-change detection")
        cuts = len(re.findall(r"pts_time:", proc.stderr or ""))

    silence = 0.0
    if info.has_audio:
        proc = run([ffmpeg(), "-hide_banner", "-nostats", "-i", path, "-vn",
                    "-af", "silencedetect=n=-35dB:d=0.35", "-f", "null", "-"], what="silence detection")
        silence = sum(float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", proc.stderr or ""))

    dur = max(info.duration, 0.1)
    cpm = cuts / (dur / 60.0)
    silence_ratio = min(1.0, silence / dur)

    # Shot changes are the strongest signal of an edited, high-energy reel; silence refines it.
    if cpm >= 24 or (cpm >= 10 and silence_ratio < 0.08):
        pacing = "fast"
    elif cpm <= 6 and silence_ratio > 0.20:
        pacing = "chill"
    else:
        pacing = "normal"

    profile = {
        "source": src,
        "duration": round(dur, 2),
        "width": info.width, "height": info.height,
        "shot_changes": cuts,
        "cuts_per_minute": round(cpm, 1),
        "seconds_per_shot": round(dur / (cuts + 1), 2),
        "silence_ratio": round(silence_ratio, 3),
        "suggested": {
            "target_duration": round(min(max(dur, 10.0), PLATFORM_LIMITS["tiktok"])),
            "pacing": pacing,
            "zoom_cuts": cpm >= 10,
        },
    }
    log(AGENT, f"{dur:.0f}s long, a new shot every {profile['seconds_per_shot']}s, "
               f"{silence_ratio * 100:.0f}% silence -> pacing '{pacing}', target ~{profile['suggested']['target_duration']}s")
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_to.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def apply(profile: dict, settings: Settings, explicit: set[str]) -> None:
    """Copy suggested values into settings unless the user set them on the command line."""
    sug = profile.get("suggested", {})
    if "target_duration" in sug and "max_duration" not in explicit:
        settings.target_duration = float(sug["target_duration"])
    if "pacing" in sug and "pacing" not in explicit:
        settings.pacing = sug["pacing"]
    if "zoom_cuts" in sug and "zoom_cuts" not in explicit:
        settings.zoom_cuts = bool(sug["zoom_cuts"])
    settings.style_notes = [
        f"Reference video is {profile.get('duration')}s long",
        f"It changes shot every {profile.get('seconds_per_shot')}s ({profile.get('cuts_per_minute')} cuts/min)",
        f"About {round(profile.get('silence_ratio', 0) * 100)}% of it is silence",
    ]
