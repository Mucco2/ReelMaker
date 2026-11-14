"""Shared helpers: finding tools, running commands, probing media, logging."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = APP_DIR / "engine"
WORK_DIR = APP_DIR / "work"
OUTPUT_DIR = APP_DIR / "output"

# Windows consoles understand ANSI colours on Windows 10+, but make sure.
if os.name == "nt":
    os.system("")

_COLORS = {
    "Fetcher": "36", "Reference": "35", "Transcriber": "34", "Director": "33",
    "Cutter": "32", "Animator": "95", "Finisher": "96", "ReelMaker": "97",
}


class PipelineError(RuntimeError):
    """A step failed in a way the user should see."""


USE_COLOR = not os.environ.get("NO_COLOR")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _out(text: str) -> None:
    print(text if USE_COLOR else _ANSI.sub("", text), flush=True)


def log(agent: str, msg: str) -> None:
    color = _COLORS.get(agent, "37")
    _out(f"\033[{color}m[{agent}]\033[0m {msg}")


def warn(agent: str, msg: str) -> None:
    _out(f"\033[{_COLORS.get(agent, '37')}m[{agent}]\033[0m \033[93mWARNING:\033[0m {msg}")


# --------------------------------------------------------------------------- tools

def _winget_candidates(exe: str) -> list[str]:
    local = os.environ.get("LOCALAPPDATA", "")
    if not local:
        return []
    pats = [
        os.path.join(local, "Microsoft", "WinGet", "Links", exe),
        os.path.join(local, "Microsoft", "WinGet", "Packages", "*", exe),
        os.path.join(local, "Microsoft", "WinGet", "Packages", "*", "*", exe),
        os.path.join(local, "Microsoft", "WinGet", "Packages", "*", "*", "bin", exe),
    ]
    found: list[str] = []
    for p in pats:
        found.extend(sorted(glob.glob(p), reverse=True))
    # Prefer the Gyan "full" FFmpeg build (has libass, libvpx, ...) over other bundled copies.
    found.sort(key=lambda p: 0 if "Gyan.FFmpeg" in p else 1)
    return found


@lru_cache(maxsize=None)
def find_tool(name: str) -> str | None:
    """Locate an executable even if the PATH has not been refreshed since install."""
    exe = name + (".exe" if os.name == "nt" and not name.endswith(".exe") else "")
    hit = shutil.which(name)
    if hit:
        return hit
    for cand in _winget_candidates(exe):
        if os.path.isfile(cand):
            return cand
    return None


def require_tool(name: str) -> str:
    path = find_tool(name)
    if not path:
        raise PipelineError(
            f"'{name}' was not found. Run setup.bat first (it installs FFmpeg, yt-dlp and the rest)."
        )
    return path


def ffmpeg() -> str:
    return require_tool("ffmpeg")


def ffprobe() -> str:
    return require_tool("ffprobe")


def tool_env() -> dict:
    """Environment for child processes with FFmpeg's folder on PATH (HyperFrames needs it)."""
    env = os.environ.copy()
    ff = find_tool("ffmpeg")
    if ff:
        env["PATH"] = str(Path(ff).parent) + os.pathsep + env.get("PATH", "")
    return env


def run(cmd: list, *, cwd: str | Path | None = None, capture: bool = True,
        timeout: float | None = None, what: str = "command") -> subprocess.CompletedProcess:
    """Run a command; raise PipelineError with the useful part of stderr if it fails."""
    cmd = [str(c) for c in cmd]
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, env=tool_env(), timeout=timeout,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        raise PipelineError(f"Could not start {what}: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise PipelineError(f"{what} timed out after {timeout:.0f}s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise PipelineError(f"{what} failed (exit {proc.returncode}):\n  " + "\n  ".join(tail))
    return proc


# --------------------------------------------------------------------------- media

@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int          # display width (rotation applied)
    height: int         # display height (rotation applied)
    fps: float
    has_audio: bool
    has_video: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def is_vertical(self) -> bool:
        return self.aspect < 0.7


def _parse_rate(rate: str | None) -> float:
    if not rate or rate in ("0/0", "0"):
        return 0.0
    if "/" in rate:
        n, d = rate.split("/", 1)
        try:
            return float(n) / float(d) if float(d) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def _duration_from_packets(path: Path) -> float:
    """For files with no stored duration (e.g. browser/screen-recorder WebM): read packet timestamps."""
    proc = run([ffprobe(), "-v", "error", "-show_entries", "packet=pts_time,duration_time",
                "-of", "csv=p=0", path], what="ffprobe packets", timeout=600)
    best = 0.0
    for line in (proc.stdout or "").splitlines():
        vals = []
        for x in line.strip().strip(",").split(","):
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(0.0)
        if vals:
            best = max(best, sum(vals[:2]))
    return best


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    proc = run([ffprobe(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
               what="ffprobe")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0)
    if not duration and v:
        duration = float(v.get("duration") or 0)
    w = h = 0
    fps = 30.0
    if v:
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(float(sd["rotation"]))
        rot = int(float((v.get("tags") or {}).get("rotate", rot)))
        if abs(rot) % 180 == 90:
            w, h = h, w
        fps = _parse_rate(v.get("avg_frame_rate")) or _parse_rate(v.get("r_frame_rate")) or 30.0
    if duration <= 0 and (v or a):
        duration = _duration_from_packets(path)
    return MediaInfo(path=path, duration=duration, width=w, height=h, fps=fps,
                     has_audio=a is not None, has_video=v is not None)


# --------------------------------------------------------------------------- misc

def slugify(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    text = re.sub(r"_+", "_", text)
    return (text[:maxlen] or "video").rstrip("_")


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8", "replace"))
    return h.hexdigest()[:10]


def fmt_time(sec: float) -> str:
    m, s = divmod(max(0.0, sec), 60)
    return f"{int(m)}:{s:04.1f}"


class Stopwatch:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> str:
        return f"{time.time() - self.t0:.1f}s"


def python_exe() -> str:
    return sys.executable
