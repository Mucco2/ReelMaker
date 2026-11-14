"""Fetcher agent - gets videos from links (YouTube, TikTok, Instagram, ...) with yt-dlp."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from .utils import PipelineError, find_tool, log, python_exe, run

AGENT = "Fetcher"
URL_RE = re.compile(r"^https?://", re.I)


def is_url(text: str) -> bool:
    return bool(URL_RE.match(text.strip()))


def _ytdlp_cmd() -> list[str]:
    # Prefer the pip-installed module: `pip install -U yt-dlp` keeps it current when sites change.
    if importlib.util.find_spec("yt_dlp") is not None:
        return [python_exe(), "-m", "yt_dlp"]
    exe = find_tool("yt-dlp")
    if exe:
        return [exe]
    raise PipelineError("yt-dlp is not installed. Run setup.bat.")


def download(url: str, dest_dir: Path, max_height: int = 1080) -> Path:
    """Download a single video as MP4 and return its path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(AGENT, f"Downloading {url}")
    fmt = (f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
           f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b")
    cmd = _ytdlp_cmd() + [
        "-f", fmt, "--merge-output-format", "mp4",
        "--no-playlist", "--restrict-filenames", "--no-progress", "--no-warnings",
        "-o", str(dest_dir / "%(title).60B_%(id)s.%(ext)s"),
        "--print", "after_move:filepath",
    ]
    ff = find_tool("ffmpeg")
    if ff:
        cmd += ["--ffmpeg-location", str(Path(ff).parent)]
    cmd.append(url)
    proc = run(cmd, what="yt-dlp download", timeout=1800)
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        p = Path(ln)
        if p.exists():
            log(AGENT, f"Saved {p.name}")
            return p
    # fall back to newest file in the folder
    vids = sorted((p for p in dest_dir.iterdir() if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if vids:
        return vids[0]
    raise PipelineError("yt-dlp finished but no video file was found.")


def resolve_input(src: str, downloads_dir: Path) -> Path:
    """Accept a file path or a link; always return a local file."""
    src = src.strip().strip('"').strip("'")
    if is_url(src):
        return download(src, downloads_dir)
    p = Path(src).expanduser()
    if not p.exists():
        raise PipelineError(f"File not found: {p}")
    return p.resolve()
