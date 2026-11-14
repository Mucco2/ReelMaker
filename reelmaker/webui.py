"""ReelMaker's modern window.

A tiny local web server (127.0.0.1 only, protected by a random token) serves a single-page UI and runs the
pipeline as a subprocess. The page opens in Microsoft Edge "app mode" so it looks like a normal desktop app
(no address bar, own icon). If Edge is missing it falls back to Chrome or the default browser.

    python -m reelmaker.webui [video]
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

from . import gui as core          # shared helpers: build_command, parse_line, settings, post text
from .utils import APP_DIR, OUTPUT_DIR, WORK_DIR, find_tool, probe, run

HTML_FILE = Path(__file__).with_name("webui") / "index.html"
UPLOADS = WORK_DIR / "uploads"
THUMBS = WORK_DIR / "ui-thumbs"
PROFILE = WORK_DIR / "ui-profile"
LOG_FILE = WORK_DIR / "webui.log"
TOKEN = secrets.token_urlsafe(24)
IDLE_EXIT = 180          # seconds without any request (and no job) before the server quits
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

STATE = {"initial": "", "last_seen": time.time(), "bye_at": None, "port": 0}
STOP = threading.Event()


def _log(msg: str) -> None:
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


def file_url(p: Path) -> str:
    return "/api/file?p=" + urllib.parse.quote(str(p)) + "&t=" + TOKEN


def _inside(p: Path, *roots: Path) -> bool:
    try:
        rp = p.resolve()
        return any(rp.is_relative_to(r.resolve()) for r in roots)
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- results

def result_info(folder: str | Path) -> dict:
    f = Path(folder)
    title = ""
    try:
        first = (f / "post.txt").read_text(encoding="utf-8").splitlines()[0]
        if first.startswith("TITLE:"):
            title = first[6:].strip()
    except (OSError, IndexError):
        pass
    plan = {}
    try:
        plan = json.loads((f / "edit_plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {
        "dir": str(f), "name": f.name, "title": title, "post": core.read_post_text(f),
        "duration": plan.get("final_duration"), "hook": plan.get("hook"),
        "decided_by": plan.get("decided_by"), "language": plan.get("language"), "mode": plan.get("mode"),
        "video": file_url(f / "reel.mp4"), "cover": file_url(f / "cover.jpg") if (f / "cover.jpg").exists() else "",
        "time": (f / "reel.mp4").stat().st_mtime if (f / "reel.mp4").exists() else 0,
    }


def recent_outputs(limit: int = 12) -> list[dict]:
    if not OUTPUT_DIR.exists():
        return []
    dirs = [d for d in OUTPUT_DIR.iterdir() if (d / "reel.mp4").exists()]
    dirs.sort(key=lambda d: (d / "reel.mp4").stat().st_mtime, reverse=True)
    return [result_info(d) for d in dirs[:limit]]


# --------------------------------------------------------------------------- the running job

# progress ranges: shared steps first, then Cutter/Animator/Finisher once per reel
_SHARED = {"ReelMaker": 1, "Fetcher": 3, "Reference": 6, "Transcriber": 10, "Director": 32}
_PER_REEL = {"Cutter": 0.08, "Animator": 0.30, "Finisher": 0.88}


class Job:
    def __init__(self, cmd: list[str], count: int):
        self.lock = threading.Lock()
        self.cmd, self.count = cmd, count
        self.lines: list[tuple[int, str]] = []
        self.seq = 0
        self.running, self.stopped = True, False
        self.rc: int | None = None
        self.agent, self.status, self.progress = "ReelMaker", "Starter…", 1.0
        self.reel = 0
        self.results: list[dict] = []
        self.fallback: list[str] = []
        self.error_lines: list[str] = []
        self.download = None
        self.started = time.time()
        env = os.environ.copy()
        env.update({"NO_COLOR": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        self.proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1,
                                     creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                self._handle(raw)
            except Exception:  # never let one odd line kill the reader
                _log(traceback.format_exc())
        rc = self.proc.wait()
        with self.lock:
            self.rc, self.running = rc, False
            if not self.results:
                self.results = [result_info(f) for f in self.fallback]
            if rc == 0 and self.results:
                self.progress, self.status = 100.0, "Færdig!"

    def _add(self, text: str) -> None:
        self.seq += 1
        self.lines.append((self.seq, text))
        if len(self.lines) > 3000:
            del self.lines[:1000]

    def _handle(self, raw: str) -> None:
        info = core.parse_line(raw)
        text = info["text"].strip()
        if not text:
            return
        with self.lock:
            if info["progressbar"]:              # model download bars: show % in the status only
                m = re.search(r"(\d+)%\|", text)
                if m:
                    self.download = int(m.group(1))
                    self.status = f"Henter talemodellen… {self.download}%"
                return
            self._add(text)
            if self.error_lines or info["error"]:
                if len(self.error_lines) < 10:
                    self.error_lines.append(text)
            m = re.match(r"\[ReelMaker\] --- Reel (\d+) of (\d+) ---", text)
            if m:
                self.reel = int(m.group(1)) - 1
            if info["result"]:
                self.results.append(result_info(info["result"]))
                return
            if info["finished"]:
                self.fallback.append(info["finished"])
            agent = info["agent"]
            if agent in core.STATUS:
                self.agent = agent
                status = core.STATUS[agent]
                if agent == "Transcriber" and "first run" in text:
                    status = "Henter talemodellen første gang (kun én gang)…"
                self.status = status
                if agent in _SHARED:
                    p = _SHARED[agent]
                elif agent in _PER_REEL:
                    p = 35 + 65 * (self.reel + _PER_REEL[agent]) / max(1, self.count)
                else:
                    p = self.progress
                self.progress = max(self.progress, min(99.0, p))

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            return {
                "running": self.running, "stopped": self.stopped, "rc": self.rc,
                "agent": self.agent, "status": self.status, "progress": round(self.progress, 1),
                "reel": self.reel, "count": self.count, "seq": self.seq,
                "lines": [[s, t] for s, t in self.lines if s > since][-400:],
                "results": list(self.results), "error": "\n".join(self.error_lines),
                "elapsed": round(time.time() - self.started),
            }

    def stop(self) -> None:
        with self.lock:
            self.stopped = True
        if self.proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True,
                           creationflags=CREATE_NO_WINDOW)


JOB: Job | None = None


# --------------------------------------------------------------------------- helpers for the page

def pick_file(kind: str) -> str | list[str]:
    """Native Windows file dialog, run in a separate process so it never fights the server threads.

    kind "videos" lets the user pick several files at once and returns a list."""
    types = core.AUDIO_TYPES if kind == "music" else core.VIDEO_TYPES
    title = {"music": "Vælg musik", "videos": "Vælg en eller flere videoer"}.get(kind, "Vælg en video")
    if kind == "videos":
        ask = (f"p=filedialog.askopenfilenames(parent=r, title={title!r}, filetypes={types!r})\n"
               "out=[os.path.normpath(x) for x in p] if p else []\n")
    else:
        ask = (f"p=filedialog.askopenfilename(parent=r, title={title!r}, filetypes={types!r})\n"
               "out=os.path.normpath(p) if p else ''\n")
    code = (
        "import ctypes,json,os,sys,tkinter as tk\n"
        "from tkinter import filedialog\n"
        "try: ctypes.windll.shcore.SetProcessDpiAwareness(1)\n"
        "except Exception: pass\n"
        "r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True); r.update()\n"
        + ask +
        "r.destroy(); sys.stdout.write(json.dumps(out))\n"
    )
    proc = subprocess.run([core.python_for_subprocess(), "-c", code], capture_output=True, text=True,
                          encoding="utf-8", creationflags=CREATE_NO_WINDOW, timeout=600)
    empty = [] if kind == "videos" else ""
    try:
        got = json.loads(proc.stdout.strip() or "null")
    except ValueError:
        return empty
    if kind == "videos":
        return [str(x) for x in got] if isinstance(got, list) else empty
    return got if isinstance(got, str) else empty


def make_thumb(src: Path, fill: str = "blur") -> Path | None:
    """A 9:16 preview frame of the chosen video: blurred fill (talking videos) or full crop (montage)."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    key = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{fill}".encode("utf-8", "replace")).hexdigest()[:16]
    out = THUMBS / f"{key}.jpg"
    if out.exists():
        return out
    info = probe(src)
    if not info.has_video:
        return None
    t = min(1.5, max(0.0, info.duration / 3))
    if fill == "crop" and not info.is_vertical:
        t = min(max(5.0, info.duration * 0.2), max(0.0, info.duration - 1))   # past the intro, like the montage
    if info.is_vertical or fill == "crop":
        vf = "scale=540:960:force_original_aspect_ratio=increase,crop=540:960"
    else:
        vf = ("split=2[bg][fg];[bg]scale=135:240:force_original_aspect_ratio=increase,crop=135:240,"
              "boxblur=6:2,scale=540:960,eq=brightness=-0.07[b];"
              "[fg]scale=540:960:force_original_aspect_ratio=decrease[f];[b][f]overlay=(W-w)/2:(H-h)/2")
    run([find_tool("ffmpeg") or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
         "-frames:v", "1", "-vf", vf, "-q:v", "3", out], what="thumbnail", timeout=60)
    return out if out.exists() else None


def ai_enabled() -> bool:
    from .director import _have_credentials
    return _have_credentials()


KEYS_URL = "https://platform.claude.com/settings/keys"


def masked_key() -> str:
    k = os.environ.get("ANTHROPIC_API_KEY", "")
    return f"sk-ant-…{k[-4:]}" if len(k) > 12 else ""


def save_api_key(key: str | None) -> None:
    """Store the key as a Windows user environment variable (same as `setx`), without putting it on a
    command line, and use it right away in this window."""
    if os.name == "nt":
        import ctypes
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
            if key:
                winreg.SetValueEx(k, "ANTHROPIC_API_KEY", 0, winreg.REG_SZ, key)
            else:
                try:
                    winreg.DeleteValue(k, "ANTHROPIC_API_KEY")
                except FileNotFoundError:
                    pass
        res = ctypes.c_ulong()   # tell Windows so programs started later see the change
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 2000, ctypes.byref(res))
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)


# --------------------------------------------------------------------------- HTTP

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ReelMaker"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep quiet (pythonw has no console)
        pass

    # -- plumbing
    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 2_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            return {}

    def _guard(self, qs: dict) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):          # blocks DNS-rebinding tricks
            self._send(403, b"forbidden", "text/plain")
            return False
        if self.headers.get("X-Token") != TOKEN and qs.get("t", [""])[0] != TOKEN:
            self._send(403, b"forbidden", "text/plain")
            return False
        return True

    # -- routes
    def do_GET(self):
        STATE["last_seen"], STATE["bye_at"] = time.time(), None
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                host = (self.headers.get("Host") or "").split(":")[0]
                if host not in ("127.0.0.1", "localhost"):
                    return self._send(403, b"forbidden", "text/plain")
                html = HTML_FILE.read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            if url.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if not url.path.startswith("/api/") or not self._guard(qs):
                if not url.path.startswith("/api/"):
                    self._send(404, b"not found", "text/plain")
                return
            if url.path == "/api/state":
                return self._json({
                    "settings": core.load_settings(), "initial": STATE["initial"], "ai": ai_enabled(),
                    "ai_key": masked_key(),
                    "job": JOB.snapshot() if JOB else None, "recent": recent_outputs(),
                })
            if url.path == "/api/poll":
                since = int(qs.get("since", ["0"])[0] or 0)
                return self._json({"job": JOB.snapshot(since) if JOB else None})
            if url.path == "/api/recent":
                return self._json({"recent": recent_outputs()})
            if url.path == "/api/thumb":
                p = Path(qs.get("p", [""])[0])
                if not p.is_file():
                    return self._send(404, b"", "text/plain")
                thumb = make_thumb(p, "crop" if qs.get("fill", [""])[0] == "crop" else "blur")
                if not thumb:
                    return self._send(404, b"", "text/plain")
                return self._send(200, thumb.read_bytes(), "image/jpeg")
            if url.path == "/api/file":
                return self._serve_file(Path(qs.get("p", [""])[0]))
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            _log(traceback.format_exc())
            try:
                self._json({"error": str(e)}, 500)
            except OSError:
                pass

    def do_POST(self):
        global JOB
        STATE["last_seen"], STATE["bye_at"] = time.time(), None
        url = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(url.query)
        try:
            if not self._guard(qs):
                return
            if url.path == "/api/upload":
                return self._upload()
            data = self._body()
            if url.path == "/api/settings":
                core.save_settings({**core.DEFAULTS, **(data.get("opts") or {})})
                return self._json({"ok": True})
            if url.path == "/api/pick":
                got = pick_file(data.get("kind", "video"))
                return self._json({"paths": got} if isinstance(got, list) else {"path": got})
            if url.path == "/api/start":
                return self._start(data)
            if url.path == "/api/stop":
                if JOB and JOB.running:
                    JOB.stop()
                return self._json({"ok": True})
            if url.path == "/api/open":
                p = Path(data.get("path", ""))
                if p.exists() and _inside(p, OUTPUT_DIR):
                    os.startfile(str(p))  # type: ignore[attr-defined]
                    return self._json({"ok": True})
                return self._json({"error": "Ukendt sti"}, 400)
            if url.path == "/api/apikey":
                key = re.sub(r"\s+", "", data.get("key") or "")
                if not key.startswith("sk-ant-") or len(key) < 30:
                    return self._json({"ok": False, "message": "Det ligner ikke en API-nøgle. Den skal starte med sk-ant-"})
                from .director import test_claude
                works, valid, message = test_claude(key)
                if valid:
                    save_api_key(key)
                _log(f"api key test: works={works} valid={valid}")
                return self._json({"ok": works, "saved": valid, "message": message, "ai_key": masked_key()})
            if url.path == "/api/apikey/remove":
                save_api_key(None)
                return self._json({"ok": True})
            if url.path == "/api/keys-page":
                os.startfile(KEYS_URL)  # type: ignore[attr-defined]   # the user's normal browser (logged in)
                return self._json({"ok": True})
            if url.path == "/api/bye":
                STATE["bye_at"] = time.time()
                return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            _log(traceback.format_exc())
            try:
                self._json({"error": str(e)}, 500)
            except OSError:
                pass

    def _start(self, data: dict) -> None:
        global JOB
        if JOB and JOB.running:
            return self._json({"error": "Der bliver allerede lavet en reel."}, 409)
        raw = data.get("videos") or [data.get("video") or ""]
        videos = [str(v).strip().strip('"') for v in raw if str(v or "").strip()][:8]
        opts = {**core.DEFAULTS, **(data.get("opts") or {})}
        if not videos:
            return self._json({"error": "Vælg en video først."}, 400)
        for video in videos:
            if not re.match(r"^https?://", video, re.I) and not Path(video).is_file():
                return self._json({"error": f"Kan ikke finde filen:\n{video}"}, 400)
        music = (opts.get("music") or "").strip().strip('"')
        if music and not Path(music).is_file():
            return self._json({"error": f"Kan ikke finde musikfilen:\n{music}"}, 400)
        core.save_settings(opts)
        try:
            count = max(1, min(5, int(opts.get("count") or 1)))
        except (TypeError, ValueError):
            count = 1
        cmd = core.build_command(videos if len(videos) > 1 else videos[0], opts)
        _log("start: " + " ".join(cmd[3:]))
        JOB = Job(cmd, count)
        return self._json({"ok": True})

    def _upload(self) -> None:
        name = urllib.parse.unquote(self.headers.get("X-Filename") or "video.mp4")
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", Path(name).name).strip(" .") or "video.mp4"
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return self._json({"error": "Tom fil"}, 400)
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / name
        k = 2
        while dest.exists():                     # two different "clip.mp4" files must not overwrite each other
            dest = UPLOADS / f"{Path(name).stem}_{k}{Path(name).suffix}"
            k += 1
        tmp = dest.with_suffix(dest.suffix + ".part")
        left = n
        with tmp.open("wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            tmp.unlink(missing_ok=True)
            return self._json({"error": "Upload afbrudt"}, 400)
        os.replace(tmp, dest)
        return self._json({"path": str(dest)})

    def _serve_file(self, p: Path) -> None:
        if not p.is_file() or not _inside(p, OUTPUT_DIR, UPLOADS, THUMBS):
            return self._send(404, b"not found", "text/plain")
        size = p.stat().st_size
        ctype = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".srt": "text/plain; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".json": "application/json"}.get(p.suffix.lower(),
                                                                                     "application/octet-stream")
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:                                   # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with p.open("rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


# --------------------------------------------------------------------------- window

def _browser() -> str | None:
    cands = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    return next((c for c in cands if os.path.isfile(c)), shutil.which("msedge") or shutil.which("chrome"))


def _window_rect() -> tuple[int, int, int, int]:
    """Window size/position in screen units that fits above the taskbar (falls back to 1380x900)."""
    w, h, x, y = 1380, 900, 60, 20
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            r = wintypes.RECT()
            # this process is not DPI-aware, so the work area comes back in the same units Edge uses
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):
                aw, ah = r.right - r.left, r.bottom - r.top
                w, h = min(1400, aw - 40), min(920, ah - 24)
                x, y = r.left + (aw - w) // 2, r.top + max(8, (ah - h) // 2)
        except (AttributeError, OSError):
            pass
    return w, h, x, y


def open_window(url: str) -> subprocess.Popen | None:
    exe = _browser()
    if not exe:
        import webbrowser
        webbrowser.open(url)
        return None
    PROFILE.mkdir(parents=True, exist_ok=True)
    w, h, x, y = _window_rect()
    args = [exe, f"--app={url}", f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
            f"--window-size={w},{h}", f"--window-position={x},{y}",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-features=Translate,msEdgeSidebarV2,msUndersideButton"]
    return subprocess.Popen(args, creationflags=CREATE_NO_WINDOW)


def _watch_browser(proc: subprocess.Popen) -> None:
    t0 = time.time()
    proc.wait()
    if time.time() - t0 < 5:     # handed over to an already-open browser: rely on the idle timer instead
        return
    _log("window closed")
    STOP.set()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    from .utils import load_saved_api_key
    load_saved_api_key()
    STATE["initial"] = argv[0].strip().strip('"') if argv and argv[0].strip() else ""
    if "--no-window" in argv:
        STATE["initial"] = ""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("REELMAKER_PORT", "0"))
    srv = Server(("127.0.0.1", port), Handler)
    STATE["port"] = srv.server_address[1]
    url = f"http://127.0.0.1:{STATE['port']}/"
    _log(f"serving {url}")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if "--no-window" in argv:
        print(f"URL={url}?t={TOKEN}", flush=True)
    else:
        proc = open_window(url)
        if proc:
            threading.Thread(target=_watch_browser, args=(proc,), daemon=True).start()
    try:
        while not STOP.is_set():
            time.sleep(1)
            busy = JOB is not None and JOB.running
            now = time.time()
            if not busy and STATE["bye_at"] and now - STATE["bye_at"] > 4:
                break                                   # page closed (and not just reloaded)
            if not busy and now - STATE["last_seen"] > IDLE_EXIT:
                break
    except KeyboardInterrupt:
        pass
    if JOB and JOB.running:
        JOB.stop()
    srv.shutdown()
    _log("bye")


if __name__ == "__main__":
    main()
