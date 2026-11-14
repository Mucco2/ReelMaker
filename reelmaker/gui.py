"""ReelMaker - dansk vindue (tkinter).

Starter selve programmet som en underproces (python -m reelmaker make ...), viser loggen live,
en fremdriftsbjælke og knapper til de færdige reels.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = APP_DIR / "gui_settings.json"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
AGENT_RE = re.compile(r"^\[(\w+)\]\s?(.*)$")
TQDM_RE = re.compile(r"\d+%\|")
FINISHED_RE = re.compile(r"->\s*(.+?reel\.mp4)\s*$")

AGENTS = ["Fetcher", "Reference", "Transcriber", "Director", "Cutter", "Animator", "Finisher"]
STATUS = {
    "ReelMaker": "Starter…",
    "Fetcher": "Henter videoen…",
    "Reference": "Studerer reference-reelen…",
    "Transcriber": "Lytter til videoen…",
    "Director": "Vælger de bedste øjeblikke…",
    "Cutter": "Klipper…",
    "Animator": "Laver animationer…",
    "Finisher": "Gør lyden klar og eksporterer…",
}

STYLES = [("Fed", "bold"), ("Boks", "box"), ("Ren", "clean")]
PACINGS = [("Rolig", "chill"), ("Normal", "normal"), ("Hurtig", "fast")]
LANGS = [("Automatisk", ""), ("Dansk", "da"), ("Engelsk", "en"), ("Svensk", "sv"),
         ("Norsk", "no"), ("Tysk", "de"), ("Spansk", "es")]
LENGTHS = ["15", "30", "45", "60", "90"]
VIDEO_TYPES = [("Video", "*.mp4 *.mov *.mkv *.webm *.avi *.m4v *.wmv *.flv *.mts *.3gp"), ("Alle filer", "*.*")]
AUDIO_TYPES = [("Lyd", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac"), ("Alle filer", "*.*")]

DEFAULTS = {
    "mode": "auto", "fx": "insane", "prompt": "",
    "hook": "", "reference": "", "max_len": "60", "count": 1, "style": "bold", "accent": "",
    "pacing": "normal", "language": "", "music": "", "cta": "", "no_cta": False,
    "captions": True, "zoom": True, "progress": True, "fillers": True, "enhance": True,
}


# --------------------------------------------------------------------------- pure helpers (testable)

def python_for_subprocess() -> str:
    """The console python.exe to run the pipeline with (pythonw has no stdout)."""
    venv = APP_DIR / ".venv" / "Scripts" / "python.exe"
    if venv.exists():
        return str(venv)
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return str(exe)


def build_command(video: str | list[str], opts: dict, python: str | None = None) -> list[str]:
    """Turn the window's settings into the reelmaker command line (one video, or several to mix)."""
    o = {**DEFAULTS, **opts}
    videos = [video] if isinstance(video, str) else list(video)
    videos = [v.strip().strip('"') for v in videos if v and v.strip().strip('"')]
    cmd = [python or python_for_subprocess(), "-m", "reelmaker", "make", *videos]
    if o.get("mode") in ("speech", "montage"):
        cmd += ["--mode", o["mode"]]
    if o.get("fx") in ("insane", "classic"):
        cmd += ["--fx", o["fx"]]
    cmd += ["--text-language", "da"]          # Danish hook/post text for videos without speech
    if o["hook"].strip():
        cmd += ["--hook", o["hook"].strip()]
    if str(o.get("prompt") or "").strip():
        cmd += ["--prompt=" + " ".join(str(o["prompt"]).split())]   # "=" form: text may start with "-"
    if o["reference"].strip():
        cmd += ["--reference", o["reference"].strip().strip('"')]
    try:
        max_len = float(str(o["max_len"]).strip())
        if max_len > 0:
            cmd += ["--max-duration", f"{max_len:g}"]
    except ValueError:
        pass
    try:
        count = max(1, min(5, int(o["count"])))
    except (TypeError, ValueError):
        count = 1
    if count > 1:
        cmd += ["--count", str(count)]
    cmd += ["--caption-style", o["style"] or "bold"]
    if o["accent"]:
        cmd += ["--accent", o["accent"]]
    cmd += ["--pacing", o["pacing"] or "normal"]
    if o["language"]:
        cmd += ["--language", o["language"]]
    if o["music"].strip():
        cmd += ["--music", o["music"].strip().strip('"')]
    if o["no_cta"]:
        cmd += ["--no-cta"]
    elif o["cta"].strip():
        cmd += ["--cta", o["cta"].strip()]
    if not o["captions"]:
        cmd += ["--no-captions"]
    if not o["zoom"]:
        cmd += ["--no-zoom"]
    if not o["progress"]:
        cmd += ["--no-progress"]
    if not o["fillers"]:
        cmd += ["--keep-fillers"]
    if not o["enhance"]:
        cmd += ["--no-enhance"]
    return cmd


def parse_line(line: str) -> dict:
    """Classify one output line: {'text', 'agent', 'result', 'finished', 'error', 'progressbar'}."""
    text = ANSI_RE.sub("", line).rstrip("\r\n")
    info = {"text": text, "agent": None, "result": None, "finished": None, "error": False,
            "progressbar": bool(TQDM_RE.search(text))}
    stripped = text.strip()
    if stripped.startswith("RESULT_DIR="):
        info["result"] = stripped.split("=", 1)[1].strip()
        return info
    if stripped.startswith("ERROR:"):
        info["error"] = True
        return info
    m = AGENT_RE.match(stripped)
    if m:
        info["agent"] = m.group(1)
        f = FINISHED_RE.search(stripped)
        if f and m.group(1) == "Finisher":
            info["finished"] = str(Path(f.group(1)).parent)
    return info


def progress_for(agent: str | None) -> float | None:
    """0-100 for the agent that is working now."""
    if agent not in AGENTS:
        return None
    return (AGENTS.index(agent) + 0.5) / len(AGENTS) * 100.0


def read_post_text(folder: str | Path) -> str:
    """Caption + hashtags from post.txt (the part before the line of dashes)."""
    try:
        text = (Path(folder) / "post.txt").read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = []
    for ln in text.splitlines():
        if re.fullmatch(r"\s*-{10,}\s*", ln):
            break
        lines.append(ln)
    body = "\n".join(lines)
    if "CAPTION:" in body:
        body = body.split("CAPTION:", 1)[1]
    return body.strip()


def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    except (OSError, ValueError, AttributeError):
        return dict(DEFAULTS)


NOT_REMEMBERED = {"hook", "prompt"}   # they belong to one video, so they are not carried over to the next


def save_settings(opts: dict) -> None:
    try:
        keep = {k: opts.get(k, v) for k, v in DEFAULTS.items() if k not in NOT_REMEMBERED}
        SETTINGS_FILE.write_text(json.dumps(keep, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _label(pairs: list[tuple[str, str]], code: str) -> str:
    return next((lab for lab, c in pairs if c == code), pairs[0][0])


def _code(pairs: list[tuple[str, str]], label: str) -> str:
    return next((c for lab, c in pairs if lab == label), pairs[0][1])


# --------------------------------------------------------------------------- the window

class ReelMakerApp:
    def __init__(self, root: tk.Tk, initial_video: str = ""):
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.stopped = False
        self.results: list[str] = []
        self.fallback_results: list[str] = []
        self.error_lines: list[str] = []
        self.count = 1
        self.last_was_bar = False

        s = load_settings()
        root.title("ReelMaker")
        # size in "96-dpi pixels", scaled to the screen and never taller/wider than it
        k = max(1.0, root.winfo_fpixels("1i") / 96.0)
        w = min(int(900 * k), root.winfo_screenwidth() - 40)
        h = min(int(780 * k), root.winfo_screenheight() - int(90 * k))
        root.geometry(f"{w}x{h}+{max(0, (root.winfo_screenwidth() - w) // 2)}+{int(10 * k)}")
        root.minsize(min(int(720 * k), w), min(int(560 * k), h))
        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20))
        style.configure("Sub.TLabel", font=("Segoe UI", 10), foreground="#555555")
        style.configure("Big.TButton", font=("Segoe UI Semibold", 12), padding=(18, 8))
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

        self.v_video = tk.StringVar(value=initial_video)
        self.v_hook = tk.StringVar(value=s["hook"])
        self.v_ref = tk.StringVar(value=s["reference"])
        self.v_len = tk.StringVar(value=str(s["max_len"]))
        self.v_count = tk.StringVar(value=str(s["count"]))
        self.v_style = tk.StringVar(value=_label(STYLES, s["style"]))
        self.accent = s["accent"] or ""
        self.v_pacing = tk.StringVar(value=_label(PACINGS, s["pacing"]))
        self.v_lang = tk.StringVar(value=_label(LANGS, s["language"]))
        self.v_music = tk.StringVar(value=s["music"])
        self.v_cta = tk.StringVar(value=s["cta"])
        self.v_no_cta = tk.BooleanVar(value=s["no_cta"])
        self.v_captions = tk.BooleanVar(value=s["captions"])
        self.v_zoom = tk.BooleanVar(value=s["zoom"])
        self.v_progress = tk.BooleanVar(value=s["progress"])
        self.v_fillers = tk.BooleanVar(value=s["fillers"])
        self.v_enhance = tk.BooleanVar(value=s["enhance"])
        self.v_status = tk.StringVar(value="Klar. Vælg en video og tryk på “Lav reel”.")

        self._build()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        head = ttk.Frame(main)
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text="ReelMaker", style="Title.TLabel").pack(anchor="w")
        ttk.Label(head, text="Gør din video klar til TikTok, Instagram Reels, YouTube Shorts og Facebook.",
                  style="Sub.TLabel").pack(anchor="w")

        vid = ttk.LabelFrame(main, text="Video", padding=10)
        vid.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        vid.columnconfigure(1, weight=1)
        ttk.Label(vid, text="Video (fil eller link)").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(vid, textvariable=self.v_video).grid(row=0, column=1, sticky="ew")
        ttk.Button(vid, text="Vælg fil…", command=self._pick_video).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(vid, text="Hook-tekst (valgfri)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(vid, textvariable=self.v_hook).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(vid, text="Reference-reel link (valgfri)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(vid, textvariable=self.v_ref).grid(row=2, column=1, columnspan=2, sticky="ew", pady=(6, 0))

        opt = ttk.LabelFrame(main, text="Indstillinger", padding=10)
        opt.grid(row=2, column=0, sticky="ew", pady=6)
        for c in (1, 3):
            opt.columnconfigure(c, weight=1)

        ttk.Label(opt, text="Maks længde (sek.)").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(opt, textvariable=self.v_len, values=LENGTHS, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(opt, text="Antal reels").grid(row=0, column=2, sticky="w", padx=(16, 8))
        ttk.Spinbox(opt, from_=1, to=5, textvariable=self.v_count, width=6, state="readonly").grid(
            row=0, column=3, sticky="w")

        ttk.Label(opt, text="Undertekst-stil").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.v_style, values=[l for l, _ in STYLES], state="readonly",
                     width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Farve").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=(6, 0))
        col = ttk.Frame(opt)
        col.grid(row=1, column=3, sticky="w", pady=(6, 0))
        self.swatch = tk.Label(col, width=10, relief="groove", borderwidth=1)
        self.swatch.pack(side="left")
        ttk.Button(col, text="Vælg…", command=self._pick_color).pack(side="left", padx=(6, 0))
        ttk.Button(col, text="Nulstil", command=self._reset_color).pack(side="left", padx=(4, 0))
        self._show_color()

        ttk.Label(opt, text="Tempo").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.v_pacing, values=[l for l, _ in PACINGS], state="readonly",
                     width=12).grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Sprog").grid(row=2, column=2, sticky="w", padx=(16, 8), pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.v_lang, values=[l for l, _ in LANGS], state="readonly",
                     width=14).grid(row=2, column=3, sticky="w", pady=(6, 0))

        ttk.Label(opt, text="Musik (valgfri)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        mus = ttk.Frame(opt)
        mus.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        mus.columnconfigure(0, weight=1)
        ttk.Entry(mus, textvariable=self.v_music).grid(row=0, column=0, sticky="ew")
        ttk.Button(mus, text="Vælg…", command=self._pick_music).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(opt, text="Tekst til sidst (call-to-action)").grid(row=4, column=0, sticky="w", pady=(6, 0))
        cta = ttk.Frame(opt)
        cta.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        cta.columnconfigure(0, weight=1)
        ttk.Entry(cta, textvariable=self.v_cta).grid(row=0, column=0, sticky="ew")
        ttk.Checkbutton(cta, text="Ingen call-to-action", variable=self.v_no_cta).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(opt, text="Tom = automatisk tekst på videoens sprog", style="Sub.TLabel").grid(
            row=5, column=1, columnspan=3, sticky="w")

        checks = ttk.Frame(opt)
        checks.grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))
        for text, var in (("Undertekster", self.v_captions), ("Zoom på klip", self.v_zoom),
                          ("Fremdriftsbjælke", self.v_progress), ("Fjern 'øh' og fyldord", self.v_fillers),
                          ("Farveforbedring", self.v_enhance)):
            ttk.Checkbutton(checks, text=text, variable=var).pack(side="left", padx=(0, 14))

        run = ttk.Frame(main)
        run.grid(row=3, column=0, sticky="ew", pady=(8, 4))
        run.columnconfigure(2, weight=1)
        self.btn_make = ttk.Button(run, text="Lav reel", style="Big.TButton", command=self._start)
        self.btn_make.grid(row=0, column=0)
        self.btn_stop = ttk.Button(run, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=(8, 0))
        self.bar = ttk.Progressbar(run, mode="determinate", maximum=100)
        self.bar.grid(row=0, column=2, sticky="ew", padx=(14, 0))
        ttk.Label(main, textvariable=self.v_status, style="Status.TLabel").grid(row=4, column=0, sticky="w")

        self.res = ttk.LabelFrame(main, text="Resultat", padding=8)
        self.res.grid(row=5, column=0, sticky="ew", pady=6)
        self._clear_results()

        logf = ttk.LabelFrame(main, text="Log", padding=6)
        logf.grid(row=6, column=0, sticky="nsew", pady=(4, 0))
        main.rowconfigure(6, weight=1)
        self.log = ScrolledText(logf, height=9, wrap="word", font=("Consolas", 9), state="disabled",
                                background="#1e1e1e", foreground="#dddddd", insertbackground="#dddddd")
        self.log.pack(fill="both", expand=True)

    # ---------------------------------------------------------------- small actions
    def _pick_video(self) -> None:
        p = filedialog.askopenfilename(title="Vælg en video", filetypes=VIDEO_TYPES)
        if p:
            self.v_video.set(os.path.normpath(p))

    def _pick_music(self) -> None:
        p = filedialog.askopenfilename(title="Vælg musik", filetypes=AUDIO_TYPES)
        if p:
            self.v_music.set(os.path.normpath(p))

    def _pick_color(self) -> None:
        _, hexcol = colorchooser.askcolor(color=self.accent or "#FFE135", title="Vælg fremhævningsfarve")
        if hexcol:
            self.accent = hexcol.upper()
            self._show_color()

    def _reset_color(self) -> None:
        self.accent = ""
        self._show_color()

    def _show_color(self) -> None:
        if self.accent:
            self.swatch.configure(background=self.accent, text="")
        else:
            self.swatch.configure(background=self.root.cget("background"), text="Standard")

    def _opts(self) -> dict:
        return {
            "hook": self.v_hook.get(), "reference": self.v_ref.get(), "max_len": self.v_len.get(),
            "count": self.v_count.get(), "style": _code(STYLES, self.v_style.get()), "accent": self.accent,
            "pacing": _code(PACINGS, self.v_pacing.get()), "language": _code(LANGS, self.v_lang.get()),
            "music": self.v_music.get(), "cta": self.v_cta.get(), "no_cta": self.v_no_cta.get(),
            "captions": self.v_captions.get(), "zoom": self.v_zoom.get(), "progress": self.v_progress.get(),
            "fillers": self.v_fillers.get(), "enhance": self.v_enhance.get(),
        }

    def _log(self, text: str, replace_last: bool = False) -> None:
        self.log.configure(state="normal")
        if replace_last:
            self.log.delete("end-2l linestart", "end-1c")
        self.log.insert("end", text + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 5000:
            self.log.delete("1.0", f"{lines - 4000}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------------------------------------------------------- results
    def _clear_results(self) -> None:
        for w in self.res.winfo_children():
            w.destroy()
        ttk.Label(self.res, text="Her kommer dine færdige reels.", style="Sub.TLabel").pack(anchor="w")

    def _add_result(self, folder: str) -> None:
        if len(self.results) == 1:
            for w in self.res.winfo_children():
                w.destroy()
        row = ttk.Frame(self.res)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=f"Reel {len(self.results)}:  {Path(folder).name}").pack(side="left")
        ttk.Button(row, text="Kopiér opslagstekst", command=lambda f=folder: self._copy_post(f)).pack(side="right")
        ttk.Button(row, text="Åbn mappe", command=lambda f=folder: self._open(f)).pack(side="right", padx=4)
        ttk.Button(row, text="Afspil", command=lambda f=folder: self._open(Path(f) / "reel.mp4")).pack(side="right")

    def _open(self, path) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as e:
            messagebox.showerror("Kunne ikke åbne", f"{path}\n\n{e}")

    def _copy_post(self, folder: str) -> None:
        text = read_post_text(folder)
        if not text:
            messagebox.showinfo("Ingen tekst", "Fandt ingen opslagstekst (post.txt).")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.v_status.set("Opslagsteksten er kopieret – indsæt den i din app.")

    # ---------------------------------------------------------------- running
    def _start(self) -> None:
        video = self.v_video.get().strip().strip('"')
        if not video:
            messagebox.showwarning("Mangler video", "Indsæt en video-fil eller et link først.")
            return
        if not re.match(r"^https?://", video, re.I) and not Path(video).exists():
            messagebox.showwarning("Filen findes ikke", f"Kan ikke finde filen:\n{video}")
            return
        music = self.v_music.get().strip().strip('"')
        if music and not Path(music).exists():
            messagebox.showwarning("Musik findes ikke", f"Kan ikke finde musikfilen:\n{music}")
            return
        opts = self._opts()
        save_settings(opts)
        cmd = build_command(video, opts)
        try:
            self.count = max(1, min(5, int(opts["count"])))
        except ValueError:
            self.count = 1

        self.results, self.fallback_results, self.error_lines = [], [], []
        self.stopped, self.last_was_bar = False, False
        self._clear_results()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.bar["value"] = 0
        self.v_status.set("Starter…")
        self.btn_make.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        env = os.environ.copy()
        env.update({"NO_COLOR": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(APP_DIR), env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        except OSError as e:
            self._finish_ui()
            messagebox.showerror("Kunne ikke starte", f"ReelMaker kunne ikke starte:\n\n{e}\n\nKør setup.bat.")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.q.put(("line", line))
        self.q.put(("done", proc.wait()))

    def _poll(self) -> None:
        try:
            while True:
                kind, value = self.q.get_nowait()
                if kind == "line":
                    self._handle_line(value)
                else:
                    self._done(value)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle_line(self, line: str) -> None:
        info = parse_line(line)
        if not info["text"].strip():
            return
        self._log(info["text"], replace_last=info["progressbar"] and self.last_was_bar)
        self.last_was_bar = info["progressbar"]
        if self.error_lines or info["error"]:
            if len(self.error_lines) < 8:
                self.error_lines.append(info["text"])
        if info["result"]:
            self.results.append(info["result"])
            self._add_result(info["result"])
            self.bar["value"] = 100
            return
        if info["finished"]:
            self.fallback_results.append(info["finished"])
        agent = info["agent"]
        if agent in STATUS:
            status = STATUS[agent]
            if agent == "Transcriber" and "first run" in info["text"]:
                status += " (første gang hentes talemodellen – det kan tage et par minutter)"
            if self.count > 1:
                status = f"Reel {min(len(self.results) + 1, self.count)} af {self.count} – {status}"
            self.v_status.set(status)
            p = progress_for(agent)
            if p is not None:
                self.bar["value"] = p

    def _done(self, rc: int) -> None:
        self.proc = None
        self._finish_ui()
        if not self.results:
            for f in self.fallback_results:
                self.results.append(f)
                self._add_result(f)
        if self.stopped:
            self.v_status.set("Stoppet.")
            self.bar["value"] = 0
            return
        if rc == 0 and self.results:
            self.bar["value"] = 100
            n = len(self.results)
            self.v_status.set("Færdig! Din reel er klar." if n == 1 else f"Færdig! {n} reels er klar.")
            return
        self.bar["value"] = 0
        self.v_status.set("Noget gik galt – se loggen.")
        msg = "\n".join(self.error_lines).replace("ERROR:", "").strip() or f"Programmet stoppede (kode {rc})."
        messagebox.showerror("Noget gik galt", f"Reelen kunne ikke laves.\n\n{msg}\n\nSe loggen for detaljer.")

    def _finish_ui(self) -> None:
        self.btn_make.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except OSError:
                self.proc.kill()

    def _stop(self) -> None:
        self.stopped = True
        self.v_status.set("Stopper…")
        self._kill()

    def _on_close(self) -> None:
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Luk ReelMaker", "En reel er ved at blive lavet. Stop og luk?"):
                return
            self.stopped = True
            self._kill()
        save_settings(self._opts())
        self.root.destroy()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    initial = argv[0].strip().strip('"') if argv and argv[0].strip() else ""
    if os.name == "nt":
        try:  # crisp text on scaled (125%/150%) laptop screens instead of a blurry stretched window
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    ReelMakerApp(root, initial)
    root.mainloop()


if __name__ == "__main__":
    main()
