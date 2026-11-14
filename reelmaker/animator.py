"""Animator agent - animated captions, hook title, progress bar and call-to-action.

Main path: builds a HyperFrames (HTML + GSAP) composition and renders it as a TRANSPARENT video
layer that the Finisher lays over the edited footage.
Backup path: if HyperFrames is unavailable it writes styled karaoke subtitles (.ass) that FFmpeg burns in.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import FPS, HEIGHT, WIDTH, Settings
from .cutter import Timeline
from .director import EditPlan, clean_word, is_filler
from .transcribe import all_words
from .utils import ENGINE_DIR, PipelineError, Stopwatch, log, probe, run, warn

AGENT = "Animator"

HOOK_Y, CAPTION_Y, CTA_Y = 300, 1150, 400
# TikTok / Reels / Shorts put like-comment-share buttons on the right edge: keep captions out of it.
SAFE_X = 130
SAFE_W = WIDTH - 2 * SAFE_X


@dataclass
class CapWord:
    text: str
    start: float
    end: float
    emph: bool


@dataclass
class Chunk:
    words: list[CapWord]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


# --------------------------------------------------------------------------- caption grouping

def _display(word: str, upper: bool) -> str:
    w = word.strip()
    w = re.sub(r"^[\"'“”‘’«»(\[]+", "", w)
    w = re.sub(r"[\"'“”‘’«»)\],.;:…]+$", "", w)
    return w.upper() if upper else w


def build_chunks(transcript: dict, plan: EditPlan, tl: Timeline, settings: Settings) -> list[Chunk]:
    style = settings.caption_params()
    max_words = 3 if style["upper"] else 4
    max_chars = 16 if style["upper"] else 22
    mapped: list[tuple[CapWord, str]] = []
    if plan.mode == "montage":      # the few stray words would land on unrelated clips
        return []
    for w in all_words(transcript):
        if settings.remove_fillers and is_filler(w, settings.speech_language):
            continue
        r = tl.map_range(w["start"], w["end"])
        if not r:
            continue
        text = _display(w["word"], style["upper"])
        if not text:
            continue
        mapped.append((CapWord(text, round(r[0], 3), round(max(r[1], r[0] + 0.08), 3),
                               clean_word(w["word"]) in plan.emphasis), w["word"]))
    mapped.sort(key=lambda m: m[0].start)

    chunks: list[Chunk] = []
    cur: list[CapWord] = []
    prev_raw = ""
    for cw, raw in mapped:
        chars = sum(len(x.text) + 1 for x in cur) + len(cw.text)
        new = (not cur or len(cur) >= max_words or chars > max_chars
               or cw.start - cur[-1].end > 0.6 or prev_raw.rstrip().endswith((".", "?", "!", ",", ";", ":")))
        if new and cur:
            chunks.append(Chunk(cur, cur[0].start, cur[-1].end))
            cur = []
        cur.append(cw)
        prev_raw = raw
    if cur:
        chunks.append(Chunk(cur, cur[0].start, cur[-1].end))

    # timing polish: stay on screen a bit after the last word, never overlap the next chunk
    for c in chunks:
        c.start = round(max(0.0, c.start - 0.03), 3)
    for i, c in enumerate(chunks):
        nxt = chunks[i + 1].start if i + 1 < len(chunks) else tl.duration
        c.end = round(min(max(c.end + 0.35, c.start + 0.4), nxt, tl.duration), 3)
    return [c for c in chunks if c.end - c.start > 0.05]


# --------------------------------------------------------------------------- HyperFrames

_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=__W__, height=__H__" />
<script src="gsap.min.js"></script>
<style>
  html, body { margin: 0; padding: 0; background: transparent; }
  #root { position: relative; width: __W__px; height: __H__px; overflow: hidden; }
  #layer { position: absolute; inset: 0; }
  #caps { position: absolute; left: __SAFEX__px; right: __SAFEX__px; top: __CAPY__px; height: 0; }
  .cap {
    position: absolute; left: 0; right: 0; transform: translateY(-50%);
    text-align: center; line-height: 1.12; opacity: 0; visibility: hidden;
    font-family: '__FONT__', 'Arial Black', sans-serif; font-weight: __WEIGHT__;
    color: __COLOR__; letter-spacing: 0.5px;
  }
  .w { display: inline-block; padding: 0 0.12em; margin: 0.04em 0; border-radius: 0.18em;
       -webkit-text-stroke: __STROKE__px #000; paint-order: stroke fill;
       text-shadow: 0 6px 16px rgba(0,0,0,0.55); }
  .w.em { color: __ACCENT__; }
  #hook { position: absolute; left: 60px; right: 60px; top: __HOOKY__px; text-align: center;
          opacity: 0; visibility: hidden; }
  #hook span { display: inline-block; background: #FFFFFF; color: #111111; padding: 18px 34px;
               border-radius: 22px; font-family: 'Montserrat', 'Arial Black', sans-serif; font-weight: 900;
               font-size: __HOOKSIZE__px; line-height: 1.12; box-shadow: 0 14px 40px rgba(0,0,0,0.45);
               transform: rotate(-2deg); }
  #hook.big span { background: none; box-shadow: none; padding: 0; border-radius: 0; transform: none;
                   font-size: inherit; color: inherit; }
  #hook.big #hooktext { display: block; color: #FFFFFF; font-family: 'Montserrat', 'Arial Black', sans-serif;
                        font-weight: 900; font-style: italic; font-size: __HOOKBIG__px; line-height: 1.04;
                        letter-spacing: -1px; -webkit-text-stroke: 12px #000; paint-order: stroke fill;
                        text-shadow: 0 0 34px rgba(0,0,0,0.65), 0 8px 0 rgba(0,0,0,0.5); }
  #hook.big .hw { display: inline-block; }
  #hook.big .hw.last { color: __ACCENT__; }
  #cta { position: absolute; left: 0; right: 0; top: __CTAY__px; text-align: center; opacity: 0; visibility: hidden; }
  #cta span { display: inline-block; background: __ACCENT__; color: #111111; padding: 16px 36px; border-radius: 999px;
              font-family: 'Montserrat', 'Arial Black', sans-serif; font-weight: 900; font-size: 58px;
              box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
  #bar { position: absolute; left: 0; top: 0; height: 12px; width: 100%; background: __ACCENT__;
         transform-origin: 0 50%; transform: scaleX(0); box-shadow: 0 0 12px rgba(0,0,0,0.35); }
</style>
</head>
<body>
<div id="root" data-composition-id="main" data-start="0" data-duration="__D__"
     data-width="__W__" data-height="__H__" data-fps="__FPS__">
  <div id="layer" class="clip" data-start="0" data-duration="__D__" data-track-index="1">
    <div id="bar"></div>
    <div id="hook"><span id="hooktext"></span></div>
    <div id="caps"></div>
    <div id="cta"><span id="ctatext"></span></div>
  </div>
</div>
<script>
  const DATA = __DATA__;
  window.__timelines = window.__timelines || {};
  const tl = gsap.timeline({ paused: true });
  const base = DATA.color, accent = DATA.accent, box = DATA.box;

  if (DATA.bar) {
    tl.fromTo("#bar", { scaleX: 0 }, { scaleX: 1, duration: DATA.D, ease: "none" }, 0);
  } else {
    document.getElementById("bar").remove();
  }

  if (DATA.hook && DATA.hook.big) {
    document.getElementById("hook").classList.add("big");
    const ht = document.getElementById("hooktext");
    const words = DATA.hook.text.split(/\\s+/);
    tl.set("#hook", { autoAlpha: 1 }, DATA.hook.s);
    words.forEach((w, i) => {
      const sp = document.createElement("span");
      sp.className = "hw" + (i === words.length - 1 && words.length > 1 ? " last" : "");
      sp.textContent = w;
      ht.appendChild(sp);
      ht.appendChild(document.createTextNode(" "));
      const t = (DATA.hook.times && DATA.hook.times[i] !== undefined) ? DATA.hook.times[i] : DATA.hook.s;
      tl.fromTo(sp, { autoAlpha: 0, scale: 2.4, y: -24, rotation: i % 2 ? 4 : -4 },
                { autoAlpha: 1, scale: 1, y: 0, rotation: 0, duration: 0.16, ease: "power3.out" }, t);
    });
    tl.to("#hook", { scale: 1.7, autoAlpha: 0, duration: 0.14, ease: "power2.in" }, Math.max(0, DATA.hook.e - 0.04));
  } else if (DATA.hook) {
    document.getElementById("hooktext").textContent = DATA.hook.text;
    tl.fromTo("#hook", { autoAlpha: 0, scale: 0.4, y: -40 },
              { autoAlpha: 1, scale: 1, y: 0, duration: 0.45, ease: "back.out(2.4)" }, DATA.hook.s);
    tl.to("#hook", { scale: 1.06, duration: 0.18, yoyo: true, repeat: 1, ease: "power1.inOut" }, DATA.hook.s + 1.1);
    tl.to("#hook", { autoAlpha: 0, scale: 0.85, duration: 0.25, ease: "power2.in" }, DATA.hook.e - 0.25);
  }

  if (DATA.cta) {
    document.getElementById("ctatext").textContent = DATA.cta.text;
    tl.fromTo("#cta", { autoAlpha: 0, y: 60, scale: 0.7 },
              { autoAlpha: 1, y: 0, scale: 1, duration: 0.4, ease: "back.out(2)" }, DATA.cta.s);
    tl.to("#cta span", { scale: 1.08, duration: 0.3, yoyo: true, repeat: 3, ease: "sine.inOut" }, DATA.cta.s + 0.5);
  }

  const caps = document.getElementById("caps");
  DATA.chunks.forEach((c, ci) => {
    const el = document.createElement("div");
    el.className = "cap";
    el.style.fontSize = c.size + "px";
    const spans = c.w.map((w) => {
      const sp = document.createElement("span");
      sp.className = "w" + (w.x ? " em" : "");
      sp.textContent = w.t;
      el.appendChild(sp);
      el.appendChild(document.createTextNode(" "));
      return sp;
    });
    caps.appendChild(el);
    tl.fromTo(el, { autoAlpha: 0, scale: 0.8, y: 20 },
              { autoAlpha: 1, scale: 1, y: 0, duration: 0.14, ease: "back.out(2.2)" }, c.s);
    c.w.forEach((w, wi) => {
      const sp = spans[wi];
      const rest = w.x ? accent : base;
      if (box) {
        tl.fromTo(sp, { backgroundColor: "rgba(0,0,0,0)", scale: 1 },
                  { backgroundColor: accent, scale: 1.08, duration: 0.07 }, w.s);
        tl.to(sp, { backgroundColor: "rgba(0,0,0,0)", scale: 1, duration: 0.07 }, Math.max(w.s + 0.07, w.e));
      } else {
        tl.fromTo(sp, { color: rest, scale: 1 }, { color: accent, scale: 1.12, duration: 0.07 }, w.s);
        tl.to(sp, { color: rest, scale: 1, duration: 0.07 }, Math.max(w.s + 0.07, w.e));
      }
    });
    tl.to(el, { autoAlpha: 0, duration: 0.05 }, Math.max(c.s, c.e - 0.05));
  });

  window.__timelines.main = tl;
</script>
</body>
</html>
"""


def _hyperframes_cli() -> list[str]:
    node = shutil.which("node")
    pkg = ENGINE_DIR / "node_modules" / "hyperframes" / "package.json"
    if not node or not pkg.exists():
        raise PipelineError("HyperFrames is not installed (run setup.bat).")
    meta = json.loads(pkg.read_text(encoding="utf-8"))
    binf = meta.get("bin")
    if isinstance(binf, dict):
        binf = binf.get("hyperframes") or next(iter(binf.values()))
    if not binf:
        raise PipelineError("Could not find the HyperFrames CLI entry point.")
    return [node, str((pkg.parent / binf).resolve())]


def _font_size(chunk: Chunk, base: int, upper: bool = True) -> int:
    """Shrink long caption groups so they fit the safe zone on one line (same rule as the UI preview)."""
    avg = 0.70 if upper else 0.56          # average glyph width in em for the bold / clean fonts
    fit = int(SAFE_W / (max(1, len(chunk.text)) * avg))
    return max(int(base * 0.6), min(base, fit))


def _timed_overlays(plan: EditPlan, settings: Settings, D: float) -> tuple[dict | None, dict | None]:
    hook = None
    if plan.hook:
        hook = {"text": plan.hook.upper(), "s": 0.05, "e": round(min(3.2, max(1.5, D * 0.35)), 3)}
        if plan.shots:                    # insane montage: big title, one word per beat, gone on the drop
            hook["big"] = True
            if plan.hook_end:
                hook["e"] = round(plan.hook_end, 3)
            words = hook["text"].split()
            beats = sorted(b for b in plan.hook_beats if b < hook["e"] - 0.35) or [0.0]
            if len(words) > len(beats) and len(beats) > 1:          # not enough beats: use half-beats
                step = beats[1] - beats[0]
                beats = sorted(set(beats + [round(b + step / 2, 3) for b in beats if b + step / 2 < hook["e"] - 0.35]))
            times = [max(0.05, beats[min(i, len(beats) - 1)]) for i in range(len(words))]
            hook["times"] = [round(t, 3) for t in times]
    cta = None
    if settings.cta and D >= 8:
        s = round(max(D - 2.6, (hook["e"] + 0.5) if hook else 0), 3)
        if D - s >= 1.2:
            cta = {"text": settings.cta.upper(), "s": s, "e": D}
    return hook, cta


def render_hyperframes(chunks: list[Chunk], plan: EditPlan, settings: Settings, D: float, work: Path) -> Path:
    sw = Stopwatch()
    style = settings.caption_params()
    proj = work / "overlay"
    if proj.exists():
        shutil.rmtree(proj, ignore_errors=True)
    proj.mkdir(parents=True)
    gsap = ENGINE_DIR / "node_modules" / "gsap" / "dist" / "gsap.min.js"
    if not gsap.exists():
        raise PipelineError("gsap is missing from engine/node_modules (run setup.bat).")
    shutil.copy(gsap, proj / "gsap.min.js")

    hook, cta = _timed_overlays(plan, settings, D)
    data = {
        "D": round(D, 3), "color": style["color"], "accent": style["accent"], "box": style["box"],
        "bar": settings.progress_bar, "hook": hook, "cta": cta,
        "chunks": [{"s": c.start, "e": c.end, "size": _font_size(c, style["size"], style["upper"]),
                    "w": [{"t": w.text, "s": w.start, "e": w.end, "x": w.emph} for w in c.words]}
                   for c in chunks] if settings.captions else [],
    }
    hook_size = 84 if not hook or len(hook["text"]) <= 22 else (70 if len(hook["text"]) <= 40 else 58)
    html = (_HTML
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__W__", str(WIDTH)).replace("__H__", str(HEIGHT)).replace("__FPS__", str(FPS))
            .replace("__D__", f"{D:.3f}").replace("__CAPY__", str(CAPTION_Y)).replace("__SAFEX__", str(SAFE_X))
            .replace("__HOOKY__", str(HOOK_Y)).replace("__CTAY__", str(CTA_Y))
            .replace("__HOOKSIZE__", str(hook_size))
            .replace("__HOOKBIG__", str(118 if not hook or len(hook["text"]) <= 20
                                        else 100 if len(hook["text"]) <= 34 else 82))
            .replace("__FONT__", style["font"]).replace("__WEIGHT__", str(style["weight"]))
            .replace("__COLOR__", style["color"]).replace("__ACCENT__", style["accent"])
            .replace("__STROKE__", str(style["stroke"])))
    (proj / "index.html").write_text(html, encoding="utf-8")

    out = work / "overlay.webm"
    if out.exists():
        out.unlink()
    log(AGENT, f"Rendering animations with HyperFrames ({len(chunks)} caption groups, {D:.1f}s)...")
    run(_hyperframes_cli() + ["render", str(proj), "--format", "webm", "--fps", str(FPS),
                              "--output", str(out), "--quiet"],
        cwd=ENGINE_DIR, timeout=max(600, D * 30), what="HyperFrames render")
    if not out.exists() or out.stat().st_size < 1000:
        raise PipelineError("HyperFrames did not produce an overlay video.")
    got = probe(out)
    if abs(got.duration - D) > 0.5:
        warn(AGENT, f"Overlay length {got.duration:.2f}s differs from video {D:.2f}s")
    log(AGENT, f"Animation layer ready ({sw.lap()})")
    return out


# --------------------------------------------------------------------------- ASS backup

def _ass_color(hex_color: str, alpha: int = 0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_ass(chunks: list[Chunk], plan: EditPlan, settings: Settings, D: float, path: Path) -> Path:
    style = settings.caption_params()
    base, acc = _ass_color(style["color"]), _ass_color(style["accent"])
    outline = max(4, style["stroke"] - 3)
    cap_size = int(style["size"] * 1.3)   # libass sizes by ascent+descent, so it needs a bigger number than CSS
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {WIDTH}", f"PlayResY: {HEIGHT}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,Arial Black,{cap_size},{base},{base},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},3,5,{SAFE_X},{SAFE_X},0,1",
        "Style: Hook,Arial Black,72,&H00111111,&H00111111,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,-2,3,18,0,5,80,80,0,1",
        f"Style: Cta,Arial Black,54,&H00111111,&H00111111,{acc},{acc},-1,0,0,0,100,100,0,0,3,16,0,5,80,80,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hook, cta = _timed_overlays(plan, settings, D)
    if hook:
        lines.append(f"Dialogue: 2,{_ass_time(hook['s'])},{_ass_time(hook['e'])},Hook,,0,0,0,,"
                     f"{{\\pos({WIDTH // 2},{HOOK_Y + 60})\\fscx40\\fscy40\\t(0,350,\\fscx100\\fscy100)\\fad(0,250)}}"
                     f"{_ass_escape(hook['text'])}")
    if cta:
        lines.append(f"Dialogue: 2,{_ass_time(cta['s'])},{_ass_time(cta['e'])},Cta,,0,0,0,,"
                     f"{{\\pos({WIDTH // 2},{CTA_Y + 40})\\fad(250,0)}}{_ass_escape(cta['text'])}")
    if settings.captions:
        for c in chunks:
            size = int(_font_size(c, style["size"], style["upper"]) * 1.3)
            for i, w in enumerate(c.words):
                s = c.start if i == 0 else w.start
                e = c.words[i + 1].start if i + 1 < len(c.words) else c.end
                if e <= s:
                    continue
                parts = []
                for j, x in enumerate(c.words):
                    col = acc if (j == i or x.emph) else base
                    tag = f"{{\\c{col}" + ("\\fscx110\\fscy110" if j == i else "\\fscx100\\fscy100") + "}"
                    parts.append(tag + _ass_escape(x.text))
                pop = "\\fscx80\\fscy80\\t(0,120,\\fscx100\\fscy100)" if i == 0 else ""
                lines.append(f"Dialogue: 1,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,"
                             f"{{\\pos({WIDTH // 2},{CAPTION_Y})\\fs{size}{pop}}}" + " ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def write_srt(chunks: list[Chunk], path: Path) -> Path:
    def ts(t: float) -> str:
        ms = int(round(max(0.0, t) * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    out = []
    for i, c in enumerate(chunks, 1):
        out += [str(i), f"{ts(c.start)} --> {ts(c.end)}", " ".join(w.text for w in c.words), ""]
    path.write_text("\n".join(out), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- main entry

@dataclass
class OverlayResult:
    kind: str            # "hyperframes" | "ass" | "none"
    path: Path | None
    chunks: list[Chunk]


def animate(transcript: dict, plan: EditPlan, tl: Timeline, settings: Settings, work: Path) -> OverlayResult:
    chunks = build_chunks(transcript, plan, tl, settings)
    D = tl.duration
    nothing = not settings.captions and not plan.hook and not settings.cta and not settings.progress_bar
    if nothing:
        return OverlayResult("none", None, chunks)
    if settings.animator == "hyperframes":
        try:
            return OverlayResult("hyperframes", render_hyperframes(chunks, plan, settings, D, work), chunks)
        except PipelineError as e:
            warn(AGENT, f"HyperFrames failed, switching to backup subtitles.\n{e}")
    ass = write_ass(chunks, plan, settings, D, work / "captions.ass")
    log(AGENT, "Wrote styled subtitles (backup renderer)")
    return OverlayResult("ass", ass, chunks)
