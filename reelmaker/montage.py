"""Montage mode - for videos without speech (cars, sport, travel, gaming, music videos).

The Director uses this when Whisper hears (almost) no words:
1. Scans the whole video: loudness (engines revving, crowds), motion and brightness, plus shot changes.
2. Finds the beat of the background music and builds a cut grid on it, so every cut lands on a beat.
3. Picks the most exciting, well-spread moments (best one first as the hook). With a Claude API key,
   Claude looks at frames of the candidate moments and chooses + writes the hook and post text.
"""

from __future__ import annotations

import base64
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import FPS, Settings
from .director import EditPlan, _FALLBACK_MODELS, _have_credentials
from .utils import MediaInfo, Stopwatch, ffmpeg, log, run, tool_env, warn

AGENT = "Director"
GRID = 0.25            # analysis resolution in seconds

LANG_NAMES = {"da": "Danish", "en": "English", "sv": "Swedish", "no": "Norwegian", "de": "German",
              "es": "Spanish", "fr": "French", "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "pl": "Polish"}


# --------------------------------------------------------------------------- media decoding

def _pcm(path: Path, sr: int, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    cmd = [ffmpeg(), "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, env=tool_env(), timeout=900,
                          creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _rms_curve(y: np.ndarray, sr: int, hop_s: float) -> np.ndarray:
    hop = max(1, int(sr * hop_s))
    n = len(y) // hop
    if n == 0:
        return np.zeros(0)
    frames = y[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


# --------------------------------------------------------------------------- video analysis

def analyze(info: MediaInfo, work: Path) -> dict:
    """Per-0.25s motion, brightness and loudness, plus shot boundaries. Cached per video."""
    cache = work / "montage_analysis_v2.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            pass
    sw = Stopwatch()
    log(AGENT, "No speech - scanning the whole video for the best moments (motion, sound, shot changes)...")
    n = max(1, int(math.ceil(info.duration / GRID)))
    motion = np.zeros(n)
    bright = np.full(n, 100.0)
    proc = run([ffmpeg(), "-hide_banner", "-nostats", "-i", info.path, "-an",
                # format=yuv420p: 8-bit brightness values even for 10-bit (iPhone HDR) sources
                "-vf", f"fps={1 / GRID:g},scale=160:-2,format=yuv420p,signalstats,select='gte(scene,0)',"
                       f"metadata=print:file=-",
                "-f", "null", "-"], what="video analysis", timeout=3600)
    t = None
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        if t is None:
            continue
        i = min(n - 1, int(round(t / GRID)))
        if "lavfi.scene_score=" in line:
            motion[i] = float(line.split("=", 1)[1])
        elif "lavfi.signalstats.YAVG=" in line:
            bright[i] = float(line.split("=", 1)[1])
    loud = np.zeros(n)
    if info.has_audio:
        y = _pcm(info.path, 8000)
        rms = _rms_curve(y, 8000, GRID)
        loud[: min(n, len(rms))] = 20 * np.log10(rms[:n] + 1e-6)
        if len(rms) < n:
            loud[len(rms):] = loud[: len(rms)].min() if len(rms) else -60
    # shot boundaries = big jumps in the scene score
    shots = []
    for i in range(1, n):
        if motion[i] > 0.32 and (not shots or i * GRID - shots[-1] > 0.6):
            shots.append(round(i * GRID, 2))
    data = {"grid": GRID, "duration": info.duration, "motion": motion.round(4).tolist(),
            "bright": bright.round(1).tolist(), "loud": loud.round(2).tolist(), "shots": shots,
            "has_audio": info.has_audio}
    cache.write_text(json.dumps(data), encoding="utf-8")
    log(AGENT, f"Scanned {info.duration:.0f}s of video: {len(shots)} shot changes ({sw.lap()})")
    return data


def _z(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def excitement(an: dict) -> np.ndarray:
    """One 'how exciting is this moment' number per 0.25s."""
    motion = np.array(an["motion"], dtype=float)
    loud = np.array(an["loud"], dtype=float)
    bright = np.array(an["bright"], dtype=float)
    n = len(motion)

    def smooth(x: np.ndarray) -> np.ndarray:
        return np.convolve(x, np.ones(4) / 4, mode="same") if len(x) >= 4 else x

    # motion: ignore hard cuts (handled separately), care about sustained movement
    mot = np.minimum(motion, 0.3)
    score = 0.45 * _z(smooth(mot))
    if an.get("has_audio") and len(loud) == n and loud.std() > 0.5:
        score += 0.55 * _z(smooth(loud))
    score -= np.where(bright < 28, 2.5, 0.0)                     # black / fade frames
    score -= np.where(bright > 245, 1.0, 0.0)                    # white flashes
    return score


# --------------------------------------------------------------------------- music beat grid

MIN_BEAT_STRENGTH = 0.12   # normalised autocorrelation needed to trust a tempo (noise/ambient falls below)
ONSET_LATENCY = 0.020      # spectral flux peaks ~20 ms before the hit; calibrated on kick/click tracks 85-174 BPM


def _onset_envelope(y: np.ndarray, sr: int, hop: int = 512, n_fft: int = 2048):
    """Spectral-flux onset strength. Frames are CENTRED (padding), so frame i is at time i*hop/sr."""
    y = np.concatenate([np.zeros(n_fft // 2, dtype=y.dtype), y, np.zeros(n_fft // 2, dtype=y.dtype)])
    n = 1 + (len(y) - n_fft) // hop
    if n < 16:
        return None, hop / sr
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx].astype(np.float32) * np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    logm = np.log1p(10.0 * mag)
    flux = np.concatenate([[0.0], np.maximum(0.0, np.diff(logm, axis=0)).sum(axis=1)])
    k = max(3, int(0.4 * sr / hop))
    trend = np.convolve(flux, np.ones(k) / k, mode="same")
    return np.maximum(0.0, flux - trend), hop / sr


def beat_grid(music: Path, start: float, dur: float) -> tuple[float, float] | None:
    """Returns (beat period seconds, offset of the first beat after `start`) or None."""
    sr = 22050
    y = _pcm(music, sr, start, dur)
    env, dt = _onset_envelope(y, sr)
    if env is None or env.max() <= 0:
        return None
    es = np.convolve(env, [0.25, 0.5, 0.25], mode="same")   # smoothing keeps one tempo peak per lag
    e = es - es.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    if ac[0] <= 0:
        return None
    lags = np.arange(len(ac))
    lo, hi = int(60 / 200 / dt), int(60 / 60 / dt)
    if hi >= len(ac) or lo < 1:
        return None
    cand = lags[lo:hi]
    bpm = 60.0 / (cand * dt)
    prior = np.exp(-0.5 * (np.log2(bpm / 120.0) / 0.8) ** 2)   # prefer common tempos around 120 BPM
    best = cand[int(np.argmax(ac[lo:hi] * prior))]
    if ac[best] / ac[0] < MIN_BEAT_STRENGTH:                    # no steady pulse (noise, ambient, speech)
        return None
    if 1 <= best < len(ac) - 1:                                  # parabolic refinement
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    period = (best + shift) * dt
    bpmv = 60 / period
    while bpmv < 80:
        period /= 2
        bpmv *= 2
    while bpmv > 175:
        period *= 2
        bpmv /= 2
    # Joint fine search of tempo (+-2%) and phase: the grid that lands on the most onset energy wins.
    best = (-1.0, period, 0.0)
    for cand_p in period * (1 + np.linspace(-0.02, 0.02, 81)):
        lagp = cand_p / dt
        nb = int(len(env) / lagp)
        if nb < 2:
            continue
        offs = np.arange(0, lagp, 0.25)
        idx = np.minimum((offs[:, None] + lagp * np.arange(nb)[None, :]).round().astype(int), len(env) - 1)
        sums = env[idx].sum(axis=1)
        i = int(np.argmax(sums))
        if sums[i] > best[0]:
            best = (float(sums[i]), float(cand_p), float(offs[i] * dt))
    _, period, offset = best

    # Refine: fit a straight line through the real onset peaks near each predicted beat (two passes).
    nz = env[env > 0]
    strong = float(np.median(nz)) if len(nz) else 0.0
    for _ in range(2):
        ks, ts = [], []
        for k in range(int((len(env) * dt - offset) / period) + 1):
            b_t = offset + k * period
            i0, i1 = max(0, int((b_t - 0.08) / dt)), min(len(env), int((b_t + 0.08) / dt) + 1)
            if i1 - i0 < 3:
                continue
            j = i0 + int(np.argmax(env[i0:i1]))
            if env[j] <= strong or j == i0 or j == i1 - 1:      # weak, or not a real peak inside the window
                continue
            a_, m_, c_ = env[j - 1], env[j], env[j + 1]
            den = a_ - 2 * m_ + c_
            ks.append(k)
            ts.append((j + (0.5 * (a_ - c_) / den if den != 0 else 0.0)) * dt)
        if len(ks) < 6:
            break
        slope, intercept = np.polyfit(np.array(ks, float), np.array(ts, float), 1)
        if abs(slope - period) / period >= 0.03:
            break
        period, offset = float(slope), float(intercept)
    offset = (offset + ONSET_LATENCY) % period
    return period, offset


def music_start(music: Path, length: float) -> float:
    """Start the song at its most energetic stretch (skips quiet intros)."""
    y = _pcm(music, 4000)
    if len(y) == 0:
        return 0.0
    total = len(y) / 4000
    if total <= length + 1:
        return 0.0
    rms = _rms_curve(y, 4000, 0.5)
    win = max(1, int(length / 0.5))
    sums = np.convolve(rms, np.ones(win), mode="valid")
    starts = np.arange(len(sums)) * 0.5
    sums = sums - 0.002 * starts * sums.mean()      # tiny preference for earlier starts
    return float(starts[int(np.argmax(sums))])


# --------------------------------------------------------------------------- selection

def _window_means(score: np.ndarray, length: float) -> np.ndarray:
    w = max(1, int(round(length / GRID)))
    if len(score) < w:
        return np.array([score.mean()]) if len(score) else np.zeros(1)
    c = np.concatenate([[0.0], np.cumsum(score)])
    return (c[w:] - c[:-w]) / w


def _windows(an: dict, score: np.ndarray, length: float, lo: float, hi: float) -> list[tuple[float, float]]:
    """(start, score) for every possible window of `length` inside [lo, hi], penalising mid-clip shot cuts."""
    means = _window_means(score, length)
    shots = np.array(an["shots"])
    out = []
    for i, m in enumerate(means):
        t0 = i * GRID
        if t0 < lo or t0 + length > hi:
            continue
        if len(shots):
            inside = shots[(shots > t0 + 0.2 * length) & (shots < t0 + 0.8 * length)]
            if len(inside):
                m -= 0.35
        out.append((t0, float(m)))
    return out


def _pick(wins: list[tuple[float, float]], length: float, n: int, taken: list[tuple[float, float]],
          spread: float) -> list[tuple[float, float]]:
    """Greedy best-first selection: no overlaps, spread out over the video."""
    chosen: list[tuple[float, float]] = []
    order = sorted(wins, key=lambda w: -w[1])
    for sp in (spread, spread / 2, 0.0):
        for t0, sc in order:
            if len(chosen) >= n:
                break
            a, b = t0, t0 + length
            if any(a < tb + 0.3 and b > ta - 0.3 for ta, tb in taken + [(c, c + length) for c, _ in chosen]):
                continue
            mid = (a + b) / 2
            if sp and any(abs(mid - (c + length / 2)) < sp for c, _ in chosen):
                continue
            chosen.append((t0, sc))
        if len(chosen) >= n:
            break
    return chosen


def _slots(settings: Settings, target: float, usable: float, music: Path | None) -> dict:
    """Clip lengths that add up to ~target and fit the video. With music every cut lands on a beat.

    Returns {slots, start, note, period, loop}: `start` = where the song starts (on a beat), `loop` = a
    whole number of beats to loop if the song is shorter than the reel (0 = no beat-exact loop)."""
    beats_per_clip = {"fast": 2, "normal": 4, "chill": 8}.get(settings.pacing, 4)
    out = {"slots": [], "start": 0.0, "note": "", "period": 0.0, "loop": 0.0}
    if music:
        from .utils import probe
        mlen = probe(music).duration
        out["start"] = music_start(music, target + 2)
        grid = beat_grid(music, out["start"], min(target + 8, 90))
        if grid:
            period, offset = grid
            out["start"] += offset                  # beat 1 of the reel = first frame
            L = beats_per_clip * period
            while L < 0.7:
                L *= 2
            while L > 5.0:
                L /= 2
            while L > usable and L / 2 >= period * 0.99:   # short video: shorter clips (still whole beats)
                L /= 2
            hook = 2 * L if L < 2.0 else L
            if hook > usable:
                hook = L
            if hook > usable:                        # shorter than one beat: just use what there is
                out["slots"] = [usable]
                return out
            slots = [hook]
            while sum(slots) + L <= target + 1e-6:
                slots.append(L)
            n_beats = int((mlen - out["start"]) / period) if mlen > 0 else 0
            if n_beats >= 8:
                n_beats -= n_beats % 4          # loop whole bars so beat 1 stays beat 1
            out.update(slots=slots, period=period, loop=n_beats * period if n_beats >= 4 else 0.0,
                       note=f"{60 / period:.0f} BPM, cuts every {L:.2f}s on the beat")
            return out
        warn(AGENT, "No clear beat in the music - cutting at a steady pace instead")
    L = min({"fast": 1.2, "normal": 2.0, "chill": 3.2}.get(settings.pacing, 2.0), usable)
    slots = [min(L * 1.5, usable)]
    while sum(slots) + L <= target + 1e-6:
        slots.append(L)
    out["slots"] = slots
    return out


def _frame_snap(slots: list[float]) -> list[int]:
    """Whole-frame clip lengths whose running total never drifts off the beat grid."""
    cuts = np.concatenate([[0.0], np.cumsum(slots)])
    frames = np.round(cuts * FPS).astype(int)
    return [max(1, int(b - a)) for a, b in zip(frames[:-1], frames[1:])]


# --------------------------------------------------------------------------- Claude looks at the frames

def _frames_b64(src: Path, times: list[float]) -> list[str]:
    def grab(t: float) -> str:
        proc = subprocess.run([ffmpeg(), "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                               "-vf", "scale=384:-2", "-q:v", "5", "-f", "image2", "-c:v", "mjpeg", "-"],
                              capture_output=True, env=tool_env(), timeout=60,
                              creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        return base64.b64encode(proc.stdout).decode("ascii") if proc.returncode == 0 and proc.stdout else ""
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(grab, times))


_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reels": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "moments": {"type": "array", "items": {"type": "integer"}},
                "hook": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "hashtags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["moments", "hook", "title", "description", "hashtags"],
            "additionalProperties": False,
        }},
    },
    "required": ["reels"],
    "additionalProperties": False,
}

_VISION_SYSTEM = """You are a top short-form video editor making fast, exciting vertical montages for TikTok, Reels and Shorts.
You get numbered still frames, each from a short candidate clip of a video without speech (cars, sport, travel, etc.),
with its time and an excitement score from audio/motion analysis. Build the reel(s):
- moments: the clip numbers to use, in play order. The FIRST one is the hook: the most striking, eye-catching shot.
  After the hook, mostly keep the original time order so the story flows. Use exactly the number of clips asked for
  when there are enough good ones. Skip frames that look like intros, logos, title cards, black/blurry frames,
  talking-to-camera filler or repeated near-identical shots.
- Different reels must not share clips.
- hook: max 6 words, punchy on-screen title, no emojis/hashtags, in the requested language. title: max 80 characters.
  description: 1-2 sentences. hashtags: 5-8 relevant ones starting with #. All text in the requested language."""


def _claude_pick(src: Path, cands: list[tuple[float, float]], length: float, n_slots: int, count: int,
                 settings: Settings, name_hint: str) -> list[dict] | None:
    if settings.ai == "off" or (settings.ai == "auto" and not _have_credentials()):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    times = [t0 + length / 2 for t0, _ in cands]
    images = _frames_b64(src, times)
    content: list[dict] = []
    for i, ((t0, sc), img) in enumerate(zip(cands, images)):
        if not img:
            continue
        content.append({"type": "text", "text": f"[{i}] t={t0 + length / 2:.1f}s score={sc:+.2f}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
    if not content:
        return None
    lang = LANG_NAMES.get((settings.text_language or "en").lower(), "English")
    content.append({"type": "text", "text": (
        f"Video file name: {name_hint}\nMake {count} reel{'s' if count > 1 else ''} with {n_slots} clips each. "
        f"Write all text in {lang}.")})
    kwargs = dict(model=settings.ai_model, max_tokens=16000, system=_VISION_SYSTEM,
                  messages=[{"role": "user", "content": content}],
                  output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _VISION_SCHEMA}})
    if settings.ai_model in _FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    log(AGENT, f"Claude is looking at {len(content) // 2} moments to pick the best ones...")
    try:
        resp = anthropic.Anthropic().beta.messages.create(**kwargs)
    except Exception as e:   # any API problem -> the built-in picker takes over
        warn(AGENT, f"Claude unavailable ({type(e).__name__}) - picking moments with the built-in rules")
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        reels = json.loads(text).get("reels") or []
    except (ValueError, AttributeError):
        return None
    return [r for r in reels if isinstance(r, dict) and r.get("moments")] or None


# --------------------------------------------------------------------------- text without AI

def _name_text(stem: str) -> tuple[str, list[str]]:
    name = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", stem)              # (1080p, h264) / [id]
    name = re.sub(r"_[A-Za-z0-9_-]{11}$", "", name)                # yt-dlp id suffix
    name = re.split(r"\s[-|–]\s", name.replace("_", " "))[0]       # " - Channel"
    name = re.sub(r"\s+", " ", name).strip()
    words = [w for w in re.findall(r"[A-Za-zÆØÅæøåÄÖÜäöüß0-9]+", name) if len(w) >= 3 and not w.isdigit()]
    from .director import _STOPWORDS
    skip = _STOPWORDS | {"official", "video", "performance", "old", "new", "full", "best", "part", "edit",
                         "1080p", "720p", "4k", "hd", "h264", "school"}
    seen, tags = set(), []
    for w in words:
        k = w.lower()
        if k in seen or k in skip:
            continue
        seen.add(k)
        tags.append("#" + k)
    return name[:80], tags[:5]


# --------------------------------------------------------------------------- main entry

def plan_montage(info: MediaInfo, settings: Settings, work: Path, count: int = 1) -> list[EditPlan]:
    if settings.fx == "insane":
        from .hype import plan_hype
        return plan_hype(info, settings, work, count)
    an = analyze(info, work)
    score = excitement(an)
    dur = max(0.1, info.duration)
    # skip channel intros/outros on longer videos; short clips are used whole
    intro = min(4.0, dur * 0.03) if dur > 12 else 0.0
    outro = min(6.0, dur * 0.05) if dur > 12 else 0.0
    lo = intro
    hi = min(dur, max(intro + 1.0, dur - outro))
    usable = max(0.1, hi - lo)
    max_len = settings.effective_max()
    count = max(1, min(int(count), 5))
    # each reel gets its own share of the material, so --count works on shorter videos too
    per_reel = usable * 0.75 / count
    target = min(max_len, max(min(usable / count, 6.0), per_reel))
    music = Path(settings.music) if settings.music else None

    grid = _slots(settings, target, usable, music)
    slots = grid["slots"]
    frames = _frame_snap(slots)
    L = slots[1] if len(slots) > 1 else slots[0]
    n = len(slots)
    log(AGENT, f"Montage: {n} clips, {sum(frames) / FPS:.1f}s" + (f" ({grid['note']})" if grid["note"] else ""))

    spread = usable / (n * 2.2)
    taken: list[tuple[float, float]] = []      # source ranges used by earlier reels
    plans: list[EditPlan] = []

    def interval(centre: float, d: float) -> tuple[float, float]:
        s = min(max(lo, centre - d / 2), max(lo, hi - d))
        return s, s + d

    def free(a: float, b: float, own: list[tuple[float, float]]) -> bool:
        return not any(a < tb + 0.3 and b > ta - 0.3 for ta, tb in taken + own)

    # Candidate pool for Claude: the best non-overlapping windows over the whole video.
    pool_n = min(40, max(n * count + 8, int(n * count * 1.5)))
    pool = sorted(_pick(_windows(an, score, L, lo, hi), L, pool_n, [], spread / 3), key=lambda w: w[0])
    title_hint, name_tags = _name_text(info.path.stem)
    ai_reels = _claude_pick(info.path, pool, L, n, count, settings, title_hint) if pool else None

    for k in range(count):
        picks: list[tuple[float, float]] = []  # real source ranges, in play order; slot j has frames[j] frames
        ai = ai_reels[k] if ai_reels and k < len(ai_reels) else None
        if ai and not isinstance(ai.get("moments"), list):
            ai = None
        if ai:
            for i in ai["moments"]:
                if len(picks) >= n:
                    break
                if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(pool):
                    a, b = interval(pool[i][0] + L / 2, frames[len(picks)] / FPS)
                    if free(a, b, picks):
                        picks.append((a, b))
        if len(picks) < n:
            # built-in picker: best moment first (hook), then the rest in time order
            if not picks:
                d = frames[0] / FPS
                hw = _pick(_windows(an, score, d, lo, hi), d, 1, taken, 0)
                if hw:
                    picks.append(interval(hw[0][0] + d / 2, d))
                elif not taken:                 # tiny video: the whole thing is the one clip
                    picks.append(interval((lo + hi) / 2, d))
            if picks:
                rest = _pick(_windows(an, score, L, lo, hi), L, n - len(picks), taken + picks, spread)
                for t0, _ in sorted(rest, key=lambda w: w[0]):
                    if len(picks) >= n:
                        break
                    a, b = interval(t0 + L / 2, frames[len(picks)] / FPS)
                    if free(a, b, picks):
                        picks.append((a, b))
        if not picks:
            break
        if k > 0 and len(picks) < min(n, max(3, n // 2)):
            break                                   # not enough fresh material for another good reel
        segments = []
        for j, (a, _b) in enumerate(picks):
            s = round(a * FPS) / FPS
            e = s + frames[j] / FPS
            segments.append((round(s, 4), round(e, 4)))
            taken.append((s, e))
        title, tags = title_hint, name_tags + ["#fyp", "#viral", "#reels", "#shorts"]
        hook = settings.hook if k == 0 else None
        plan = EditPlan(segments=segments, hook=hook, title=title, description=title, hashtags=tags,
                        decided_by="Claude" if ai else "rules", mode="montage",
                        music_start=round(grid["start"], 4), music_loop=round(grid["loop"], 4))
        if ai:
            plan.hook = hook or (ai.get("hook") or "").strip() or None
            plan.title = (ai.get("title") or title).strip()
            plan.description = (ai.get("description") or title).strip()
            if ai.get("hashtags"):
                plan.hashtags = [t if t.startswith("#") else "#" + t
                                 for t in (str(x).strip().replace(" ", "") for x in ai["hashtags"]) if t]
        log(AGENT, f"Montage reel {k + 1}: {len(segments)} clips, {plan.duration:.1f}s, decided by {plan.decided_by}"
                   + (f' | hook "{plan.hook}"' if plan.hook else ""))
        plans.append(plan)
    if count > 1 and len(plans) < count:
        warn(AGENT, f"The video only has material for {len(plans)} good reel(s), not {count}.")
    return plans
