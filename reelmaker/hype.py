"""Hype planner - the recipe for the "insane" montage edit.

1. Listens to the song: beat grid, bass energy per beat, the DROPS (where the bass kicks back in) and every bass hit.
2. Starts the song so the reel opens with a short build-up and the first drop lands after ~2-4 seconds (the hook).
3. Writes the edit like a pro editor would: slow-motion intro under the title, accelerating cuts with strobe and a
   riser into the drop, then an impact (flash, shake, RGB split, boom) and a hype section with cut patterns that
   change every bar, speed ramps, close-ups, whip/zoom transitions and zoom pulses on every bass hit.
4. Picks the footage for every shot (Claude looks at the frames and also says where to zoom in).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import FPS, Settings
from .director import EditPlan, _FALLBACK_MODELS, _have_credentials
from .fx import Shot, source_span
from .montage import (GRID, LANG_NAMES, _frames_b64, _name_text, _pcm, _pick, _windows, analyze, beat_grid,
                      excitement)
from .utils import MediaInfo, log, probe, warn

AGENT = "Director"
MIN_CLIP = 0.27            # shortest clip in seconds (half-beats get merged below this)


# --------------------------------------------------------------------------- the song

@dataclass
class Song:
    period: float                       # seconds per beat
    offset: float                       # first beat (song seconds)
    duration: float
    bass: np.ndarray                    # bass level per beat (dB), beat k = offset + k*period
    drops: list[float] = field(default_factory=list)          # song seconds where the bass slams back in
    hits: list[tuple[float, float]] = field(default_factory=list)  # (song seconds, strength 0-1)

    @property
    def high_ref(self) -> float:
        return float(np.percentile(self.bass, 75)) if len(self.bass) else 0.0

    def beat_at(self, t: float) -> int:
        return int(np.clip(np.floor((t - self.offset) / self.period), 0, max(0, len(self.bass) - 1)))

    def is_low(self, t: float) -> bool:
        return bool(len(self.bass)) and self.bass[self.beat_at(t)] < self.high_ref - 7.0


def _bass_curves(y: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> tuple[np.ndarray, float]:
    y = np.concatenate([np.zeros(n_fft // 2, np.float32), y, np.zeros(n_fft // 2, np.float32)])
    n = 1 + (len(y) - n_fft) // hop
    if n < 8:
        return np.zeros(0), hop / sr
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = freqs < 150
    bass = np.zeros(n, np.float32)
    for a in range(0, n, 2000):
        b = min(n, a + 2000)
        idx = np.arange(n_fft)[None, :] + hop * np.arange(a, b)[:, None]
        bass[a:b] = np.abs(np.fft.rfft(y[idx] * win, axis=1))[:, band].sum(axis=1)
    return bass, hop / sr


def analyze_song(music: Path) -> Song | None:
    total = probe(music).duration
    grid = beat_grid(music, 0.0, min(total, 150.0))
    if not grid:
        return None
    period, offset = grid
    sr = 11025
    y = _pcm(music, sr)
    bass, dt = _bass_curves(y, sr)
    if len(bass) == 0:
        return None
    nb = max(1, int((len(y) / sr - offset) / period))
    per_beat = np.zeros(nb)
    for k in range(nb):
        a, b = int((offset + k * period) / dt), int((offset + (k + 1) * period) / dt)
        seg = bass[a:max(a + 1, b)]
        per_beat[k] = 20 * np.log10(float(seg.mean()) + 1e-6) if len(seg) else -60
    song = Song(period=period, offset=offset, duration=len(y) / sr, bass=per_beat)

    # drops: the bass comes back hard after a quieter stretch
    hi = song.high_ref
    cands = []
    for k in range(1, nb - 2):
        after = per_beat[k:k + 4].mean()
        before = per_beat[max(0, k - 4):k].mean()
        if after >= hi - 5 and after - before >= 7:
            cands.append((after - before, k))
    chosen: list[int] = []
    for _, k in sorted(cands, reverse=True):
        if all(abs(k - c) >= 8 for c in chosen):
            chosen.append(k)
    song.drops = sorted(offset + k * period for k in chosen)

    # bass hits: sharp rises of the bass level (kicks / 808s)
    logb = np.log1p(bass / (np.percentile(bass, 90) + 1e-9) * 20)
    env = np.maximum(0.0, np.diff(logb, prepend=logb[0]))
    w = max(1, int(0.09 / dt))
    ref = float(np.percentile(env[env > 0], 97)) if np.any(env > 0) else 0.0
    if ref > 0:
        for i in range(1, len(env) - 1):
            v = env[i]
            if v < ref * 0.35 or v < env[max(0, i - w):i + w + 1].max():
                continue
            t = i * dt - 0.01
            if song.hits and t - song.hits[-1][0] < 0.16:
                continue
            song.hits.append((t, float(min(1.0, v / ref))))
    return song


def choose_start(song: Song, length: float) -> float:
    """Song second (on a beat) where the reel begins: a short build-up, then a drop after ~2-4 s."""
    P = song.period
    nb_reel = max(1, int(length / P))
    best_t, best_s = None, -1e9
    span = max(1e-6, song.high_ref - (song.high_ref - 12))
    levels = np.clip((song.bass - (song.high_ref - 12)) / span, 0, 1)
    drops = [round((d - song.offset) / P) for d in song.drops]
    for b0 in range(0, len(song.bass)):
        t0 = song.offset + b0 * P
        if t0 + length > song.duration - 0.5 and b0 > 0:
            break
        win = levels[b0:b0 + nb_reel]
        score = float(win.mean()) if len(win) else 0.0
        bonus = 0.0
        for d in drops:
            lead = (d - b0) * P
            if 1.6 <= lead <= 4.8:
                v = 1.0
            elif 0.9 <= lead <= 7.0:
                v = 0.55
            else:
                continue
            if (d - b0) % 4 == 0:
                v += 0.25
            bonus = max(bonus, v)
        score += 1.2 * bonus - 0.0015 * t0
        if score > best_s:
            best_t, best_s = t0, score
    if best_t is None:
        return 0.0
    return best_t


# --------------------------------------------------------------------------- the edit recipe

BUILD = {
    "fast":   {4: [2, 1, 0.5, 0.5], 3: [1, 1, 0.5, 0.5], 2: [1, 0.5, 0.5], 1: [0.5, 0.5]},
    "normal": {4: [2, 1, 0.5, 0.5], 3: [1, 1, 0.5, 0.5], 2: [1, 0.5, 0.5], 1: [0.5, 0.5]},
    "chill":  {4: [2, 2], 3: [2, 1], 2: [2], 1: [1]},
}
PATTERNS = {
    "fast":   [[1, 1, 1, 1], [0.5, 0.5, 1, 1, 1], [1, 1, 2], [1, 1, 0.5, 0.5, 1], [2, 1, 1], [1, 1, 1, 1]],
    "normal": [[1, 1, 2], [1, 1, 1, 1], [2, 1, 1], [0.5, 0.5, 1, 2], [1, 1, 1, 1], [2, 2]],
    "chill":  [[2, 2], [2, 1, 1], [4], [1, 1, 2]],
}
HIGH_FRAMES = ["wide", "tight", "pan", "medium", "push", "wide", "tight", "pull"]
CALM_FRAMES = ["push", "pan", "pull", "medium"]
BUILD_FRAMES = ["tight", "wide", "medium", "tight", "wide"]
BAR_TRANSITIONS = ["zoom", "whip", "cut", "flash", "zoom", "whip", "glitch"]


@dataclass
class Slot:
    start: float            # in units
    length: float           # in units
    section: str            # calm | build | drop | high
    phrase: bool = False    # first clip of a 4-bar phrase
    bar_start: bool = False


def _split_halves(pattern: list[float], half_ok: bool) -> list[float]:
    if half_ok:
        return list(pattern)
    out: list[float] = []
    for x in pattern:
        if x < 1:
            if out and out[-1] == 0.5:
                out[-1] = 1.0
            else:
                out.append(0.5)
        else:
            out.append(x)
    return [1.0 if x == 0.5 else x for x in out]


def build_slots(labels: list[str], drops: list[int], NU: int, pacing: str, half_ok: bool) -> list[Slot]:
    """Cut lengths (in beats or half-time beats) following the music's sections."""
    pacing = pacing if pacing in PATTERNS else "normal"
    patterns = PATTERNS[pacing]
    drops = sorted(d for d in drops if 0 < d < NU)
    anchor = drops[0] if drops else 0
    slots: list[Slot] = []
    j = 0.0
    bar = 0
    while j < NU - 1e-6:
        ji = int(j)
        nd = next((d for d in drops if d >= j - 1e-6), None)
        if nd is not None and abs(j - nd) < 1e-6:                     # the drop: impact clip
            L = min(2.0, NU - j)
            slots.append(Slot(j, L, "drop", phrase=True, bar_start=True))
            j += L
            bar = 0
            continue
        if nd is not None and 0 < nd - j <= 4:                       # build-up into the drop
            for L in _split_halves(BUILD[pacing][int(round(nd - j)) if nd - j >= 1 else 1], half_ok):
                L = min(L, nd - j)
                if L <= 0:
                    break
                slots.append(Slot(j, L, "build"))
                j += L
            j = float(nd)
            continue
        limit = float(NU)
        if nd is not None:
            limit = min(limit, nd - 4.0)
        if labels[min(ji, NU - 1)] == "low":                         # calm, slow motion
            end = ji + 1
            while end < limit and labels[end] == "low":
                end += 1
            L = min(4.0 if pacing != "fast" else 2.0, max(1.0, end - j), NU - j)
            slots.append(Slot(j, L, "calm"))
            j += L
            continue
        end = ji + 1
        while end < limit and labels[end] != "low":
            end += 1
        end = max(end, ji + 1)
        phase = (j - anchor) % 4
        if phase > 1e-6:                                             # get onto the bar grid first
            L = min(4 - phase, 1.0, end - j)
            slots.append(Slot(j, L if L >= 0.5 else end - j, "high"))
            j += slots[-1].length
            continue
        pat = _split_halves(patterns[bar % len(patterns)], half_ok)
        first = True
        for L in pat:
            if j >= end - 1e-6:
                break
            L = min(L, end - j)
            slots.append(Slot(j, L, "high", phrase=first and bar % 4 == 0 and bar > 0, bar_start=first))
            first = False
            j += L
        bar += 1
    # never end on a tiny sliver: merge it into the previous clip
    if len(slots) > 1 and slots[-1].length < 1 and slots[-1].section != "drop":
        slots[-2].length += slots[-1].length
        slots.pop()
    return slots


def _cycle(options: list[str], k: int, prev: str | None) -> str:
    v = options[k % len(options)]
    if v == prev:
        v = options[(k + 1) % len(options)]
    return v


def design_shots(slots: list[Slot], unit: float, beat: float, hits: list[tuple[float, float]], src_fps: float,
                 pacing: str) -> tuple[list[Shot], list[tuple[float, str, float]]]:
    """Turns slots into shots with speed, framing, transitions and beat-synced effects + the sound effects."""
    cuts = np.round(np.array([s.start * unit for s in slots] + [(slots[-1].start + slots[-1].length) * unit])
                    * FPS).astype(int)
    shots: list[Shot] = []
    sfx: list[tuple[float, str, float]] = []
    counters = {"calm": 0, "build": 0, "high": 0, "drop": 0, "trans": 0, "tilt": 0, "vel": 0, "one": 0}
    prev_frame = None
    for i, sl in enumerate(slots):
        frames = max(1, int(cuts[i + 1] - cuts[i]))
        t0 = cuts[i] / FPS
        sec = sl.section
        sh = Shot(src_start=0.0, frames=frames, out_start=t0, section=sec)
        last = i == len(slots) - 1
        if sec == "calm":
            sh.speed = "slowmo" if counters["calm"] % 2 == 0 else "velocity"
            sh.framing = _cycle(CALM_FRAMES, counters["calm"], prev_frame)
            sh.grade, sh.letterbox, sh.audio_gain = "moody", (1.0, 1.0), 1.4
            sh.t_in = "dip" if i > 0 and slots[i - 1].section == "calm" else "cut"
            counters["calm"] += 1
        elif sec == "build":
            if sl.length >= 2:
                sh.speed = "ramp_up"
            elif sl.length >= 1:
                sh.speed, sh.base_speed = "normal", 1.2
            else:
                nxt_drop = i + 1 < len(slots) and slots[i + 1].section == "drop"
                sh.speed, sh.base_speed = ("stutter", 1.0) if nxt_drop else ("normal", 1.3)
            sh.framing = _cycle(BUILD_FRAMES, counters["build"], prev_frame)
            sh.grade, sh.letterbox, sh.audio_gain = "moody", (1.0, 1.0), 1.6
            if sl.length < 1:
                sh.t_in = "flash" if counters["build"] % 2 == 0 else "glitch"
                sh.events.append([0.0, "rgb", 0.45])
            counters["build"] += 1
        elif sec == "drop":
            sh.speed = "ramp_down"
            sh.framing = "pull"
            sh.grade, sh.letterbox, sh.audio_gain = "hype", (1.0, 0.0), 1.0
            sh.t_in = "flash"
            sh.events += [[0.0, "shake", 1.0], [0.0, "rgb", 1.0], [0.0, "pulse", 0.2]]
            sfx.append((t0, "impact", 0.75))
            counters["drop"] += 1
        else:
            if sl.length >= 2:
                sh.speed = "velocity" if counters["vel"] % 2 == 0 else "ramp_up"
                counters["vel"] += 1
                opts = ["push", "pan", "pull", "medium"]
            elif sl.length >= 1:          # one-beat clips: plain, hit-and-slow, plain, slow-and-shoot
                sh.speed = ["normal", "ramp_down", "normal", "ramp_up"][counters["one"] % 4]
                sh.base_speed = 1.12
                counters["one"] += 1
                opts = HIGH_FRAMES
            else:
                sh.speed, sh.base_speed = "normal", 1.2
                opts = HIGH_FRAMES
            sh.framing = _cycle(opts, counters["high"], prev_frame)
            sh.grade, sh.audio_gain = "hype", 0.75
            if counters["high"] % 3 == 2 and sh.framing not in ("pan",):
                sh.tilt = 4.0 if counters["tilt"] % 2 == 0 else -4.0
                counters["tilt"] += 1
            if sl.phrase:
                sh.t_in = "flash"
                sh.events += [[0.0, "shake", 0.55], [0.0, "rgb", 0.5]]
                sfx.append((t0, "impact", 0.3))
            elif sl.bar_start:
                sh.t_in = BAR_TRANSITIONS[counters["trans"] % len(BAR_TRANSITIONS)]
                counters["trans"] += 1
            counters["high"] += 1
        if last and sl.length * unit >= 1.2 and sec in ("high", "calm"):
            sh.speed = "slowmo"
        sh.direction = 1 if i % 2 == 0 else -1
        prev_frame = sh.framing
        shots.append(sh)

    # transitions need a few frames on both sides
    for i, sh in enumerate(shots):
        if sh.t_in in ("zoom", "whip"):
            if i == 0 or sh.frames < 10 or shots[i - 1].frames < 10:
                sh.t_in = "cut"
            else:
                prev = shots[i - 1]
                prev.t_out = sh.t_in
                sh.direction = prev.direction
                sfx.append((sh.out_start, "whoosh", 0.32))
        elif sh.t_in == "dip" and i > 0:
            shots[i - 1].t_out = "dip"

    # punch on every cut and zoom pulses on the bass hits
    for sh in shots:
        D = sh.duration
        if sh.section != "calm" and sh.t_in in ("cut", "glitch", "flash"):
            sh.events.append([0.0, "pulse", 0.10 if sh.t_in != "flash" else 0.14])
        amt = {"calm": 0.015, "build": 0.04, "drop": 0.07, "high": 0.07}[sh.section]
        for t, st in hits:
            lt = t - sh.out_start
            if lt < 0.08 or lt >= D - 0.02:
                continue
            sh.events.append([round(lt, 4), "pulse", round(amt * (0.5 + 0.5 * st), 4)])
            if st >= 0.75 and sh.section in ("high", "drop"):
                sh.events += [[round(lt, 4), "shake", 0.3], [round(lt, 4), "rgb", 0.35]]

    # riser into every drop that has a build-up in front of it
    for i, sh in enumerate(shots):
        if sh.section != "drop":
            continue
        k = i
        while k > 0 and shots[k - 1].section == "build":
            k -= 1
        start = shots[k].out_start
        if sh.out_start - start >= 0.9:
            sfx.append((start, f"riser:{sh.out_start - start:.3f}", 0.34))
    return shots, sfx


# --------------------------------------------------------------------------- Claude picks the shots

_SCHEMA = {
    "type": "object",
    "properties": {
        "reels": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "hook": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "hashtags": {"type": "array", "items": {"type": "string"}},
                "shots": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "video": {"type": "integer"},
                        "time": {"type": "number"},
                        "align": {"type": "string", "enum": ["start", "center", "end"]},
                        "effect": {"type": "string", "enum": ["auto", "slowmo", "speed_ramp", "normal"]},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                    },
                    "required": ["video", "time", "align", "effect", "x", "y"],
                    "additionalProperties": False}},
            },
            "required": ["hook", "title", "description", "hashtags", "shots"],
            "additionalProperties": False,
        }},
    },
    "required": ["reels"],
    "additionalProperties": False,
}

_SYSTEM = """You are a top short-form video editor making INSANE, high-energy vertical edits for TikTok, Reels and Shorts
(think car / sport / phonk edits: speed ramps, slow motion, close-ups, flashes, everything on the beat).

You get a storyboard: still frames from one or more source videos. Each frame is labelled with its video number, its
time in seconds and an excitement score from audio/motion analysis (higher = more action). Frames from the same video
are in time order, so you can see what happens when. You also get the shot list of the edit, already cut to the music:
for every shot its role (intro, build-up, DROP, hype...), its length and its planned speed effect.

Choose the footage for every shot:
- shots: one entry per shot in the shot list, in the same order, exactly as many as listed.
  video = the video number. time = the second where the key moment of that shot happens; use the frame times you saw,
  or a time between two frames when the action clearly happens between them.
  align = where that moment sits inside the shot: "start" (it happens right on the cut, on the beat), "center", or
  "end" (the shot ends right after it, so the next shot follows the moment).
  effect = "auto" (keep the planned speed), "slowmo", "speed_ramp" or "normal" (real speed). Change it only when it
  serves the moment or the creator asked for it.
  x, y = the point of the most interesting detail for a close-up, from 0 to 1 measured from the top-left of the frame
  (a speedometer needle, a wheel, the exhaust, a face, a logo, the subject itself). The editor zooms in there.
- The first shot opens the reel under the title and must be striking. The DROP shot gets the single most intense
  moment. Never put two near-identical shots next to each other, mix close details and wide shots, and keep a story
  that builds. Skip intros, logos, title cards, black or blurry frames.
- With several videos, use footage from all of them unless the creator says otherwise, and connect them so it feels
  like one edit, for example by cutting from an action in one video to a matching moment in another.
- The creator's instructions come first. If they ask for an order, a specific moment, a cut from one video to another
  or slow motion on something, do exactly that, and place the moment so it lands on a cut or on the drop.
- Different reels must not reuse the same moments.
- Always write the text, never leave it empty: hook = max 6 words, a punchy on-screen title, no emojis/hashtags.
  title = max 80 characters. description = 1-2 sentences. hashtags = 5-8 relevant ones starting with #. All text in
  the requested language. Write the text fields first, then the shots."""

_ROLE = {"calm": "calm, slow motion", "build": "build-up, quick cut", "drop": "DROP - impact, most intense moment",
         "high": "hype"}
_SPEED = {"slowmo": "slow motion", "velocity": "speed ramp fast-slow-fast", "ramp_up": "slow, then shoots fast",
          "ramp_down": "fast, then melts into slow motion", "stutter": "stutter repeat", "normal": "real speed"}


@dataclass
class Source:
    """One input video, analysed."""
    info: MediaInfo
    an: dict
    score: np.ndarray
    lo: float
    hi: float
    fps: int

    @property
    def usable(self) -> float:
        return max(0.1, self.hi - self.lo)

    def score_at(self, t: float) -> float:
        if len(self.score) == 0:
            return 0.0
        i = int(np.clip(t / GRID, 0, len(self.score) - 1))
        return float(self.score[max(0, i - 2):i + 3].mean())


def _storyboard(sources: list[Source], pool: list[tuple[int, float, float]], budget: int = 90) -> list[tuple[int, float, float]]:
    """(video, time, score) frames for Claude: the most exciting candidates plus even coverage of every video."""
    want = [max(8, int(math.ceil(s.usable))) for s in sources]          # about one frame per second...
    if sum(want) > budget:                                              # ...shared out when there is a lot
        want = [max(4, int(w * budget / sum(want))) for w in want]
    while sum(want) > budget and max(want) > 2:                         # never more images than the budget
        want[want.index(max(want))] -= 1
    out: list[tuple[int, float, float]] = []
    for si, s in enumerate(sources):
        b = want[si]
        step = s.usable / b
        grid = [s.lo + step * (i + 0.5) for i in range(b)]             # even coverage, so events can be timed
        # the most exciting candidate moments replace the nearest grid frame (exact timing on the action)
        for c, _ in sorted(((c, sc) for v, c, sc in pool if v == si), key=lambda x: -x[1])[: max(1, b // 2)]:
            k = int(np.argmin([abs(g - c) for g in grid]))
            if abs(grid[k] - c) > 0.15:
                grid[k] = c
        out += [(si, round(t, 2), s.score_at(t)) for t in sorted(set(round(g, 2) for g in grid))]
    return out


def _shot_lines(template: list[Shot]) -> str:
    lines = []
    for j, sh in enumerate(template):
        lines.append(f"{j + 1}. {_ROLE.get(sh.section, sh.section)}, {sh.duration:.2f}s, {_SPEED.get(sh.speed, sh.speed)}")
    return "\n".join(lines)


def _claude_shots(sources: list[Source], frames: list[tuple[int, float, float]], template: list[Shot], count: int,
                  settings: Settings, names: list[str], save: Path | None = None) -> list[dict] | None:
    if settings.ai == "off" or (settings.ai == "auto" and not _have_credentials()):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    images: dict[tuple[int, float], str] = {}
    for si, s in enumerate(sources):
        ts = [t for v, t, _ in frames if v == si]
        for t, img in zip(ts, _frames_b64(s.info.path, ts)):
            images[(si, t)] = img
    content: list[dict] = []
    multi = len(sources) > 1
    for si, s in enumerate(sources):
        content.append({"type": "text", "text": f"=== Video {si + 1}: \"{names[si]}\" ({s.info.duration:.1f}s) ==="})
        for v, t, sc in frames:
            img = images.get((v, t))
            if v != si or not img:
                continue
            content.append({"type": "text", "text": f"[video {si + 1} @ {t:.1f}s, score {sc:+.1f}]"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
    n_img = sum(1 for c in content if c["type"] == "image")
    if not n_img:
        return None
    lang = LANG_NAMES.get((settings.text_language or "en").lower(), "English")
    ask = (settings.prompt or "").strip()
    content.append({"type": "text", "text": (
        f"Shot list ({len(template)} shots, already cut to the music):\n{_shot_lines(template)}\n\n"
        f"Make {count} reel{'s' if count > 1 else ''}. Write the hook, title, description and hashtags in {lang}.\n\n"
        + (f"The creator's instructions:\n{ask}" if ask else "The creator gave no extra instructions."))})
    kwargs = dict(model=settings.ai_model, max_tokens=64000, system=_SYSTEM,
                  messages=[{"role": "user", "content": content}],
                  output_config={"effort": "high" if (ask or multi) else "medium",
                                 "format": {"type": "json_schema", "schema": _SCHEMA}})
    if settings.ai_model in _FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    what = f"{len(sources)} videos" if multi else "the video"
    log(AGENT, f"Claude is watching {what} ({n_img} frames)" + (" and reading your instructions" if ask else "")
        + " to pick every shot...")
    try:
        with anthropic.Anthropic().beta.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()
    except Exception as e:   # any API problem -> built-in picker
        warn(AGENT, f"Claude unavailable ({type(e).__name__}) - picking shots with the built-in rules")
        return None
    if resp.stop_reason in ("refusal", "max_tokens"):
        warn(AGENT, f"Claude could not finish ({resp.stop_reason}) - picking shots with the built-in rules")
        return None
    text = "".join(b.text for b in resp.content if b.type == "text")
    if save is not None:                      # kept for troubleshooting: what Claude decided and why
        try:
            save.write_text(json.dumps({"stop_reason": resp.stop_reason, "usage": resp.usage.to_dict(),
                                        "instructions": ask, "answer": text}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        except (OSError, TypeError, ValueError):
            pass
    try:
        reels = json.loads(text).get("reels") or []
    except (ValueError, AttributeError):
        warn(AGENT, "Claude's answer could not be read - picking shots with the built-in rules")
        return None
    return [r for r in reels if isinstance(r, dict) and isinstance(r.get("shots"), list) and r["shots"]] or None


# --------------------------------------------------------------------------- main entry

def _reel_map(song: Song | None, music: Path | None, settings: Settings, target: float) -> dict:
    """Song start, beat length, the cut unit, per-unit labels, drops and hits - all in reel time."""
    if song is None:
        beat = {"fast": 0.45, "normal": 0.55, "chill": 0.7}.get(settings.pacing, 0.55)
        NU = max(1, int(target / beat))
        labels = ["low"] * min(4, NU) + ["high"] * max(0, NU - 4)
        drops = [4] if NU >= 10 else []
        hits = [(k * beat, 0.5) for k in range(NU) if k >= 4]
        return {"start": 0.0, "beat": beat, "unit": beat, "NU": NU, "labels": labels, "drops": drops,
                "hits": hits, "loop": 0.0, "note": "steady pace (no music)"}
    mlen = song.duration
    start = choose_start(song, target)
    refined = beat_grid(music, max(0.0, start - song.period / 2), min(target + 8, 120))
    if refined:
        period, off = refined
        if abs(period - song.period) / song.period < 0.03:
            start = max(0.0, start - song.period / 2) + off
        else:
            period = song.period
    else:
        period = song.period
    # the beat grid can lock onto off-beat hi-hats: if nearly all strong bass hits sit half a beat off, shift
    ph = np.array([((t - start) / period) % 1 for t, st in song.hits
                   if st >= 0.6 and start <= t < start + target + 8])
    if len(ph) >= 6 and np.mean(np.abs(ph - 0.5) < 0.15) >= 0.65 and np.mean((ph < 0.15) | (ph > 0.85)) < 0.2:
        start += period / 2
    unit = period * (2 if period < 0.45 else 1)
    if unit > period * 1.5:      # cutting every 2 beats: the drop must be an even number of beats away
        d = next((d for d in song.drops if 0.5 * unit <= d - start < target), None)
        if d is not None and int(round((d - start) / period)) % 2 == 1:
            start = start - period if start - period >= 0 else start + period
    NU = max(1, int(target / unit + 1e-6))
    loop = 0.0
    if start + NU * unit > mlen - 0.05:                     # song too short: loop whole bars
        n_beats = int((mlen - start) / period)
        if n_beats >= 8:
            n_beats -= n_beats % 4
        loop = n_beats * period if n_beats >= 4 else 0.0

    def song_t(t: float) -> float:
        return start + (t % loop if loop > 0 else t)

    labels = ["low" if song.is_low(song_t((k + 0.5) * unit)) else "high" for k in range(NU)]
    span = NU * unit
    passes = int(np.ceil(span / loop)) if loop > 0 else 1
    seg = loop if loop > 0 else span
    drops_set: set[int] = set()
    hits = []
    for p_ in range(passes):
        base = p_ * seg
        for d in song.drops:
            rt = d - start
            if 0 <= rt < seg and 0.5 * unit <= base + rt < span:
                drops_set.add(int(round((base + rt) / unit)))
        for t, st in song.hits:
            rt = t - start
            if 0 <= rt < seg and base + rt < span:
                rt += base
                grid_t = round(rt / (period / 2)) * (period / 2)   # snap to the half-beat grid when close
                hits.append((grid_t if abs(grid_t - rt) < 0.035 else rt, st))
    drops = sorted(drops_set)
    note = f"{60 / period:.0f} BPM, song starts at {start:.1f}s"
    if drops:
        note += f", drop after {drops[0] * unit:.1f}s"
    elif NU >= 10:              # steady song without a drop: still open with a short build-up and an impact
        d = next((d for d in (2, 4, 8) if d * unit >= 1.6), 8)
        if d < NU - 4:
            drops = [d]
            note += f", impact after {d * unit:.1f}s"
    return {"start": start, "beat": period, "unit": unit, "NU": NU, "labels": labels, "drops": drops,
            "hits": hits, "loop": loop, "note": note}


_EFFECT = {"slowmo": "slowmo", "speed_ramp": "velocity", "normal": "normal"}


def plan_hype(infos: MediaInfo | list[MediaInfo], settings: Settings, works: Path | list[Path],
              count: int = 1) -> list[EditPlan]:
    """Plans the insane edit from one or more videos. With Claude, the creator's instructions steer the shots."""
    from .fx import _src_fps, time_map
    if isinstance(infos, MediaInfo):
        infos, works = [infos], [works]
    multi = len(infos) > 1
    sources: list[Source] = []
    for info, w in zip(infos, works):
        an = analyze(info, w)
        dur = max(0.1, info.duration)
        skip_from = 45.0 if multi else 12.0       # clips the creator picked are used whole; long videos skip intros
        intro = min(4.0, dur * 0.03) if dur > skip_from else 0.0
        outro = min(6.0, dur * 0.05) if dur > skip_from else 0.0
        sources.append(Source(info, an, excitement(an), intro, min(dur, max(intro + 1.0, dur - outro)),
                              _src_fps(info)))
    usable = sum(s.usable for s in sources)
    count = max(1, min(int(count), 5))
    max_len = settings.effective_max()
    # slow motion and ramps stretch footage, so even short videos can fill a longer edit
    target = min(max_len, max(min(usable / count * 1.2, 8.0), usable * 0.9 / count))
    music = Path(settings.music) if settings.music else None

    song = None
    if music:
        try:
            song = analyze_song(music)
        except Exception as e:  # unreadable file etc. -> steady pace
            warn(AGENT, f"Could not analyse the music ({type(e).__name__})")
        if song is None:
            warn(AGENT, "No clear beat in the music - cutting at a steady pace instead")
    rm = _reel_map(song, music if song else None, settings, target)
    fps0 = sources[0].fps
    half_ok = rm["unit"] / 2 >= MIN_CLIP
    slots = build_slots(rm["labels"], rm["drops"], rm["NU"], settings.pacing, half_ok)
    template, sfx = design_shots(slots, rm["unit"], rm["beat"], rm["hits"], fps0, settings.pacing)
    n = len(template)
    total = sum(sh.frames for sh in template) / FPS
    drop_slot = next((i for i, sh in enumerate(template) if sh.section == "drop"), None)
    log(AGENT, f"Insane edit: {n} shots, {total:.1f}s from {len(sources)} video{'s' if multi else ''} ({rm['note']})")

    # candidate moments in every video (each video gets its share)
    spans0 = [source_span(sh, fps0) for sh in template]
    Lc = float(np.clip(np.median(spans0) if spans0 else 1.2, 0.6, 2.0))
    pool_n = min(60, max(n * count + 10, int(n * count * 1.3)))
    pool: list[tuple[int, float, float]] = []          # (video, centre second, score)
    for si, s in enumerate(sources):
        L = min(Lc, s.usable)
        k_s = max(3, int(round(pool_n * s.usable / usable)))
        for t0, sc in _pick(_windows(s.an, s.score, L, s.lo, s.hi), L, k_s, [], s.usable / (k_s * 2.5)):
            pool.append((si, t0 + L / 2, sc))
    if not pool:
        pool = [(si, (s.lo + s.hi) / 2, 0.0) for si, s in enumerate(sources)]
    names = [_name_text(s.info.path.stem)[0] or s.info.path.stem for s in sources]
    title_hint, name_tags = _name_text(sources[0].info.path.stem)
    frames = _storyboard(sources, pool)
    ai_reels = (_claude_shots(sources, frames, template, count, settings, names, save=works[0] / "claude_shots.json")
                if frames else None)
    if settings.prompt and not ai_reels:
        warn(AGENT, "Your instructions need Claude (an API key) - editing with the built-in rules instead")

    taken: list[tuple[int, float, float]] = []      # (video, start, end) used by earlier reels
    plans: list[EditPlan] = []
    for k in range(count):
        ai = ai_reels[k] if ai_reels and k < len(ai_reels) else None
        ai_shots = [c for c in ai["shots"] if isinstance(c, dict)] if ai else []
        # built-in order: best first, 2nd best on the drop, the rest video by video in time order -
        # every video gets a share of the shots that matches its length
        ranked = sorted(range(len(pool)), key=lambda i: -pool[i][2])
        order: list[int] = []
        if ranked:
            hook_i = ranked[0]
            hero = ranked[1] if drop_slot is not None and len(ranked) > 1 else None
            share = [max(1, int(round(n * s.usable / usable))) for s in sources]
            main: list[int] = []
            spare: list[int] = []
            for si in range(len(sources)):
                mine = [i for i in ranked if pool[i][0] == si and i not in (hook_i, hero)]
                main += sorted(mine[:share[si]], key=lambda i: pool[i][1])
                spare += mine[share[si]:]
            order = [hook_i] + main
            if hero is not None:
                pos = max(1, drop_slot)
                order = order[:pos] + [hero] + order[pos:]
            order += sorted(spare, key=lambda i: (pool[i][0], pool[i][1]))
        own: list[tuple[int, float, float]] = []

        def free(si: int, a: float, b: float) -> bool:
            return not any(v == si and a < tb - 0.05 and b > ta + 0.05 for v, ta, tb in taken + own)

        shots: list[Shot] = []
        spans: list[float] = []
        used = 0
        cursor = 0
        for j, tmpl in enumerate(template):
            sh = Shot(**{**tmpl.to_dict(), "events": [list(e) for e in tmpl.events]})
            sh.letterbox = tuple(tmpl.letterbox)
            placed = False
            c = ai_shots[j] if j < len(ai_shots) else None
            if c is not None:
                try:
                    si = int(c.get("video", 1)) - 1
                    T = float(c.get("time"))
                except (TypeError, ValueError, OverflowError):
                    si, T = -1, 0.0
                if 0 <= si < len(sources) and math.isfinite(T):
                    s = sources[si]
                    if c.get("effect") in _EFFECT:
                        sh.speed = _EFFECT[c["effect"]]
                        sh.base_speed = 1.0 if sh.speed == "normal" else sh.base_speed
                    S = source_span(sh, s.fps)
                    D = sh.duration
                    u_al = {"start": min(0.1, D * 0.15), "end": max(D * 0.75, D - 0.18)}.get(c.get("align"), D / 2)
                    u, smap = time_map(sh, s.fps)
                    a = min(max(T, 0.0), s.info.duration) - float(np.interp(u_al, u, smap))
                    a = min(max(0.0, a), max(0.0, s.info.duration - S - 0.02))
                    sh.source = si
                    sh.src_start = round(math.floor(a * FPS + 1e-6) / FPS, 4)
                    try:
                        fx_, fy_ = float(c["x"]), float(c["y"])
                        sh.focus = ((float(np.clip(fx_, 0.05, 0.95)), float(np.clip(fy_, 0.08, 0.92)))
                                    if math.isfinite(fx_) and math.isfinite(fy_) else None)
                    except (KeyError, TypeError, ValueError, OverflowError):
                        sh.focus = None
                    placed = True
                    used += 1
            if not placed:                                   # built-in pick
                choice = None
                tries = 0
                while tries < len(order) and choice is None:
                    si, centre, _ = pool[order[(cursor + tries) % len(order)]]
                    s = sources[si]
                    S = source_span(sh, s.fps)
                    for shift in (0.0, S, -S, 2 * S):
                        a = min(max(s.lo, centre - S / 2 + shift), max(s.lo, s.hi - S))
                        if free(si, a, a + S):
                            choice = (si, a)
                            break
                    tries += 1
                if choice is None:                           # out of fresh footage: reuse a moment
                    si, centre, _ = pool[order[cursor % len(order)]]
                    s = sources[si]
                    S = source_span(sh, s.fps)
                    choice = (si, min(max(s.lo, centre - S / 2), max(s.lo, s.hi - S)))
                    cursor += 1
                else:
                    cursor += tries
                    used += 1
                si, a = choice
                sh.source = si
                sh.src_start = round(math.floor(a * FPS + 1e-6) / FPS, 4)   # floor: never past the end
            S = source_span(sh, sources[sh.source].fps)
            shots.append(sh)
            spans.append(S)
            own.append((sh.source, sh.src_start, sh.src_start + S))
        if k > 0 and used < min(n, max(3, n // 2)):
            break                                                  # not enough fresh material for another reel
        taken += own
        title, tags = title_hint, name_tags + ["#fyp", "#viral", "#reels", "#shorts", "#edit"]
        hook = settings.hook if k == 0 else None
        plan = EditPlan(segments=[(s.src_start, round(s.src_start + sp, 4)) for s, sp in zip(shots, spans)],
                        hook=hook, title=title, description=title, hashtags=tags,
                        decided_by="Claude" if ai else "rules", mode="montage",
                        music_start=round(rm["start"], 4), music_loop=round(rm["loop"], 4))
        plan.shots = shots
        plan.sources = [s.info for s in sources]
        plan.sfx_events = list(sfx)
        first_drop = next((sh.out_start for sh in shots if sh.section == "drop"), None)
        if first_drop is not None and 1.2 <= first_drop <= 6.5:
            plan.hook_end = round(first_drop, 3)
            plan.hook_beats = [round(b * rm["beat"], 3) for b in range(int(first_drop / rm["beat"] + 1e-6))]
        if ai:
            plan.hook = hook or (ai.get("hook") or "").strip() or None
            plan.title = (ai.get("title") or title).strip()
            plan.description = (ai.get("description") or title).strip()
            if ai.get("hashtags"):
                plan.hashtags = [t if t.startswith("#") else "#" + t
                                 for t in (str(x).strip().replace(" ", "") for x in ai["hashtags"]) if t]
        mix = ""
        if multi:
            per = [sum(1 for sh in shots if sh.source == si) for si in range(len(sources))]
            mix = " | shots per video: " + ", ".join(f"{i + 1}:{c}" for i, c in enumerate(per))
        log(AGENT, f"Insane reel {k + 1}: {len(shots)} shots, {total:.1f}s, decided by {plan.decided_by}"
                   + (f' | hook "{plan.hook}"' if plan.hook else "") + mix)
        plans.append(plan)
    if count > 1 and len(plans) < count:
        warn(AGENT, f"The video only has material for {len(plans)} good reel(s), not {count}.")
    return plans
