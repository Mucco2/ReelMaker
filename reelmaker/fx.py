"""FX engine - renders the "insane" montage edit frame by frame.

Every shot is rebuilt from the source footage in Python (numpy + OpenCV):
- time remapping: slow motion, speed ramps / velocity edits and stutters, with frame blending and motion blur
- a virtual camera: wide / close-up / pan / push-in / pull-out framing, dutch tilt, beat-synced zoom pulses and shake
- transitions: zoom blur, whip pan, flash, glitch, dip to black
- a colour grade per section (moody intro, punchy drop), letterbox bars and RGB split
FFmpeg decodes the footage and encodes the frames. The original sound is time-remapped exactly like the picture
(the pitch follows the speed, like a tape), and synthetic sound effects (whoosh, impact, riser) go on their own track.
"""

from __future__ import annotations

import math
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import FPS, HEIGHT, WIDTH, Settings
from .utils import MediaInfo, Stopwatch, ffmpeg, log, run, tool_env, warn

AGENT = "Cutter"
OW, OH = WIDTH, HEIGHT
SR = 48000
OS = 8                    # time-map oversampling per output frame
MAX_SRC = 1920            # longest side the footage is decoded at

ZOOM = {"wide": 1.0, "medium": 1.38, "tight": 1.8, "pan": 1.06, "push": 1.0, "pull": 1.0}

_NOWIN = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


@dataclass
class Shot:
    src_start: float                    # source second of the shot's first frame
    frames: int                         # output frames
    source: int = 0                     # which input video (multi-video edits)
    out_start: float = 0.0              # where it sits in the reel (seconds)
    speed: str = "normal"               # normal | slowmo | velocity | ramp_down | ramp_up | stutter
    base_speed: float = 1.0
    framing: str = "wide"               # wide | medium | tight | pan | push | pull
    focus: tuple[float, float] | None = None    # point of interest (0-1) for close-ups
    centre_x: float | None = None       # where a wide 9:16 crop sits (0-1), set by the renderer if None
    tilt: float = 0.0                   # dutch angle in degrees
    direction: int = 1                  # pan / whip direction
    grade: str = "hype"                 # hype | moody
    letterbox: tuple[float, float] = (0.0, 0.0)  # bar size at start / after 0.3 s (0-1)
    t_in: str = "cut"                   # cut | zoom | whip | flash | glitch | dip
    t_out: str = "cut"                  # cut | zoom | whip | dip
    events: list = field(default_factory=list)   # [local seconds, kind, strength]: pulse | shake | rgb | flash
    audio_gain: float = 1.0
    section: str = "high"

    @property
    def duration(self) -> float:
        return self.frames / FPS

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- time remapping

def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def speed_limits(src_fps: float) -> tuple[float, float]:
    """(slowest, fastest) playback speed. 50/60 fps footage can go slower and still look smooth."""
    return (0.4, 2.4) if src_fps >= 47 else (0.55, 2.2)


def time_map(shot: Shot, src_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """(output times, source times relative to src_start) on a grid of FPS*OS points per second."""
    n = shot.frames * OS + 1
    D = shot.frames / FPS
    u = np.linspace(0.0, D, n)
    x = u / D if D > 0 else u
    vmin, vmax = speed_limits(src_fps)
    k = shot.speed
    if k == "stutter":                                  # play the first half twice (beat repeat)
        c = D / 2
        return u, np.mod(u, c) * shot.base_speed if c > 0 else u
    if k == "slowmo":
        v = np.full(n, vmin + 0.05)
    elif k == "velocity":                               # fast - slow on the beat - fast
        v = vmin + (vmax - vmin) * _smooth(0.12, 0.9, np.abs(2 * x - 1))
    elif k == "ramp_down":                              # hit hard, then melt into slow motion
        v = vmax - (vmax - vmin) * _smooth(0.0, 0.5, x)
    elif k == "ramp_up":                                # slow motion that shoots into the next cut
        v = vmin + (vmax - vmin) * _smooth(0.45, 1.0, x)
    else:
        v = np.full(n, shot.base_speed)
    s = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(u))])
    return u, s


def source_span(shot: Shot, src_fps: float) -> float:
    _, s = time_map(shot, src_fps)
    return float(s.max())


# --------------------------------------------------------------------------- colour

def _grade_lut(grade: str) -> tuple[np.ndarray, float]:
    """256-entry BGR LUT (contrast S-curve, teal shadows / warm highlights) + saturation factor."""
    x = np.linspace(0.0, 1.0, 256)
    if grade == "moody":
        contrast, crush, expo, tint, sat = 0.55, 0.05, 0.9, 0.09, 0.55
    else:
        contrast, crush, expo, tint, sat = 0.38, 0.03, 1.02, 0.07, 1.22
    y = np.clip(x * expo, 0, 1)
    y = y + contrast * (y * y * (3 - 2 * y) - y)          # S-curve
    y = np.clip((y - crush) / (1 - crush), 0, 1)           # deeper blacks
    b = np.clip(y - tint * (y - 0.5), 0, 1)                # shadows bluer, highlights warmer
    g = np.clip(y - 0.3 * tint * (y - 0.5), 0, 1)
    r = np.clip(y + tint * (y - 0.5), 0, 1)
    lut = np.stack([b, g, r], axis=1).reshape(256, 1, 3)
    return (lut * 255 + 0.5).astype(np.uint8), sat


_LUTS: dict[str, tuple[np.ndarray, float]] = {}


def _apply_grade(cv2, img: np.ndarray, grade: str, sat_boost: float = 0.0) -> np.ndarray:
    if grade not in _LUTS:
        _LUTS[grade] = _grade_lut(grade)
    lut, sat = _LUTS[grade]
    out = cv2.LUT(img, lut)
    sat = sat + sat_boost
    if abs(sat - 1.0) > 0.02:
        gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, sat, gray, 1.0 - sat, 0)
    return out


# --------------------------------------------------------------------------- camera

def _noise(u: float, seed: float) -> tuple[float, float, float]:
    """Smooth pseudo-random shake (-1..1) from incommensurate sines."""
    a = math.sin(u * 71.0 + seed) * 0.6 + math.sin(u * 43.0 + seed * 2.1) * 0.4
    b = math.sin(u * 59.0 + seed * 1.7) * 0.6 + math.sin(u * 37.0 + seed * 0.3) * 0.4
    c = math.sin(u * 29.0 + seed * 0.9)
    return a, b, c


def _decay(events: list, kind: str, u: float, tau: float) -> float:
    total = 0.0
    for t, k, s in events:
        if k == kind and u >= t - 1e-6 and u - t < tau * 6:
            total += s * math.exp(-(u - t) / tau)
    return total


def _crop_size(W: float, H: float, zoom: float) -> tuple[float, float]:
    ar = OW / OH
    if W / H > ar:
        ch = H / zoom
        return ch * ar, ch
    cw = W / zoom
    return cw, cw / ar


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2x3 affine a after b."""
    A = np.vstack([a, [0, 0, 1]])
    B = np.vstack([b, [0, 0, 1]])
    return (A @ B)[:2]


def _about_centre(scale: float, angle_deg: float, dx: float, dy: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    cx, cy = OW / 2, OH / 2
    return np.array([[c, -s, cx - c * cx + s * cy + dx],
                     [s, c, cy - s * cx - c * cy + dy]], dtype=np.float64)


@dataclass
class FrameFX:
    matrix: np.ndarray        # source px -> output px
    zoom_blur: float          # 0..1
    whip_blur: float          # px
    rgb: float                # px
    flash: float              # 0..1 (white)
    dip: float                # 0..1 (black)
    glitch: bool
    letterbox: float          # 0..1
    sat_boost: float


def frame_fx(shot: Shot, j: int, W: float, H: float, seed: float) -> FrameFX:
    """Everything the virtual camera does on output frame j of the shot."""
    n = shot.frames
    D = n / FPS
    u = j / FPS
    x = j / max(1, n - 1)
    fx, fy = shot.focus if shot.focus else (shot.centre_x if shot.centre_x is not None else 0.5, 0.5)
    wide_x = shot.centre_x if shot.centre_x is not None else fx
    z = ZOOM.get(shot.framing, 1.0)
    cx, cy = (wide_x, 0.5) if shot.framing in ("wide", "pan", "push", "pull") else (fx, fy)
    e = float(_smooth(0.0, 1.0, np.array(x)))
    if shot.framing == "push":
        z *= 1.0 + 0.28 * (1 - (1 - x) ** 2)
        cx, cy = wide_x + (fx - wide_x) * 0.6 * x, 0.5 + (fy - 0.5) * 0.6 * x
    elif shot.framing == "pull":
        z *= 1.0 + 0.45 * (1 - x) ** 2
        cx, cy = wide_x + (fx - wide_x) * 0.6 * (1 - x) ** 2, 0.5 + (fy - 0.5) * 0.6 * (1 - x) ** 2
    elif shot.framing == "pan":
        cw, _ = _crop_size(W, H, z)
        room = max(0.0, 1.0 - cw / W)
        if room < 0.08:                       # no room to pan (vertical footage): slow push instead
            z *= 1.0 + 0.15 * x
        else:
            a0, a1 = 0.5 - 0.42 * room, 0.5 + 0.42 * room
            if shot.direction < 0:
                a0, a1 = a1, a0
            cx = a0 + (a1 - a0) * e
    else:
        z *= 1.0 + 0.05 * x                   # slow drift keeps every shot alive
    angle = shot.tilt * (0.6 + 0.4 * e)

    # transitions
    TZ, TW, TD = 5, 4, 6
    zoom_blur = whip = 0.0
    off_x = 0.0
    if shot.t_in == "zoom" and j < TZ:
        p = 1 - j / TZ
        z *= 1 + 0.9 * p * p
        zoom_blur = max(zoom_blur, p)
    if shot.t_out == "zoom" and j >= n - TZ:
        p = (j - (n - TZ) + 1) / TZ
        z *= 1 + 0.9 * p * p
        zoom_blur = max(zoom_blur, p)
    if shot.t_in == "whip" and j < TW:
        p = 1 - j / TW
        off_x -= shot.direction * p * p * 0.45 * OW
        whip = max(whip, 150 * p)
    if shot.t_out == "whip" and j >= n - TW:
        p = (j - (n - TW) + 1) / TW
        off_x += shot.direction * p * p * 0.45 * OW
        whip = max(whip, 150 * p)
    dip = 0.0
    if shot.t_in == "dip" and j < TD:
        dip = max(dip, 1 - j / TD)
    if shot.t_out == "dip" and j >= n - TD:
        dip = max(dip, (j - (n - TD) + 1) / TD)
    flash = 0.85 * max(0.0, 1 - j / 4) ** 1.5 if shot.t_in == "flash" else 0.0
    flash = min(1.0, flash + _decay(shot.events, "flash", u, 0.06))
    glitch = shot.t_in == "glitch" and j < 3

    # beat pulses and shake (output space, so they never hit the frame edge thanks to the extra zoom)
    pulse = _decay(shot.events, "pulse", u, 0.10)
    shake = min(1.5, _decay(shot.events, "shake", u, 0.18))
    rgb = 16 * _decay(shot.events, "rgb", u, 0.07) + (10 if glitch else 0) + 6 * zoom_blur
    n1, n2, n3 = _noise(u, seed)
    sh_scale = 1.0 + pulse + shake * 0.07
    sh_angle = shake * 1.8 * n3
    dx, dy = shake * 0.032 * OW * n1, shake * 0.022 * OH * n2

    # crop window in source pixels (rotated window kept inside the frame)
    cw, ch = _crop_size(W, H, z)
    a = math.radians(angle)
    ca, sa = abs(math.cos(a)), abs(math.sin(a))
    f = max(1.0, (cw * ca + ch * sa) / W, (cw * sa + ch * ca) / H)
    cw, ch = cw / f, ch / f
    bw, bh = (cw * ca + ch * sa) / 2, (cw * sa + ch * ca) / 2
    px = min(max(cx * W, bw), W - bw)
    py = min(max(cy * H, bh), H - bh)
    k = OW / cw
    c, s = math.cos(a), math.sin(a)
    base = np.array([[k * c, -k * s, OW / 2 - k * (c * px - s * py)],
                     [k * s, k * c, OH / 2 - k * (s * px + c * py)]], dtype=np.float64)
    M = _compose(_about_centre(sh_scale, sh_angle, dx + off_x, dy), base)

    lb0, lb1 = shot.letterbox
    lb = lb1 + (lb0 - lb1) * max(0.0, 1 - u / 0.3)
    sat_boost = 0.25 * max(0.0, 1 - u / 0.5) if shot.section == "drop" else 0.0
    return FrameFX(M, zoom_blur, whip, rgb, flash, dip, glitch, lb, sat_boost)


# --------------------------------------------------------------------------- decoding

class FrameReader:
    """Sequential raw-frame reader with a small look-back cache (for blending and stutters)."""

    def __init__(self, src: Path, start: float, dur: float, fps: int, scale: tuple[int, int],
                 crop: tuple[int, int, int, int], keep: int = 28):
        x, y, w, h = crop
        self.w, self.h = w, h
        self.size = w * h * 3
        self.src, self.start, self.dur = src, start, dur
        self.vf = f"fps={fps}:start_time=0,scale={scale[0]}:{scale[1]}:flags=bicubic,crop={w}:{h}:{x}:{y}"
        self.cache: dict[int, np.ndarray] = {}
        self.next = 0
        self.eof = False
        self.keep = keep
        self.retried = False
        self.proc = self._open(start, dur)

    def _open(self, start: float, dur: float) -> subprocess.Popen:
        cmd = [ffmpeg(), "-v", "error", "-threads", "4"]
        if start > 0:
            cmd += ["-ss", f"{start:.4f}"]
        cmd += ["-i", str(self.src), "-map", "0:V:0", "-t", f"{dur:.4f}", "-vf", self.vf,
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=tool_env(),
                                creationflags=_NOWIN, bufsize=self.size * 2)

    def _read_one(self) -> bool:
        if self.eof:
            return False
        buf = self.proc.stdout.read(self.size)
        if not buf or len(buf) < self.size:
            self.eof = True
            return False
        self.cache[self.next] = np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w, 3)
        self.next += 1
        for old in [i for i in self.cache if i < self.next - self.keep]:
            del self.cache[old]
        return True

    def get(self, i: int) -> np.ndarray:
        i = max(0, i)
        while self.next <= i and self._read_one():
            pass
        if i in self.cache:
            return self.cache[i]
        if not self.cache and not self.retried and self.start > 0:
            # nothing decoded: the window is past the last video frame (audio longer than video).
            # Read the last 3 seconds instead and freeze on the final real frame.
            self.retried = True
            self.close()
            self.proc = self._open(max(0.0, self.start - 3.0), self.dur + 3.0)
            self.eof, self.next = False, 0
            while self._read_one():
                pass
            if self.cache:
                last = self.cache[max(self.cache)]
                self.cache = {0: last}
                self.next = 1
        if not self.cache:
            return np.zeros((self.h, self.w, 3), np.uint8)
        keys = sorted(self.cache)
        return self.cache[keys[-1] if i > keys[-1] else keys[0]]

    def close(self) -> None:
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:  # pragma: no cover
            pass


def _read_audio(src: Path, start: float, dur: float, has_audio: bool) -> np.ndarray:
    n = int(dur * SR) + 1
    if not has_audio:
        return np.zeros((n, 2), np.float32)
    cmd = [ffmpeg(), "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-i", str(src), "-map", "0:a:0", "-t", f"{dur:.4f}", "-ac", "2", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, env=tool_env(), timeout=300, creationflags=_NOWIN)
    data = np.frombuffer(proc.stdout, dtype=np.int16) if proc.returncode == 0 else np.zeros(0, np.int16)
    a = data[: len(data) // 2 * 2].reshape(-1, 2).astype(np.float32) / 32768.0
    if len(a) < n:
        a = np.vstack([a, np.zeros((n - len(a), 2), np.float32)])
    return a


# --------------------------------------------------------------------------- one shot

def _src_dims(info: MediaInfo) -> tuple[int, int]:
    W, H = info.width, info.height
    f = min(1.0, MAX_SRC / max(W, H))
    return int(W * f) // 2 * 2, int(H * f) // 2 * 2


def _src_fps(info: MediaInfo) -> int:
    f = info.fps or 30
    return 60 if f >= 47 else 30 if f >= 27 else 25 if f >= 24.5 else 24


def _glitch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(int(rng.integers(4, 8))):
        y0 = int(rng.integers(0, h - 40))
        bh = int(rng.integers(12, 110))
        shift = int(rng.integers(-90, 90))
        out[y0:y0 + bh] = np.roll(out[y0:y0 + bh], shift, axis=1)
    return out


def _rgb_split(img: np.ndarray, d: int) -> np.ndarray:
    if d < 1:
        return img
    out = img.copy()
    out[:, d:, 2] = img[:, :-d, 2]        # red to the right
    out[:, :-d, 0] = img[:, d:, 0]        # blue to the left
    return out


def render_shot(idx: int, shot: Shot, info: MediaInfo, settings: Settings, out: Path) -> np.ndarray:
    """Renders one shot to `out` (video only) and returns its time-remapped original audio."""
    import cv2
    cv2.setNumThreads(2)
    W, H = _src_dims(info)
    fps = _src_fps(info)
    n = shot.frames
    u, s = time_map(shot, fps)
    seed = (idx * 7.13) % 50

    # per frame: source time, shutter span and the camera
    times = s[::OS][:n]
    nxt = s[np.minimum(np.arange(n) * OS + OS, len(s) - 1)]
    speeds = np.abs(nxt - times) * FPS
    if shot.speed == "stutter":                       # the jump back is a cut, not motion
        speeds = np.minimum(speeds, shot.base_speed * 1.05)
    shutters = np.minimum(speeds / FPS * 0.9, 8.0 / fps)
    fxs = [frame_fx(shot, j, W, H, seed) for j in range(n)]

    # the part of the frame the camera ever looks at (keeps the decoded frames small)
    corners = np.array([[0, 0, 1], [OW, 0, 1], [0, OH, 1], [OW, OH, 1]], dtype=np.float64).T
    xs, ys = [], []
    for f in fxs:
        inv = cv2.invertAffineTransform(f.matrix)
        p = inv @ corners
        xs += [p[0].min(), p[0].max()]
        ys += [p[1].min(), p[1].max()]
    bx0 = int(max(0, min(xs) - 6)) // 2 * 2
    by0 = int(max(0, min(ys) - 6)) // 2 * 2
    bx1 = int(min(W, math.ceil(max(xs) + 6)))
    by1 = int(min(H, math.ceil(max(ys) + 6)))
    bw, bh = max(16, (bx1 - bx0) // 2 * 2), max(16, (by1 - by0) // 2 * 2)
    bw, bh = min(bw, W - bx0), min(bh, H - by0)

    pre = min(0.15, shot.src_start)
    read_start = shot.src_start - pre
    read_dur = float(s.max()) + pre + 0.15
    reader = FrameReader(info.path, read_start, read_dur, fps, (W, H), (bx0, by0, bw, bh))

    post = []
    if settings.enhance:
        post.append("unsharp=5:5:0.55:5:5:0")
    post += ["vignette=angle=0.5", "noise=c0s=5:c0f=t", "format=yuv420p"]
    try:
        enc = subprocess.Popen([ffmpeg(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-vf", ",".join(post),
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-g", str(FPS),
                                "-pix_fmt", "yuv420p", "-an", str(out)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=tool_env(), creationflags=_NOWIN)
    except Exception:
        reader.close()
        raise
    rng = np.random.default_rng(idx + 1)
    shift = np.array([[0, 0, bx0], [0, 0, by0]], dtype=np.float64)
    try:
        for j in range(n):
            f = fxs[j]
            t = times[j]
            fi = (t + pre) * fps
            span = shutters[j] * fps
            if span < 1.2:                                   # interpolate between the two nearest frames
                i0 = int(math.floor(fi))
                w = fi - i0
                a = reader.get(i0)
                if 0.12 < w < 0.88:
                    img = cv2.addWeighted(a, 1 - w, reader.get(i0 + 1), w, 0)
                else:
                    img = a if w <= 0.12 else reader.get(i0 + 1)
            else:                                            # fast: average frames = motion blur
                i1 = int(round(fi))
                cnt = max(2, min(8, int(round(span))))
                acc = np.zeros((bh, bw, 3), np.uint16)
                for i in range(i1 - cnt + 1, i1 + 1):
                    np.add(acc, reader.get(i), out=acc, casting="unsafe")
                img = (acc // cnt).astype(np.uint8)
            img = _apply_grade(cv2, img, shot.grade, f.sat_boost)

            M = f.matrix + f.matrix[:, :2] @ shift            # account for the decode crop
            interp = cv2.INTER_CUBIC if ZOOM.get(shot.framing, 1) > 1.2 else cv2.INTER_LINEAR
            frame = cv2.warpAffine(img, M, (OW, OH), flags=interp, borderMode=cv2.BORDER_REFLECT101)
            if f.zoom_blur > 0.05:                            # radial blur: blend slightly bigger copies
                layers = [frame.astype(np.uint16)]
                for k in (1, 2, 3):
                    Mk = _compose(_about_centre(1 + 0.035 * k * f.zoom_blur, 0, 0, 0), M)
                    layers.append(cv2.warpAffine(img, Mk, (OW, OH), flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_REFLECT101).astype(np.uint16))
                frame = (sum(layers) // len(layers)).astype(np.uint8)
            if f.whip_blur >= 3:
                frame = cv2.blur(frame, (int(f.whip_blur) | 1, 1))
            if f.rgb >= 1:
                frame = _rgb_split(frame, int(f.rgb))
            if f.glitch:
                frame = _glitch(frame, rng)
            if f.flash > 0.01:
                frame = cv2.convertScaleAbs(frame, alpha=1 - f.flash * 0.7, beta=255 * f.flash * 0.75)
            if f.dip > 0.01:
                frame = cv2.convertScaleAbs(frame, alpha=1 - f.dip, beta=0)
            if f.letterbox > 0.01:
                bar = int(OH * 0.085 * f.letterbox)
                if bar > 0:
                    frame[:bar] = 0
                    frame[OH - bar:] = 0
            enc.stdin.write(np.ascontiguousarray(frame).data)
        enc.stdin.close()
        err = enc.stderr.read().decode("utf-8", "replace") if enc.stderr else ""
        if enc.wait(timeout=300) != 0:
            raise RuntimeError(f"encoding shot {idx + 1} failed: {err.strip()[-400:]}")
    finally:
        reader.close()
        if enc.poll() is None:
            enc.kill()
        for pipe in (enc.stdin, enc.stderr):
            try:
                if pipe and not pipe.closed:
                    pipe.close()
            except OSError:
                pass
        try:
            enc.wait(timeout=10)
        except Exception:  # pragma: no cover
            pass

    # original sound, time-remapped the same way (pitch follows the speed)
    audio = _read_audio(info.path, read_start, read_dur, info.has_audio)
    m = n * (SR // FPS)
    ut = np.arange(m) / SR
    st = np.interp(ut, u, s) + pre
    pos = st * SR
    idxs = np.arange(len(audio))
    outa = np.stack([np.interp(pos, idxs, audio[:, c]) for c in range(2)], axis=1).astype(np.float32)
    if shot.speed == "stutter":                     # soften the jump back
        jumps = np.where(np.diff(st) < 0)[0]
        ramp = np.linspace(0, 1, 160, dtype=np.float32)
        for jx in jumps:
            a0, a1 = max(0, jx - 160), jx
            outa[a0:a1] *= ramp[::-1][: a1 - a0, None]
            b1 = min(m, jx + 160)
            outa[jx:b1] *= ramp[: b1 - jx, None]
    fade = min(200, m // 4)
    if fade > 0:
        r = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        outa[:fade] *= r
        outa[-fade:] *= r[::-1]
    return outa * shot.audio_gain


# --------------------------------------------------------------------------- sound effects

def _onepole_bandpass(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Noise through a time-varying band (difference of two one-pole low-passes)."""
    a_hi = 1 - np.exp(-2 * np.pi * hi / SR)
    a_lo = 1 - np.exp(-2 * np.pi * lo / SR)
    y1 = np.empty_like(x)
    y2 = np.empty_like(x)
    p1 = p2 = 0.0
    for i in range(len(x)):
        p1 += a_hi[i] * (x[i] - p1)
        p2 += a_lo[i] * (x[i] - p2)
        y1[i] = p1
        y2[i] = p2
    return y1 - y2


def sfx_whoosh(dur: float = 0.42, peak: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    env = np.where(t < peak, (t / peak) ** 2.2, np.exp(-(t - peak) / (1 - peak) * 4))
    fc = 350 + 4200 * env
    x = rng.standard_normal(n)
    y = _onepole_bandpass(x, fc * 0.45, fc * 1.6) * env
    y /= np.abs(y).max() + 1e-9
    pan = np.clip(t * 1.4 - 0.2, 0, 1)                       # sweeps left -> right
    return np.stack([y * np.cos(pan * np.pi / 2), y * np.sin(pan * np.pi / 2)], axis=1) * 1.2


def sfx_impact(dur: float = 1.5, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 32 + 48 * np.exp(-t / 0.09)                          # pitch drop 80 -> 32 Hz
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)
    thump = np.sin(2 * np.pi * 125 * t) * np.exp(-t / 0.06) * 0.6
    click = rng.standard_normal(n) * np.exp(-t / 0.012) * 0.5
    y = np.tanh(2.2 * (sub + thump + click))                 # saturation = audible on phone speakers
    y /= np.abs(y).max() + 1e-9
    return np.stack([y, y], axis=1)


def sfx_riser(dur: float, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(1, int(dur * SR))
    t = np.linspace(0, 1, n)
    env = t ** 2.5
    fc = 250 + 6500 * t ** 2
    noise = _onepole_bandpass(rng.standard_normal(n), fc * 0.5, fc * 1.5)
    noise /= np.abs(noise).max() + 1e-9
    tone = np.sin(2 * np.pi * np.cumsum(140 + 900 * t ** 2) / SR) * 0.35
    y = (noise + tone) * env
    y[-120:] *= np.linspace(1, 0, min(120, n))
    return np.stack([y, y], axis=1) / (np.abs(y).max() + 1e-9)


def build_sfx(events: list, duration: float) -> np.ndarray:
    """events: [time, kind, gain] with kind whoosh | impact | riser:<seconds>."""
    n = int(round(duration * SR))
    out = np.zeros((n, 2), np.float32)
    cache: dict[str, np.ndarray] = {}
    for k, (t, kind, gain) in enumerate(events):
        if kind == "whoosh":
            snd = cache.setdefault(f"w{k % 3}", sfx_whoosh(0.42, 0.7, seed=k % 3))
            start = t - 0.42 * 0.7
        elif kind == "impact":
            snd = cache.setdefault("impact", sfx_impact())
            start = t
        elif kind.startswith("riser:"):
            d = float(kind.split(":", 1)[1])
            snd = sfx_riser(d)
            start = t
        else:
            continue
        a = int(round(start * SR))
        s0 = max(0, -a)
        a = max(0, a)
        b = min(n, a + len(snd) - s0)
        if b > a:
            out[a:b] += snd[s0:s0 + (b - a)].astype(np.float32) * gain
    return out


def write_wav(path: Path, audio: np.ndarray) -> Path:
    data = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return path


# --------------------------------------------------------------------------- framing helpers

def place_wide_crops(info: MediaInfo, shots: list[Shot]) -> None:
    """Centre of the wide 9:16 crop for every shot: a face if there is one, else where the detail is."""
    if not shots:
        return
    if info.is_vertical:
        for sh in shots:
            sh.centre_x = 0.5
        return
    try:
        from .faces import FaceFinder
        finder = FaceFinder(info.path)
    except Exception as e:  # opencv broken -> centre
        warn(AGENT, f"Smart framing unavailable ({e}); centring the shots")
        for sh in shots:
            sh.centre_x = 0.5
        return
    crop_frac = min(1.0, (info.height * OW / OH) / max(1, info.width))
    try:
        for sh in shots:
            if sh.centre_x is not None:
                continue
            a = sh.src_start
            b = a + max(0.3, sh.duration)
            x = finder.center_x(a, b, samples=1) if finder.ok else None
            sh.centre_x = x if x is not None else finder.content_x(a, b, crop_frac)
    finally:
        finder.close()


def saliency_focus(info: MediaInfo, t: float) -> tuple[float, float]:
    """Fallback close-up point without AI: where something moves against a still background, or has detail."""
    import cv2
    proc = subprocess.run([ffmpeg(), "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(info.path),
                           "-frames:v", "3", "-vf", "fps=10,scale=192:-2,format=gray", "-f", "rawvideo", "-"],
                          capture_output=True, env=tool_env(), timeout=60, creationflags=_NOWIN)
    w = 192
    h = int(round(192 * info.height / max(1, info.width) / 2)) * 2
    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    if proc.returncode != 0 or h <= 0 or len(data) < w * h * 2:
        return 0.5, 0.5
    frames = data[: len(data) // (w * h) * w * h].reshape(-1, h, w).astype(np.float32)
    diff = np.abs(frames[-1] - frames[0])
    local = np.clip(diff - cv2.GaussianBlur(diff, (0, 0), 12), 0, None)
    g = frames[0]
    edges = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    sal = cv2.GaussianBlur(local / (local.max() + 1e-6) + 0.35 * edges / (edges.max() + 1e-6), (0, 0), 6)
    yy, xx = np.mgrid[0:h, 0:w]
    sal *= np.exp(-(((xx / w - 0.5) / 0.4) ** 2 + ((yy / h - 0.5) / 0.4) ** 2) / 2)   # prefer the middle
    y, x = np.unravel_index(int(np.argmax(sal)), sal.shape)
    return float(np.clip(x / w, 0.15, 0.85)), float(np.clip(y / h, 0.2, 0.8))


# --------------------------------------------------------------------------- main entry

def render_shots(infos: MediaInfo | list[MediaInfo], shots: list[Shot], settings: Settings,
                 work: Path) -> tuple[Path, list[np.ndarray]]:
    """Renders every shot (in parallel) and joins them. Returns (edited video with original sound, audio parts).

    `infos` is one video, or a list when the edit mixes several videos (Shot.source picks one)."""
    sw = Stopwatch()
    infos = [infos] if isinstance(infos, MediaInfo) else list(infos)
    for sh in shots:
        sh.source = sh.source if 0 <= sh.source < len(infos) else 0
    for si, info in enumerate(infos):
        place_wide_crops(info, [sh for sh in shots if sh.source == si])
    for sh in shots:
        if sh.focus is None and sh.framing in ("medium", "tight", "push", "pull"):
            sh.focus = saliency_focus(infos[sh.source], sh.src_start + 0.3)
    seg_dir = work / "pieces"
    seg_dir.mkdir(parents=True, exist_ok=True)
    for old in list(seg_dir.glob("*.mov")) + list(seg_dir.glob("*.mp4")):
        old.unlink()
    log(AGENT, f"Rendering {len(shots)} shots with effects (slow motion, speed ramps, zooms, shake, flashes)...")
    files = [seg_dir / f"s{i:04d}.mp4" for i in range(len(shots))]
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = [pool.submit(render_shot, i, sh, infos[sh.source], settings, files[i])
                   for i, sh in enumerate(shots)]
        audio = [f.result() for f in futures]
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)     # stop queued shots: the fallback takes over
        raise
    pool.shutdown(wait=True)
    listfile = seg_dir / "list.txt"
    listfile.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
    wav = write_wav(work / "original.wav", np.concatenate(audio) if audio else np.zeros((1, 2), np.float32))
    edited = work / "edited.mov"
    run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", listfile.name, "-i", wav.resolve(), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "pcm_s16le", edited.resolve()], cwd=seg_dir, what="joining shots")
    log(AGENT, f"Effects edit ready: {sum(s.frames for s in shots) / FPS:.1f}s ({sw.lap()})")
    return edited, audio
