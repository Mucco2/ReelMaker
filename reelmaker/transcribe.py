"""Transcriber agent - word-level timestamps with OpenAI's Whisper models.

Uses faster-whisper (the same OpenAI Whisper models, ~4x faster on CPU and lighter on RAM).
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

from .utils import PipelineError, Stopwatch, ffmpeg, log, run

AGENT = "Transcriber"

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


def extract_audio(video: Path, wav: Path) -> None:
    # first_pts=0 pads silence if the audio starts later than the video, so word times match the video
    run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", video,
         "-vn", "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        what="audio extraction")


def load_wav(wav: Path):
    """16 kHz mono PCM -> float32 numpy array (bypasses PyAV, whose API changes often)."""
    import numpy as np
    with wave.open(str(wav), "rb") as f:
        frames = f.readframes(f.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


LONG_VIDEO = 20 * 60   # above this, "auto" picks the faster model


def resolve_model(model_name: str, duration: float) -> str:
    """auto -> large-v3-turbo (much better captions, esp. non-English) unless the video is very long."""
    if model_name != "auto":
        return model_name
    return "small" if duration > LONG_VIDEO else "large-v3-turbo"


def transcribe(video: Path, work: Path, model_name: str = "auto", language: str | None = None,
               has_audio: bool = True, duration: float = 0.0) -> dict:
    """Return {"language", "segments": [{start, end, text, words: [{start, end, word, prob}]}]}.

    Cached in work/transcript_<model>.json so re-runs with other styles are instant.
    """
    model_name = resolve_model(model_name, duration)
    cache = work / f"transcript_v2_{model_name}_{language or 'auto'}.json"
    if cache.exists():
        log(AGENT, "Using cached transcript")
        return json.loads(cache.read_text(encoding="utf-8"))

    if not has_audio:
        log(AGENT, "Video has no audio - skipping speech recognition")
        return {"language": None, "segments": []}

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise PipelineError("faster-whisper is not installed. Run setup.bat.") from e

    sw = Stopwatch()
    wav = work / "audio16k.wav"
    extract_audio(video, wav)
    size = {"large-v3-turbo": "1.6 GB", "large-v3": "3 GB", "medium": "1.5 GB", "small": "460 MB"}.get(model_name, "")
    log(AGENT, f"Loading Whisper '{model_name}'" + (f" (first run downloads {size} once)" if size else "") + "...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=max(1, (os.cpu_count() or 4)))
    log(AGENT, "Listening to the video...")
    audio = load_wav(wav)
    if audio.size < 1600:
        log(AGENT, "Audio track is empty - no captions")
        return {"language": None, "segments": []}
    segments_iter, info = model.transcribe(
        audio, language=language, word_timestamps=True, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        beam_size=5, condition_on_previous_text=False,
    )
    segments = []
    for seg in segments_iter:
        words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word.strip(),
                  "prob": round(w.probability, 3)} for w in (seg.words or []) if w.word.strip()]
        if not words:
            continue
        segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3),
                         "text": seg.text.strip(), "words": words})
    result = {"language": info.language, "segments": segments}
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    n_words = sum(len(s["words"]) for s in segments)
    log(AGENT, f"Heard {n_words} words in '{info.language}' ({sw.lap()})")
    try:
        wav.unlink()
    except OSError:
        pass
    return result


def all_words(transcript: dict) -> list[dict]:
    out = []
    for si, seg in enumerate(transcript.get("segments", [])):
        for w in seg["words"]:
            out.append({**w, "seg": si})
    return out
