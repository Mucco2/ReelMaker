"""All the knobs in one place."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

WIDTH, HEIGHT, FPS = 1080, 1920, 30

# Max length each platform accepts for a reel/short (seconds).
PLATFORM_LIMITS = {
    "tiktok": 600,
    "reels": 180,       # Instagram Reels
    "shorts": 180,      # YouTube Shorts
    "facebook": 90,     # Facebook Reels
    "all": 90,          # safe for every platform
}

# How aggressively dead air is cut between words.
#   gap   = pauses longer than this get cut out
#   pre   = breathing room kept before speech
#   post  = breathing room kept after speech
PACING = {
    "chill":  {"gap": 0.80, "pre": 0.15, "post": 0.30},
    "normal": {"gap": 0.45, "pre": 0.10, "post": 0.20},
    "fast":   {"gap": 0.28, "pre": 0.06, "post": 0.14},
}

FILLER_WORDS = {
    # sounds that are fillers in (almost) every language
    "um", "umm", "uh", "uhh", "uhm", "hmm", "hm", "mm", "mhm", "ah", "eh", "ehm",
    # Danish / Norwegian / Swedish / German
    "øh", "øhm", "øhh", "øhhh", "æh", "æhm", "æhh", "öh", "öhm", "äh", "ähm", "ähh",
}
# Fillers that are REAL words in other languages ("er" = "is" in Danish/Norwegian) - only cut them here.
FILLER_WORDS_BY_LANG = {
    "en": {"er", "erm"},
}


def filler_words(language: str | None) -> set[str]:
    return FILLER_WORDS | FILLER_WORDS_BY_LANG.get((language or "").lower(), set())

# End-screen call to action, picked from the detected speech language.
CTA_TEXT = {
    "en": "FOLLOW FOR MORE", "da": "FØLG FOR MERE", "no": "FØLG FOR MER", "nn": "FØLG FOR MEIR",
    "sv": "FÖLJ FÖR MER", "de": "FOLGE FÜR MEHR", "es": "SÍGUEME PARA MÁS", "fr": "ABONNE-TOI POUR PLUS",
    "it": "SEGUIMI PER ALTRO", "nl": "VOLG VOOR MEER", "pt": "SIGA PARA MAIS", "pl": "OBSERWUJ PO WIĘCEJ",
    "fi": "SEURAA LISÄÄ", "tr": "DAHA FAZLASI İÇİN TAKİP ET",
}

CAPTION_STYLES = {
    # font is fetched from Google Fonts by HyperFrames (cached locally after first use)
    "bold":  {"font": "Montserrat", "weight": 900, "size": 92, "upper": True,
              "color": "#FFFFFF", "accent": "#FFE135", "stroke": 11, "box": False},
    "box":   {"font": "Montserrat", "weight": 900, "size": 88, "upper": True,
              "color": "#FFFFFF", "accent": "#22C55E", "stroke": 9, "box": True},
    "clean": {"font": "Poppins", "weight": 700, "size": 74, "upper": False,
              "color": "#FFFFFF", "accent": "#7DD3FC", "stroke": 0, "box": False},
}


@dataclass
class Settings:
    # what kind of edit
    mode: str = "auto"                      # auto | speech | montage (montage = no talking, cut to the music)
    fx: str = "insane"                      # montage style: insane (speed ramps, effects, SFX) | classic
    text_language: str | None = None        # language for hook/post text when the video has no speech

    # length & pacing
    max_duration: float | None = None      # None -> platform limit / auto
    target_duration: float = 60.0           # what we aim for when picking highlights
    pacing: str = "normal"
    remove_fillers: bool = True

    # look
    reframe: str = "auto"                   # auto | face | crop | blur | fit
    zoom_cuts: bool = True                  # alternate 100%/110% zoom on jump cuts
    enhance: bool = True                    # light colour + sharpness boost

    # overlays
    captions: bool = True
    caption_style: str = "bold"
    accent: str | None = None
    hook: str | None = None                 # big title at the start (auto if AI is on)
    prompt: str | None = None               # free-text instructions for Claude ("put video 2 after the jump")
    cta: str | None = "auto"                # "auto" = translated by speech language, None = off
    progress_bar: bool = True
    animator: str = "hyperframes"           # hyperframes | ass

    # audio
    music: Path | None = None
    music_volume: float = 0.12
    loudness: float = -14.0                 # LUFS, what TikTok/IG/YT normalise to

    # brains
    whisper_model: str = "auto"             # auto = large-v3-turbo (most accurate), small for very long videos
    count: int = 1                          # how many different reels to make from one video
    language: str | None = None
    ai: str = "auto"                        # auto | on | off  (Claude picks highlights + writes hook)
    ai_model: str = "claude-opus-5-5"

    # output
    platform: str = "all"
    out_dir: Path | None = None

    # filled in while running
    style_notes: list[str] = field(default_factory=list)   # from the Reference Analyst
    speech_language: str | None = None                      # detected by the Transcriber

    def effective_max(self) -> float:
        limit = PLATFORM_LIMITS.get(self.platform, 90)
        return min(self.max_duration or self.target_duration, limit)

    def pacing_params(self) -> dict:
        return PACING.get(self.pacing, PACING["normal"])

    def resolve_cta(self, language: str | None) -> None:
        if self.cta == "auto":
            self.cta = CTA_TEXT.get((language or "en").lower(), CTA_TEXT["en"])

    def caption_params(self) -> dict:
        p = dict(CAPTION_STYLES.get(self.caption_style, CAPTION_STYLES["bold"]))
        if self.accent:
            p["accent"] = self.accent
        return p

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
