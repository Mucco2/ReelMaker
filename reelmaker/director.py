"""Director agent - decides WHAT ends up in the reel.

1. Picks the best part of the video when it is longer than the target (AI or heuristic).
2. Removes dead air and filler words ("um", "uh") using Whisper's word timestamps.
3. Writes the hook title, post title/description and hashtags.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import FILLER_WORDS, FPS, Settings, filler_words
from .utils import log, warn

AGENT = "Director"

_PUNCT = re.compile(r"[^\w'%$€£-]+", re.UNICODE)
_HOOK_STARTERS = ("how", "why", "what", "stop", "never", "nobody", "this is", "here's", "here is",
                  "the secret", "you", "if you", "did you", "i tried", "the truth", "don't", "most people",
                  # Danish
                  "hvordan", "hvorfor", "hvad", "aldrig", "ingen", "det her", "her er", "hemmeligheden",
                  "du ", "hvis du", "vidste du", "jeg prøvede", "sandheden", "de fleste", "lad være")
_STOPWORDS = set("""
a an the and or but if then so to of in on at by for with from up down out over under is are was were be been
being am i me my we our you your he she it its they them their this that these those there here what which who
whom whose when where why how all any both each few more most other some such no nor not only own same than too
very can will just don should now do does did doing have has had having would could also into about again
really like know think going get got make made one thing things gonna wanna yeah okay ok well right actually
every single need want people time tried said says much many even still back good great first last next take
come came look give tell mean maybe something anything everything nothing someone everyone because while after
before through around without again always lot little kind sure today year years day days way ways
""".split()) | set("""
og jeg det at en den til er som på de med han af for ikke der var mig sig men et har om vi min havde ham hun nu
over da fra du ud sin dem os op man hans hvor eller hvad skal selv her alle vil blev kunne ind når være dog noget
ville jo deres efter ned skulle denne end dette mit også under have dig anden hende mine alt meget sit sine vor mod
disse hvis din nogle hos blive mange bliver hendes været sådan bare lige godt rigtig faktisk altså nej okay kan
får fik gør gjorde går gik kom kommer siger sagde tror ved vide helt hele mere mest lidt hver hvordan hvorfor hvem
dag dage år tid ting folk virkelig egentlig måske noget nogen ingen aldrig altid fordi mens først sidste næste
andet andre vidste fleste flere hver hvert hele nogle sammen samme stadig rigtigt sige gøre lave tage komme
""".split())


@dataclass
class EditPlan:
    segments: list[tuple[float, float]]           # source time ranges to keep, in play order
    hook: str | None = None
    title: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    emphasis: set[str] = field(default_factory=set)
    decided_by: str = "rules"
    mode: str = "speech"                          # speech | montage
    music_start: float = 0.0                      # where in the song the reel starts (montage, beat-aligned)
    music_loop: float = 0.0                       # montage: whole-beat length of song to loop if it's too short
    # insane montage edit (fx.py / hype.py)
    shots: list = field(default_factory=list)     # fx.Shot per clip; empty = classic cutter
    sfx_events: list = field(default_factory=list)  # [reel time, kind, gain] sound effects
    sfx: str | None = None                        # rendered sound-effect track (set by the cutter)
    hook_end: float | None = None                 # hook leaves exactly on the drop
    hook_beats: list = field(default_factory=list)  # beat times for the word-by-word hook

    @property
    def duration(self) -> float:
        if self.shots:
            return sum(s.frames for s in self.shots) / FPS
        return sum(e - s for s, e in self.segments)


def clean_word(w: str) -> str:
    return _PUNCT.sub("", w.lower()).strip("'-")


def is_filler(w: dict, language: str | None = None) -> bool:
    return clean_word(w["word"]) in filler_words(language)


# --------------------------------------------------------------------------- tightening

def tighten(words: list[dict], settings: Settings, src_duration: float) -> list[tuple[float, float]]:
    """Turn words into keep-ranges: cut pauses longer than `gap` and drop filler words."""
    p = settings.pacing_params()
    gap, pre, post = p["gap"], p["pre"], p["post"]
    ranges: list[list[float]] = []
    fillers: list[tuple[float, float]] = []
    broken = True
    for w in words:
        if settings.remove_fillers and is_filler(w, settings.speech_language):
            fillers.append((w["start"], w["end"]))
            broken = True
            continue
        if ranges and not broken and w["start"] - ranges[-1][1] <= gap:
            ranges[-1][1] = max(ranges[-1][1], w["end"])
        else:
            ranges.append([w["start"], w["end"]])
        broken = False

    padded = [[max(0.0, s - pre), min(src_duration, e + post)] for s, e in ranges]
    # never let padding swallow a removed filler word
    for fs, fe in fillers:
        out = []
        for s, e in padded:
            if fe <= s or fs >= e:
                out.append([s, e])
                continue
            if s < fs:
                out.append([s, fs])
            if fe < e:
                out.append([fe, e])
        padded = out
    # merge overlaps
    padded.sort()
    merged: list[list[float]] = []
    for s, e in padded:
        if merged and s <= merged[-1][1] + 0.04:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in merged if e - s >= 0.15]


def dedupe(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove source time already used earlier in play order (neighbouring clips' padding can overlap,
    which would play the same audio twice)."""
    used: list[tuple[float, float]] = []
    out: list[tuple[float, float]] = []
    for s, e in ranges:
        parts = [(s, e)]
        for us, ue in used:
            nxt = []
            for ps, pe in parts:
                if ue <= ps or us >= pe:
                    nxt.append((ps, pe))
                    continue
                if ps < us:
                    nxt.append((ps, us))
                if ue < pe:
                    nxt.append((ue, pe))
            parts = nxt
        parts = [(round(ps, 3), round(pe, 3)) for ps, pe in parts if pe - ps >= 0.15]
        out.extend(parts)
        used.append((s, e))
    return out


def cap_length(ranges: list[tuple[float, float]], words: list[dict], limit: float) -> list[tuple[float, float]]:
    """Trim the end so the total fits `limit`, cutting on a word boundary."""
    out, total = [], 0.0
    for s, e in ranges:
        if total + (e - s) <= limit:
            out.append((s, e))
            total += e - s
            continue
        remaining = limit - total
        if remaining > 0.5:
            ends = [w["end"] for w in words if s < w["end"] <= s + remaining]
            cut = (max(ends) + 0.15) if ends else s + remaining
            out.append((s, min(cut, s + remaining, e)))
        break
    return out


# --------------------------------------------------------------------------- highlight picking

def _est_len(segs: list[dict], i: int, j: int, gap: float) -> float:
    total = 0.0
    for k in range(i, j + 1):
        total += segs[k]["end"] - segs[k]["start"]
        if k < j:
            total += min(max(0.0, segs[k + 1]["start"] - segs[k]["end"]), gap)
    return total


def pick_window_rules(segs: list[dict], max_len: float, gap: float,
                      used: set[int] | None = None) -> tuple[int, int] | None:
    """Best contiguous run of segments that fits max_len and avoids `used` segments."""
    used = used or set()
    best, best_score = None, -1.0
    n = len(segs)
    for i in range(n):
        if i in used:
            continue
        j = i
        while j + 1 < n and j + 1 not in used and _est_len(segs, i, j + 1, gap) <= max_len:
            j += 1
        est = _est_len(segs, i, j, gap)
        if est > max_len * 1.05:
            continue
        words = sum(len(segs[k]["words"]) for k in range(i, j + 1))
        density = words / max(est, 1e-3)
        first = segs[i]["text"].strip().lower()
        last = segs[j]["text"].strip()
        score = 0.5 * min(est / max_len, 1.0) + 0.3 * min(density / 3.0, 1.0)
        if "?" in first or first.startswith(_HOOK_STARTERS):
            score += 0.15
        if last.endswith((".", "!", "?")):
            score += 0.10
        if i == 0 or segs[i - 1]["text"].strip().endswith((".", "!", "?")):
            score += 0.05
        if score > best_score:
            best, best_score = (i, j), score
    return best


# --------------------------------------------------------------------------- AI director (Claude)

_REEL_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "first_segment": {"type": "integer"},
                    "last_segment": {"type": "integer"},
                },
                "required": ["first_segment", "last_segment"],
                "additionalProperties": False,
            },
        },
        "hook": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "emphasis_words": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["clips", "hook", "title", "description", "hashtags", "emphasis_words"],
    "additionalProperties": False,
}

AI_SCHEMA = {
    "type": "object",
    "properties": {"reels": {"type": "array", "items": _REEL_SCHEMA}},
    "required": ["reels"],
    "additionalProperties": False,
}

AI_SYSTEM = """You are an expert short-form video editor for TikTok, Instagram Reels, YouTube Shorts and Facebook Reels.
You receive a timestamped transcript split into numbered segments. Decide which segments make the most engaging,
self-contained vertical video(s), and write the on-screen hook and the post text for each.

Rules:
- Return exactly the number of reels asked for, best one first, unless the material is too thin - then return fewer.
  Different reels must not reuse the same segments, and each must work on its own.
- Total speech of each reel's segments must be at most the stated maximum seconds (pauses get removed, so estimate
  from segment start/end times). Aim close to the target length when the material is strong enough.
- The first 2-3 seconds decide whether viewers stay: start on a strong line (a surprising claim, a question, a
  bold promise). Clips play in the order you list them; you may put a later segment first only if it works as a
  cold-open hook and the story still makes sense.
- Prefer one continuous range. Use several ranges only to skip weak or off-topic parts. Never cut mid-sentence.
- End on a complete thought or punchline.
- hook: max 7 words, written in the transcript's language, shown big on screen during the first seconds. Make it
  intriguing but true to the content. No emojis, no hashtags.
- title: max 80 characters. description: 1-2 sentences for the post caption, in the transcript's language.
- hashtags: 5-8 relevant hashtags, each starting with #.
- emphasis_words: 5-15 single words copied exactly from the chosen segments that deserve a highlight colour
  (key nouns, numbers, strong verbs)."""


def _have_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}


def _request_kwargs(segs: list[dict], src_duration: float, max_len: float, settings: Settings,
                    count: int) -> dict:
    lines = [f"[{i}] {s['start']:.1f}-{s['end']:.1f}s: {s['text']}" for i, s in enumerate(segs)]
    target = min(settings.target_duration, max_len)
    notes = ("\nStyle notes from the reference video:\n- " + "\n- ".join(settings.style_notes)) if settings.style_notes else ""
    user = (f"Make {count} reel{'s' if count > 1 else ''}. Video length: {src_duration:.1f}s. "
            f"Maximum reel length: {max_len:.0f}s. Target: about {target:.0f}s."
            f"{notes}\n\nTranscript segments:\n" + "\n".join(lines))

    kwargs = dict(
        model=settings.ai_model,
        max_tokens=16000,
        system=AI_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": AI_SCHEMA}},
    )
    if settings.ai_model in _FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    return kwargs


def test_claude(api_key: str, settings: Settings | None = None) -> tuple[bool, bool, str]:
    """Send the exact same kind of request a real reel uses, on a tiny transcript.

    Returns (works, key_is_valid, message in Danish)."""
    import anthropic
    settings = settings or Settings()
    segs = [{"start": 0.0, "end": 3.0, "text": "Vidste du at de fleste mennesker sover for lidt?"},
            {"start": 3.4, "end": 7.0, "text": "Her er tre nemme tricks til bedre søvn i aften."}]
    kwargs = _request_kwargs(segs, 7.0, 60, settings, 1)
    kwargs["max_tokens"] = 4000
    kwargs["output_config"] = {**kwargs["output_config"], "effort": "low"}
    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=90, max_retries=1)
        resp = client.beta.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        return False, False, "Nøglen virker ikke. Tjek at du har kopieret hele nøglen (den starter med sk-ant-)."
    except anthropic.PermissionDeniedError:
        return False, False, "Nøglen har ikke adgang til Claude-modellen. Lav en ny nøgle på platform.claude.com."
    except anthropic.RateLimitError:
        return True, True, "Nøglen virker (Claude er travl lige nu, men det går fint til dine videoer)."
    except anthropic.APIStatusError as e:
        msg = str(getattr(e, "message", "") or e)
        if "credit" in msg.lower() or "billing" in msg.lower():
            return False, True, ("Nøglen er rigtig, men din konto har ingen kredit. Tilføj betaling under "
                                 "Billing på platform.claude.com og prøv igen.")
        return False, True, f"Claude svarede med fejl {e.status_code}: {msg[:300]}"
    except anthropic.APIConnectionError:
        return False, False, "Kunne ikke forbinde til Claude. Tjek din internetforbindelse."
    except Exception as e:
        return False, False, f"Uventet fejl: {type(e).__name__}: {str(e)[:200]}"
    if resp.stop_reason == "refusal":
        return True, True, "Nøglen virker."
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        reels = json.loads(text).get("reels") or []
        hook = (reels[0].get("hook") or "").strip() if reels else ""
    except (json.JSONDecodeError, AttributeError, IndexError):
        return True, True, "Nøglen virker, men svaret var uventet - sig det til udvikleren."
    return True, True, "Claude AI virker!" + (f' Test-hook: "{hook}"' if hook else "")


def ai_plan(segs: list[dict], src_duration: float, max_len: float, settings: Settings,
            count: int = 1) -> list[dict] | None:
    if settings.ai == "off":
        return None
    if settings.ai == "auto" and not _have_credentials():
        return None
    try:
        import anthropic
    except ImportError:
        warn(AGENT, "anthropic package missing - using built-in rules")
        return None

    kwargs = _request_kwargs(segs, src_duration, max_len, settings, count)
    log(AGENT, f"Asking Claude ({settings.ai_model}) to pick the best moments...")
    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(**kwargs)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.CredentialsError):
        warn(AGENT, "No valid Claude API key - using built-in rules (set ANTHROPIC_API_KEY to enable AI).")
        return None
    except anthropic.RateLimitError:
        warn(AGENT, "Claude rate limit hit - using built-in rules this time.")
        return None
    except anthropic.APIStatusError as e:
        warn(AGENT, f"Claude API error {e.status_code} - using built-in rules.")
        return None
    except anthropic.APIConnectionError:
        warn(AGENT, "Could not reach the Claude API - using built-in rules.")
        return None
    except anthropic.AnthropicError as e:
        warn(AGENT, f"Claude unavailable ({type(e).__name__}) - using built-in rules.")
        return None
    except Exception as e:  # e.g. TypeError "Could not resolve authentication method" when no key is set
        if "authentication" in str(e).lower():
            warn(AGENT, "No Claude API key found - using built-in rules (set ANTHROPIC_API_KEY to enable AI).")
        else:
            warn(AGENT, f"Claude unavailable ({type(e).__name__}) - using built-in rules.")
        return None

    if resp.stop_reason == "refusal":
        warn(AGENT, "Claude declined this transcript - using built-in rules.")
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        reels = json.loads(text).get("reels") or []
    except (json.JSONDecodeError, AttributeError):
        warn(AGENT, "Claude returned unreadable output - using built-in rules.")
        return None
    return [r for r in reels if isinstance(r, dict) and r.get("clips")] or None


# --------------------------------------------------------------------------- text helpers

def _keywords(words: list[dict], n: int = 5) -> list[str]:
    counts = Counter(clean_word(w["word"]) for w in words)
    ranked = [(k, c) for k, c in counts.most_common()
              if k and len(k) >= 4 and k not in _STOPWORDS and k not in FILLER_WORDS and not k.isdigit()]
    return [k for k, _ in ranked[:n]]


def _rules_text(words: list[dict], language: str | None = None) -> tuple[str, str, list[str], set[str]]:
    spoken = " ".join(w["word"] for w in words if not is_filler(w, language))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", spoken.strip()) if s]
    first = sentences[0] if sentences else ""
    title = first if len(first) <= 80 else first[:77].rsplit(" ", 1)[0] + "..."
    desc = " ".join(sentences[:2])[:220]
    kws = _keywords(words, 6)
    tags = ["#" + k for k in kws[:4]] + ["#fyp", "#viral", "#reels", "#shorts"]
    emphasis = {clean_word(w["word"]) for w in words if any(ch.isdigit() for ch in w["word"])}
    emphasis |= {clean_word(w["word"]) for w in words if w["word"].endswith("!")}
    emphasis |= set(kws[:5])
    emphasis.discard("")
    return title, desc, tags, emphasis


# --------------------------------------------------------------------------- main entry

def _ai_blocks(clips: list[dict], n: int, seen: set[int]) -> list[tuple[int, int]]:
    """Validate Claude's clips: drop bad indices and segments already used (by this or an earlier reel)."""
    chosen: list[tuple[int, int]] = []
    for c in clips:
        try:
            i, j = int(c["first_segment"]), int(c["last_segment"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= i < n and 0 <= j < n):
            continue                      # invalid segment numbers -> ignore that clip
        if j < i:
            i, j = j, i
        fresh = [k for k in range(i, j + 1) if k not in seen]
        seen.update(fresh)
        for k in fresh:                   # keep only unused parts, as contiguous runs
            if chosen and chosen[-1][1] == k - 1 and k - 1 >= i:
                chosen[-1] = (chosen[-1][0], k)
            else:
                chosen.append((k, k))
    return chosen


def plan_edits(transcript: dict, src_duration: float, settings: Settings, count: int = 1) -> list[EditPlan]:
    """Plan `count` different reels (best first). May return fewer if the video is short."""
    segs = transcript.get("segments", [])
    max_len = settings.effective_max()
    gap = settings.pacing_params()["gap"]
    count = max(1, min(int(count), 5))

    if not segs:
        end = min(src_duration, max_len)
        log(AGENT, f"No speech found - keeping the first {end:.1f}s")
        return [EditPlan(segments=[(0.0, round(end, 3))], hook=settings.hook,
                         title="", description="", hashtags=["#fyp", "#viral", "#reels", "#shorts"])]

    n = len(segs)
    total_est = _est_len(segs, 0, n - 1, gap)
    blocks: list[tuple[list[tuple[int, int]], dict | None]] = []

    ai_reels = ai_plan(segs, src_duration, max_len, settings, count)
    if ai_reels:
        seen: set[int] = set()
        for reel in ai_reels[:count]:
            chosen = _ai_blocks(reel.get("clips", []), n, seen)
            if chosen:
                blocks.append((chosen, reel))

    if not blocks:
        if count == 1 and total_est <= max_len:
            blocks = [([(0, n - 1)], None)]
        else:
            used: set[int] = set()
            min_len = min(8.0, max_len * 0.3)
            for _ in range(count):
                win = pick_window_rules(segs, max_len, gap, used)
                if win is None:
                    break
                i, j = win
                if blocks and _est_len(segs, i, j, gap) < min_len:
                    break                 # not enough good material left for another reel
                used.update(range(i, j + 1))
                blocks.append(([(i, j)], None))
            if not blocks:
                blocks = [([(0, n - 1)], None)]   # cap_length trims it to max_len

    if count > 1 and len(blocks) < count:
        warn(AGENT, f"The video only has material for {len(blocks)} good reel(s), not {count}.")

    plans = []
    for k, (chosen, reel) in enumerate(blocks):
        # a typed hook goes on the first reel only (other reels show different moments)
        user_hook = settings.hook if k == 0 else None
        plans.append(_build_plan(segs, chosen, reel, user_hook, settings, src_duration, max_len))
    return plans


def plan_edit(transcript: dict, src_duration: float, settings: Settings) -> EditPlan:
    return plan_edits(transcript, src_duration, settings, 1)[0]


def _build_plan(segs: list[dict], chosen: list[tuple[int, int]], ai: dict | None, user_hook: str | None,
                settings: Settings, src_duration: float, max_len: float) -> EditPlan:
    decided_by = "Claude" if ai else "rules"
    # Build keep-ranges for each chosen block (keeps AI's play order).
    ranges: list[tuple[float, float]] = []
    sel_words: list[dict] = []
    for i, j in chosen:
        words = [w for k in range(i, j + 1) for w in segs[k]["words"]]
        sel_words.extend(words)
        ranges.extend(tighten(words, settings, src_duration))
    ranges = cap_length(dedupe(ranges), sel_words, max_len)

    title, desc, tags, emphasis = _rules_text(sel_words, settings.speech_language)
    hook = user_hook
    if ai:
        hook = hook or (ai.get("hook") or "").strip() or None
        title = (ai.get("title") or title).strip()
        desc = (ai.get("description") or desc).strip()
        if ai.get("hashtags"):
            tags = [t if t.startswith("#") else "#" + t for t in (x.strip().replace(" ", "") for x in ai["hashtags"]) if t]
        if ai.get("emphasis_words"):
            emphasis = {clean_word(w) for w in ai["emphasis_words"]} - {""}

    plan = EditPlan(segments=ranges, hook=hook, title=title, description=desc,
                    hashtags=tags, emphasis=emphasis, decided_by=decided_by)
    cut = src_duration - plan.duration
    blocks = ", ".join(f"{segs[i]['start']:.0f}-{segs[j]['end']:.0f}s" for i, j in chosen)
    log(AGENT, f"Decided by {decided_by}: keep {blocks} -> {len(ranges)} pieces, "
               f"{plan.duration:.1f}s final (cut {cut:.1f}s of pauses/fillers/extra)")
    if hook:
        log(AGENT, f'Hook: "{hook}"')
    return plan
