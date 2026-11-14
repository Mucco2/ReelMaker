"""Command line: python -m reelmaker <video or link> [options]"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

from .config import CAPTION_STYLES, PACING, PLATFORM_LIMITS, Settings
from .utils import ENGINE_DIR, WORK_DIR, PipelineError, find_tool

COMMANDS = {"make", "analyze", "doctor"}


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reelmaker",
        description="Turn any video (file or link) into a ready-to-post vertical reel.",
        epilog="Examples:\n"
               "  reelmaker my_video.mp4\n"
               "  reelmaker https://youtu.be/xyz --max-duration 45 --hook \"This changed everything\"\n"
               "  reelmaker talk.mp4 --reference https://www.tiktok.com/@someone/video/123\n"
               "  reelmaker analyze https://www.instagram.com/reel/abc/\n"
               "  reelmaker doctor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    m = sub.add_parser("make", help="edit a video into a reel (default)")
    m.add_argument("input", help="video file path or link (YouTube, TikTok, Instagram, ...)")
    m.add_argument("--max-duration", type=float, help="longest the reel may be, seconds (default 60)")
    m.add_argument("--count", type=int, choices=range(1, 6), metavar="N",
                   help="make N different reels from the best moments (1-5, default 1)")
    m.add_argument("--platform", choices=sorted(PLATFORM_LIMITS), help="enforce a platform's length limit")
    m.add_argument("--pacing", choices=sorted(PACING), help="how hard to cut pauses (default normal)")
    m.add_argument("--keep-fillers", action="store_true", help="keep um/uh")
    m.add_argument("--reframe", choices=["auto", "face", "crop", "blur", "fit"], help="how to make it 9:16")
    m.add_argument("--no-zoom", dest="zoom_cuts", action="store_false", default=None, help="no zoom punch-ins on cuts")
    m.add_argument("--no-enhance", dest="enhance", action="store_false", default=None, help="no colour/sharpness boost")
    m.add_argument("--no-captions", dest="captions", action="store_false", default=None)
    m.add_argument("--caption-style", choices=sorted(CAPTION_STYLES))
    m.add_argument("--accent", help="highlight colour, e.g. \"#FF3366\"")
    m.add_argument("--hook", help="big title shown in the first seconds")
    m.add_argument("--cta", help="call to action at the end (default: 'follow for more' in the video's language)")
    m.add_argument("--no-cta", action="store_true")
    m.add_argument("--no-progress", dest="progress_bar", action="store_false", default=None)
    m.add_argument("--animator", choices=["hyperframes", "ass"], help="animation engine (default hyperframes)")
    m.add_argument("--music", type=Path, help="background music file (auto-ducked under speech)")
    m.add_argument("--music-volume", type=float, help="0.0-1.0, default 0.12")
    m.add_argument("--whisper-model", help="auto | tiny | base | small | medium | large-v3 | large-v3-turbo "
                                           "(default auto = large-v3-turbo, small for videos over 20 min)")
    m.add_argument("--language", help="force speech language, e.g. en, es, de (default auto-detect)")
    m.add_argument("--ai", choices=["auto", "on", "off"], help="use Claude to pick highlights + write hook (default auto)")
    m.add_argument("--reference", help="a reel you like (link or file) - copies its length and pacing")
    m.add_argument("--style-file", type=Path, help="style.json from `reelmaker analyze`")
    m.add_argument("--out", type=Path, help="output folder")

    a = sub.add_parser("analyze", help="study a reference reel and save its style")
    a.add_argument("input", help="link or file")
    a.add_argument("--save", type=Path, default=Path("style.json"))

    sub.add_parser("doctor", help="check that everything is installed")
    return p


def _settings_from(args: argparse.Namespace) -> tuple[Settings, set[str]]:
    s = Settings()
    explicit: set[str] = set()

    def put(attr: str, value):
        if value is not None:
            setattr(s, attr, value)
            explicit.add(attr)

    put("max_duration", args.max_duration)
    if args.max_duration:
        s.target_duration = args.max_duration
    put("count", args.count)
    put("platform", args.platform)
    put("pacing", args.pacing)
    if args.keep_fillers:
        put("remove_fillers", False)
    put("reframe", args.reframe)
    put("zoom_cuts", args.zoom_cuts)
    put("enhance", args.enhance)
    put("captions", args.captions)
    put("caption_style", args.caption_style)
    put("accent", args.accent)
    put("hook", args.hook.strip() if args.hook and args.hook.strip() else None)
    put("cta", args.cta)
    if args.no_cta:
        s.cta = None
    put("progress_bar", args.progress_bar)
    put("animator", args.animator)
    if args.music:
        if not args.music.exists():
            raise PipelineError(f"Music file not found: {args.music}")
        put("music", args.music.resolve())
    put("music_volume", args.music_volume)
    put("whisper_model", args.whisper_model)
    put("language", args.language)
    put("ai", args.ai)
    put("out_dir", args.out.resolve() if args.out else None)
    return s, explicit


def doctor() -> int:
    ok = True

    def line(name: str, good: bool, detail: str = "", optional: bool = False):
        nonlocal ok
        mark = "\033[92mOK \033[0m" if good else ("\033[93m-- \033[0m" if optional else "\033[91mNO \033[0m")
        if not good and not optional:
            ok = False
        print(f"  {mark} {name:<22} {detail}")

    print("ReelMaker doctor")
    line("ffmpeg", bool(find_tool("ffmpeg")), find_tool("ffmpeg") or "run setup.bat")
    line("ffprobe", bool(find_tool("ffprobe")), find_tool("ffprobe") or "run setup.bat")
    line("yt-dlp", importlib.util.find_spec("yt_dlp") is not None or bool(find_tool("yt-dlp")))
    line("faster-whisper", importlib.util.find_spec("faster_whisper") is not None, "speech -> captions")
    line("opencv (faces)", importlib.util.find_spec("cv2") is not None, "smart crop", optional=True)
    face_model = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    line("face model", face_model.exists(), str(face_model.name), optional=True)
    line("node", bool(shutil.which("node")), shutil.which("node") or "install Node.js 22+")
    line("hyperframes", (ENGINE_DIR / "node_modules" / "hyperframes").exists(), "animations")
    line("gsap", (ENGINE_DIR / "node_modules" / "gsap").exists())
    chrome_dir = Path.home() / ".cache" / "hyperframes" / "chrome"
    line("headless chrome", chrome_dir.exists(), "used by hyperframes")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                   or (Path.home() / ".config" / "anthropic").exists())
    line("Claude AI director", has_key, "set ANTHROPIC_API_KEY to enable (optional)", optional=True)
    print("All good!" if ok else "Something is missing - run setup.bat")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `reelmaker video.mp4 ...` and `reelmaker --hook X video.mp4` both mean `make`
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "make")
    parser = _make_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "analyze":
            from .reference import analyze
            profile = analyze(args.input, WORK_DIR / "references", save_to=args.save.resolve())
            print(json.dumps(profile, indent=2))
            print(f"Saved -> {args.save.resolve()}  (use it with --style-file)")
            return 0
        from .pipeline import make_reel
        settings, explicit = _settings_from(args)
        make_reel(args.input, settings, ref=args.reference, style_file=args.style_file, explicit=explicit)
        return 0
    except PipelineError as e:
        print(f"\n\033[91mERROR:\033[0m {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
