# ReelMaker

Drop in any video (a file or a link). Get back a vertical 9:16 video that's ready to post on
TikTok, Instagram Reels, YouTube Shorts and Facebook Reels. It cuts out pauses and "um"s, adds animated
word-by-word captions, a hook title, zoom cuts and clean audio.

**Requirements:** Windows 10 or 11, an internet connection for the first setup, and about 5 GB free space.

## Download og installation (dansk)

1. Tryk på den grønne knap **Code** øverst på GitHub-siden og vælg **Download ZIP**.
2. Højreklik på ZIP-filen, vælg **Egenskaber**, sæt flueben i **Fjern blokering** og tryk OK.
   Ellers kan Windows blokere .bat-filerne.
3. Pak ZIP-filen ud, fx på skrivebordet.
4. Dobbeltklik på **`setup.bat`**. Den installerer alt, hvad programmet skal bruge, og tager 5-10 minutter.
   Hvis Windows viser "Windows beskyttede din pc", så tryk **Flere oplysninger** og derefter **Kør alligevel**.
5. Dobbeltklik på **`ReelMaker.bat`** for at starte.

## Sådan bruger du det (dansk)

1. **Første gang:** dobbeltklik på `setup.bat`. Den installerer alt, der skal bruges.
2. **Start:** dobbeltklik på `ReelMaker.bat`, eller træk en video hen på den. Så åbner ReelMaker
   som sit eget vindue. Det bruger Microsoft Edge, der allerede er på din PC.
3. Træk en video ind i vinduet, tryk **Vælg fil**, eller indsæt et link fra YouTube, TikTok eller Instagram.
   Telefonen til højre viser hele tiden, hvordan din reel kommer til at se ud.
   Slå **Vis TikTok-knapper** til for at tjekke, at teksten ikke bliver dækket af appens knapper.
4. Udfyld det, du vil (alt er valgfrit):
   - **Hook-tekst**: den store titel i starten, fx `Jeg droppede telefonen i 30 dage`.
   - **Reference-reel link**: en reel, hvis stil du kan lide. Programmet kopierer dens længde og tempo.
   - **Maks længde** og **Antal reels**: vælg 2-5 for at få flere forskellige reels fra én lang video.
   - **Undertekst-stil** (Fed / Boks / Ren), **Farve**, **Tempo**, **Sprog**, **Musik**.
5. Tryk **Lav reel**, eller tryk Ctrl+Enter. Du kan følge agenterne trin for trin.
   Når videoen er færdig, afspilles den i telefonen. Tryk **Kopiér opslagstekst** for at få teksten og
   hashtags klar til at indsætte i appen. Under **Seneste reels** finder du dine tidligere videoer.

**To slags klip (Klippestil):**
- **Tale:** til videoer hvor nogen snakker. Pauser og "øh" bliver klippet væk, og der kommer animerede undertekster.
- **Montage til musik:** til videoer uden tale, fx biler, sport, rejser og gaming. Programmet finder de vildeste
  øjeblikke ud fra lyd og bevægelse og klipper dem præcist i takt med din musik. Vælg musik for det bedste resultat.
  Med **Effekter: Vilde** (standard) bliver det en rigtig "edit":
  - Sangen starter, så videoen åbner med slowmotion og titlen (ét ord pr. beat), og så kommer droppet efter 2-4 sek.
  - Før droppet: hurtigere og hurtigere klip med flash og glitch og en "riser"-lyd. På droppet: flash, rystelse,
    farve-split og et dybt BANG.
  - Bagefter: klip der skifter mønster hver takt, speed-ramps (hurtigt - slowmotion - hurtigt), nærbilleder
    (fx speedometeret), panorering, skæve vinkler, zoom-blur og whip-pan overgange med whoosh-lyd, og zoom på
    hvert bas-slag. Farverne får et filmisk look.
  - Med Claude-nøgle kigger Claude på billederne, vælger klippene og siger hvor der skal zoomes ind.
  - **Effekter: Rolige** giver de gamle, enkle klip på beatet.
- **Automatisk** (standard) vælger selv ud fra, om der bliver talt.

Programmet forstår dansk tale, fjerner "øh" og pauser og skriver "FØLG FOR MERE" til sidst.
Teksten til sidst følger videoens sprog.
Første gang hentes talemodellen (1,6 GB), så den første video tager et par minutter ekstra.

## Download and install (English)

1. Click the green **Code** button on GitHub and choose **Download ZIP**, or run `git clone`.
2. Right-click the ZIP, choose **Properties**, tick **Unblock** and click OK. This stops Windows from
   blocking the .bat files.
3. Unzip it anywhere.
4. Double-click **`setup.bat`**. It installs FFmpeg, yt-dlp, Node.js, Python packages and the HyperFrames
   engine using winget, pip and npm, and takes 5-10 minutes. If SmartScreen says "Windows protected your PC",
   click **More info** and then **Run anyway**.
5. Double-click **`ReelMaker.bat`** to start.

## How to use it

1. **One time only:** double-click `setup.bat`. It installs everything it needs.
2. **Make a reel:** double-click `ReelMaker.bat` to open the app window. It's a local web UI shown in
   Microsoft Edge app mode, and its labels are in Danish. You can also drag a video onto the .bat file or
   into the window. The older tkinter window is still available: `.venv\Scripts\pythonw.exe -m reelmaker.gui`. For a no-window version, drag a video onto `MakeReel.bat`,
   or double-click it and paste a file path or a YouTube/TikTok/Instagram link.
3. `MakeReel.bat` asks three optional questions. Press Enter to skip any of them:
   - **Hook text**: the big title shown in the first seconds, for example `I quit my phone for 30 days`.
   - **Reference reel link**: a reel whose style you like. ReelMaker copies its length and pacing.
   - **Max length** in seconds. The default is 60.
4. The output folder opens when it's done. Inside you'll find:
   - `reel.mp4`: the finished video
   - `cover.jpg`: a thumbnail
   - `post.txt`: a title, caption and hashtags to copy and paste
   - `captions.srt`: subtitles you can upload separately
   - `edit_plan.json`: the parts of the original video that were kept

## What it does (the agents)

| Agent | Tool | Job |
|---|---|---|
| Fetcher | yt-dlp | Downloads the video when you give it a link |
| Reference Analyst | yt-dlp + FFmpeg | Studies a reel you like: its length, how often the shot changes, how much silence |
| Transcriber | OpenAI Whisper (faster-whisper) | Writes down every word with exact timing |
| Director | built-in rules, or Claude AI | Picks the best part, cuts pauses and "um"/"uh", writes the hook and hashtags. For videos without speech it switches to **montage mode**: it scans loudness, motion and shot changes, finds the music's beat, and picks the most exciting moments, cut exactly on the beat. With a key, Claude looks at frames to choose them. The **insane edit** (default) also reads the song's structure (drops, bass hits) and plans slow-motion intro, build-up, impact and hype section shot by shot |
| FX engine | numpy + OpenCV + FFmpeg | Renders the insane edit frame by frame: slow motion and speed ramps with motion blur, a virtual camera (close-ups, pans, push-ins, dutch tilt), zoom pulses and shake on the bass, zoom/whip/flash/glitch transitions, colour grade, letterbox, and synthetic whoosh/impact/riser sound effects |
| Cutter | FFmpeg | Cuts the video and turns it into 9:16: face-tracking crop for talking heads, blurred background for everything else. Adds zoom punch-ins on jump cuts and a light colour/sharpness boost |
| Animator | HyperFrames | Animated captions with the spoken word highlighted, a pop-in hook title, a progress bar and a "follow" call-to-action |
| Finisher | FFmpeg | Cleans the voice (rumble cut, light denoise, compression), sets loudness to -14 LUFS like the apps use, then exports a 1080x1920 H.264 MP4 |

## Optional: let Claude AI direct the edit

Set an Anthropic API key and the Director agent asks Claude to pick the most engaging moments and
write the hook, title and hashtags for you:

```
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Open a new window after running `setx`. Without a key, the built-in rules do the same jobs, but
you'll need to type the hook text yourself.

## Command line (more control)

```
.venv\Scripts\python.exe -m reelmaker VIDEO_OR_LINK [options]
```

| Option | What it does |
|---|---|
| `--mode montage` | `auto` (default), `speech` (edit around what's said) or `montage` (best moments cut to the music, for videos without talking) |
| `--fx classic` | Montage style: `insane` (default: speed ramps, slow motion, close-ups, shake, flashes, sound effects) or `classic` (plain cuts on the beat) |
| `--text-language da` | Language for the hook and post text when the video has no speech |
| `--max-duration 45` | Longest the reel may be, in seconds |
| `--count 3` | Make 3 different reels from the best moments of one long video |
| `--hook "text"` | The big title in the first seconds |
| `--reference LINK` | Copy the length and pacing of a reel you like |
| `--pacing chill/normal/fast` | How hard to cut pauses |
| `--caption-style bold/box/clean` | Caption look. `box` puts a coloured box behind the spoken word |
| `--accent "#FF3366"` | Highlight colour |
| `--reframe auto/face/smart/crop/blur/fit` | How the video becomes 9:16. `smart` is a full-screen crop that follows the action |
| `--music song.mp3` | Background music. It gets quieter automatically while someone talks |
| `--cta "LINK IN BIO"` / `--no-cta` | Change or remove the end call-to-action. By default it's "follow for more" in the video's language |
| `--no-captions`, `--no-zoom`, `--no-progress`, `--no-enhance`, `--keep-fillers` | Turn a feature off |
| `--whisper-model small` | Speech model. The default, `auto`, uses large-v3-turbo (most accurate, especially for Danish) and switches to `small` for videos over 20 minutes |
| `--language es` | Force the speech language. By default it's detected automatically |
| `--platform tiktok/reels/shorts/facebook` | Keep within that platform's length limit |
| `--ai on/off` | Force the Claude Director on or off |
| `--animator ass` | Use the simpler backup caption renderer instead of HyperFrames |

Other commands:

```
.venv\Scripts\python.exe -m reelmaker analyze LINK      # study a reference reel and save style.json
.venv\Scripts\python.exe -m reelmaker doctor            # check that everything is installed
```

## Notes

- The first run downloads the Whisper speech model (about 1.6 GB). Later runs are faster.
- Transcripts are cached in `work\`, so remaking the same video with another style is quick.
  You can delete `work\` at any time to free up space.
- If a link stops downloading, update the downloader:
  `.venv\Scripts\python.exe -m pip install -U yt-dlp`
- Only download videos you have the right to use.

## Credits and license

ReelMaker is released under the [MIT License](LICENSE). It builds on these tools, which `setup.bat` installs
separately (none of them are included in this repository):
[FFmpeg](https://ffmpeg.org), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running OpenAI's Whisper
models, [HyperFrames](https://github.com/heygen-com/hyperframes) with [GSAP](https://gsap.com),
[yt-dlp](https://github.com/yt-dlp/yt-dlp), [OpenCV](https://opencv.org) with the YuNet face detector, and
optionally the [Claude API](https://docs.anthropic.com).
