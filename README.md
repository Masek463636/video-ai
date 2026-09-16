# video-ai

AI-first short-form video editor. The first product goal is deliberately narrow:

> **voiceover in -> assets + captions + motion -> vertical short out**

No timeline knowledge should be required from the user.

## V0.2 status

The local pipeline now has these pieces:

- local speech transcription through optional `faster-whisper`
- deterministic first-pass director that turns timed words into 1-3 second semantic scenes
- free image/video discovery through Wikimedia Commons (no API key)
- source/license manifest for downloaded media
- local FFmpeg renderer for 1080x1920 H.264/AAC output
- image crop + zoom/pan motion
- video B-roll support
- burned ASS captions in a vertical safe zone
- one-command `create` flow

The first benchmark is a real 22.5-second reference Short supplied by the project owner. We use it as a quality target, not as training data committed to this public repository.

## Architecture

```text
voiceover
   |
   v
[faster-whisper -> timed words]
   |
   v
[director -> ShotPlan JSON]
   |
   v
[Commons asset search + ranking]
   |
   v
[FFmpeg renderer + ASS captions]
   |
   v
final.mp4
```

The `ShotPlan` is the core contract. Transcription, directing, asset providers and renderers can be replaced independently.

## Requirements

- Python 3.11+
- FFmpeg + ffprobe on PATH
- internet only when downloading stock/archive assets

For fully automatic audio transcription install the optional extra:

```bash
pip install -e ".[transcribe]"
```

## Windows quick start

```powershell
git clone https://github.com/Masek463636/video-ai.git
cd video-ai
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[transcribe]"
```

Make sure `ffmpeg` and `ffprobe` are available on PATH.

Then:

```powershell
video-ai create "voice.mp3" -o "final.mp4" --work-dir ".\work" --language ru
```

On first use faster-whisper downloads the selected model. `small` is the default.

The working directory keeps the intermediate artifacts:

```text
work/
  transcript.json
  shot_plan.json
  shot_plan.materialized.json
  assets/
    scene_000.jpg
    scene_001.jpg
    assets_manifest.json
  render/
    clips/
    captions.ass
    base.mp4
```

`assets_manifest.json` stores source URLs, author/credit and license metadata returned by Wikimedia Commons. This is important because open media licenses may require attribution.

## Separate commands

You can run each stage independently while we tune quality:

```bash
video-ai transcribe voice.mp3 -o transcript.json --language ru
video-ai plan transcript.json --audio voice.mp3 -o shot_plan.json
video-ai assets shot_plan.json --dir assets -o shot_plan.materialized.json
video-ai render shot_plan.materialized.json -o final.mp4 --work-dir render-work
```

Or start from an already-created ShotPlan and do only asset resolution + rendering:

```bash
video-ai make shot_plan.json -o final.mp4 --work-dir work
```

## Current limitation

V0.2 proves the end-to-end pipeline, but asset selection is still heuristic. A Russian phrase such as `я расстроился из-за экзамена` can fall back to a simple English concept like `sad student student exam`, but the next quality milestone is a real semantic director/ranker that understands the scene and chooses the most useful B-roll rather than the first plausible search result.

There is also no music/SFX system, face/object-aware cropping, generated visual fallback, desktop UI, or automatic quality-control pass yet.

## Product milestones

- **V0.1** local transcription + timed captions + semantic shot plan ✅
- **V0.2** free asset discovery + vertical FFmpeg renderer + one-command pipeline ✅
- **V0.3** semantic asset ranking, better scene queries, face/object-aware crops
- **V0.4** SFX/music rules + automatic QC and reranking
- **Alpha** Windows desktop: `Upload voiceover -> Create Short`
- **Later** web/mobile client after the editing engine itself is good enough

## License / upstream

The product architecture is independent. We are studying compatible open-source approaches such as `browser-use/video-use`; any substantial third-party code reused later must retain the notices required by its license.
