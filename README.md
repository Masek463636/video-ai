# video-ai

AI-first short-form video editor. The first product goal is deliberately narrow:

> **voiceover in -> assets + captions + motion -> vertical short out**

No timeline knowledge should be required from the user.

## V0.3 status

The local pipeline now has these pieces:

- local speech transcription through optional `faster-whisper`
- deterministic first-pass director that turns timed words into 1-3 second semantic scenes
- several search-query variants per scene instead of betting on one query
- free image/video discovery through Wikimedia Commons (no API key)
- candidate pooling + semantic/title/description scoring
- portrait/resolution/video bonuses for Shorts-friendly media
- duplicate/repetition penalty so neighboring scenes are less likely to look the same
- optional local OpenCV face detection + cheap saliency fallback
- focal-point metadata persisted into the ShotPlan
- focal-point-aware 9:16 crop and zoom/pan in FFmpeg
- source/license manifest with top candidates and scores for debugging
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
[multi-query asset retrieval]
   |
   v
[rank + dedupe + focal analysis]
   |
   v
[FFmpeg smart crop + motion + captions]
   |
   v
final.mp4
```

The `ShotPlan` is the core contract. Transcription, directing, asset providers, ranking and rendering can be replaced independently.

## Requirements

- Python 3.11+
- FFmpeg + ffprobe on PATH
- internet only when downloading stock/archive assets

Recommended local install for V0.3:

```bash
pip install -e ".[local]"
```

That installs local transcription plus optional OpenCV vision. If OpenCV is not installed, the renderer still works and falls back to centered crops.

## Windows quick start

```powershell
git clone https://github.com/Masek463636/video-ai.git
cd video-ai
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[local]"
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

`assets_manifest.json` stores source URLs, author/credit, license metadata, queries tried, top candidate scores and the chosen visual focus. This is useful both for attribution and for tuning why the AI picked a bad frame.

## Separate commands

```bash
video-ai transcribe voice.mp3 -o transcript.json --language ru
video-ai plan transcript.json --audio voice.mp3 -o shot_plan.json
video-ai assets shot_plan.json --dir assets -o shot_plan.materialized.json --limit 20
video-ai render shot_plan.materialized.json -o final.mp4 --work-dir render-work
```

Or start from an already-created ShotPlan and do only asset resolution + rendering:

```bash
video-ai make shot_plan.json -o final.mp4 --work-dir work --limit 20
```

## What V0.3 actually improved

V0.2 often accepted the first plausible search result. V0.3 instead builds a candidate pool from several retrieval views and compares them using the scene query + caption, Commons title/description metadata, image geometry/resolution and recent visual history.

For images, V0.3 can also detect the main face locally. If there is no face, a cheap edge-density focus estimate is used. The normalized focus point is then fed into the vertical crop and Ken Burns motion so a landscape photo is less likely to cut off the person's face.

## Current limitation

This is still not a true multimodal AI art director. The ranker does not yet visually understand whether a photo *emotionally* matches the spoken sentence; it mostly combines text metadata, composition heuristics and local face/saliency analysis.

The next quality jump is **V0.4**:

- optional local/multimodal embedding ranker (CLIP/SigLIP-style)
- music and SFX cue planner
- visual-quality/self-QC pass
- automatic reranking/replacement when a scene fails QC
- generated-image fallback when stock search has nothing useful

## Product milestones

- **V0.1** local transcription + timed captions + semantic shot plan ✅
- **V0.2** free asset discovery + vertical FFmpeg renderer + one-command pipeline ✅
- **V0.3** candidate pooling/ranking + dedupe + face-aware focal crop ✅
- **V0.4** multimodal ranking + SFX/music + automatic QC/reranking
- **Alpha** Windows desktop: `Upload voiceover -> Create Short`
- **Later** web/mobile client after the editing engine itself is good enough

## License / upstream

The product architecture is independent. We are studying compatible open-source approaches such as `browser-use/video-use`; any substantial third-party code reused later must retain the notices required by its license.
