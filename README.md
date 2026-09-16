# video-ai

AI-first short-form video editor. The first product goal is deliberately narrow:

> **voiceover in -> assets + captions + motion -> vertical short out**

No timeline knowledge should be required from the user.

## V0.4 status — in progress

V0.4 is the quality-control layer on top of the V0.3 pipeline.

Already implemented:

- local speech transcription through optional `faster-whisper`
- deterministic first-pass director that turns timed words into 1-3 second semantic scenes
- multi-query Wikimedia Commons retrieval
- candidate pooling + metadata/geometry ranking + repetition penalty
- optional OpenCV face detection + saliency fallback
- focal-point-aware 9:16 crop and Ken Burns motion
- optional local CLIP image/text ranker module (`video-ai[semantic]`)
- automatic music mood + SFX cue planning to `audio_plan.json`
- local scene QC pass to `qc.json`
- one-command `create` flow now emits transcript, materialized plan, audio plan, QC report and final MP4

Still being wired into the automatic path:

- CLIP reranking of the complete candidate pool before download/selection
- automatic replacement/re-search of scenes that fail QC
- actual SFX/music mixing (the cue plan exists; renderer mixing comes next)
- generated-image fallback when stock/archive search has no good result

The first benchmark is a real 22.5-second reference Short supplied by the project owner. We use it as a quality target, not as training data committed to this public repository.

## Architecture

```text
voiceover
   |
   v
[faster-whisper -> timed words]
   |
   v
[director -> ShotPlan]
   |
   v
[multi-query retrieval -> candidate pool]
   |
   v
[metadata rank + dedupe + optional semantic rank]
   |
   v
[focus analysis -> smart 9:16 crop]
   |
   +--> [audio-plan.json]
   +--> [qc.json]
   |
   v
[FFmpeg motion + captions -> MP4]
```

## Install

Basic local pipeline:

```powershell
pip install -e ".[local]"
```

Full experimental V0.4 stack with local CLIP dependencies:

```powershell
pip install -e ".[full]"
```

Requirements:

- Python 3.11+
- FFmpeg + ffprobe on PATH
- internet for the first model download and stock/archive retrieval

## One-command flow

```powershell
video-ai create "voice.mp3" -o "final.mp4" --work-dir ".\work" --language ru
```

It creates:

```text
work/
  transcript.json
  shot_plan.json
  shot_plan.materialized.json
  audio_plan.json
  qc.json
  assets/
    scene_000.jpg
    scene_001.jpg
    assets_manifest.json
  render/
    clips/
    captions.ass
    base.mp4
```

## Debug/tuning commands

```powershell
video-ai audio-plan work\shot_plan.materialized.json -o work\audio_plan.json
video-ai qc work\shot_plan.materialized.json -o work\qc.json
```

`audio_plan.json` currently describes the chosen music mood and exact timestamps for cues such as `whoosh`, `impact`, `notification`, `bass_hit` and `cash_pop`.

`qc.json` catches obvious failures such as missing assets, repeated assets, extreme scene durations and missing focus metadata before we invest in a final quality pass.

## Product milestones

- **V0.1** local transcription + timed captions + semantic shot plan ✅
- **V0.2** free asset discovery + vertical FFmpeg renderer + one-command pipeline ✅
- **V0.3** candidate pooling/ranking + dedupe + face-aware focal crop ✅
- **V0.4** multimodal ranker module + audio cue planner + QC engine 🟡
- **V0.4 completion** automatic semantic rerank + failed-scene replacement + real audio mixing
- **Alpha** Windows desktop: `Upload voiceover -> Create Short`
- **Later** web/mobile client after the editing engine itself is good enough

## License / upstream

The product architecture is independent. We are studying compatible open-source approaches such as `browser-use/video-use`; any substantial third-party code reused later must retain the notices required by its license.
