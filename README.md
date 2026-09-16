# video-ai

AI-first short-form video editor. The first product goal is deliberately narrow:

> **voiceover in -> assets + captions + motion + sound -> vertical short out**

No timeline knowledge should be required from the user.

## V0.4 status

V0.4 now closes the first automatic quality loop:

- local speech transcription through optional `faster-whisper`
- deterministic first-pass director that turns timed words into 1-3 second scenes
- multi-query Wikimedia Commons retrieval
- candidate pooling + metadata/geometry ranking + repetition penalty
- optional local **CLIP text-image reranking** (`--semantic`)
- CLIP similarity + retrieval score persisted into the ShotPlan
- optional OpenCV face detection + saliency fallback
- focal-point-aware 9:16 crop and Ken Burns motion
- scene QC including weak semantic-match detection
- **automatic failed-scene replacement** from the next ranked candidate
- configurable QC repair passes
- automatic music mood + SFX cue planning
- **real FFmpeg SFX mixing** into the voiceover
- optional user-supplied background music mixing
- procedural fallback SFX, so the MVP works without shipping a sound pack
- one-command `create` flow
- GitHub Actions unit-test workflow

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
[metadata rank + dedupe + optional CLIP rank]
   |
   v
[focus analysis -> smart 9:16 crop]
   |
   v
[QC -> failed scene? -> next candidate -> QC again]
   |
   +--> [audio-plan -> SFX/music mix]
   |
   v
[FFmpeg motion + captions + mixed audio -> MP4]
```

## Install

Light local pipeline:

```powershell
pip install -e ".[local]"
```

Full V0.4 stack with CLIP:

```powershell
pip install -e ".[full]"
```

Requirements:

- Python 3.11+
- FFmpeg + ffprobe on PATH
- internet for first model download and stock/archive retrieval

## Recommended V0.4 command

```powershell
video-ai create "voice.mp3" `
  -o "final.mp4" `
  --work-dir ".\work" `
  --language ru `
  --semantic `
  --repair-passes 2
```

Optional background music:

```powershell
video-ai create "voice.mp3" `
  -o "final.mp4" `
  --work-dir ".\work" `
  --language ru `
  --semantic `
  --music ".\music.mp3"
```

Optional custom SFX folder:

```text
sfx/
  whoosh.wav
  impact.wav
  notification.wav
  bass_hit.wav
  cash_pop.wav
```

```powershell
video-ai create "voice.mp3" -o final.mp4 --work-dir work --semantic --sfx-dir .\sfx
```

Without a custom SFX pack, V0.4 synthesizes simple royalty-free procedural placeholders locally with FFmpeg.

## Work directory

```text
work/
  transcript.json
  shot_plan.json
  shot_plan.materialized.json
  audio_plan.json
  qc.json
  mixed_audio.m4a
  assets/
    scene_000.jpg
    scene_001.jpg
    assets_manifest.json
  audio_mix/
    sfx_*.wav
  render/
    clips/
    captions.ass
    base.mp4
```

`assets_manifest.json` includes the queries tried, top candidates, retrieval scores and CLIP similarity where semantic mode was used.

`qc.json` records which scenes passed or failed and why. `create` / `make` automatically retry failed scenes by moving down the ranked candidate list.

## Separate commands

```powershell
video-ai transcribe voice.mp3 -o transcript.json --language ru
video-ai plan transcript.json --audio voice.mp3 -o shot_plan.json
video-ai assets shot_plan.json --dir assets -o materialized.json --semantic
video-ai qc materialized.json -o qc.json
video-ai render materialized.json -o final.mp4 --work-dir render-work
```

## Current limitations

V0.4 is still an MVP engine, not a finished consumer editor.

- Wikimedia Commons alone is not enough for every topic; more stock providers are needed.
- CLIP ranking currently applies to image candidates; video B-roll still uses metadata/geometry ranking.
- Procedural fallback SFX are functional placeholders, not polished production sound design.
- There is no generated-image fallback yet when retrieval completely fails.
- There is no Windows GUI yet.

## Product milestones

- **V0.1** local transcription + timed captions + shot plan ✅
- **V0.2** free asset discovery + vertical FFmpeg renderer ✅
- **V0.3** candidate ranking + dedupe + face-aware crop ✅
- **V0.4** CLIP rerank + semantic QC + auto-repair + actual SFX/music mixing ✅
- **V0.5** more stock providers + generated fallback + better subtitle styles + visual QC
- **Alpha** Windows desktop: `Upload voiceover -> Create Short`
- **Later** web/mobile client after the editing engine itself is good enough

## License / upstream

The product architecture is independent. We are studying compatible open-source approaches such as `browser-use/video-use`; any substantial third-party code reused later must retain the notices required by its license.
