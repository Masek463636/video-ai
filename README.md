# video-ai

AI-first short-form video editor. The first product goal is deliberately narrow:

> **voiceover in -> editable shot plan -> vertical short out**

No timeline knowledge should be required from the user.

## MVP target

Input:
- MP3/WAV voiceover
- optional transcript / timed transcript
- optional local media folder

Output:
- 1080x1920 MP4
- scene changes driven by the spoken meaning
- B-roll / image slots
- automatic pan/zoom for still images
- subtitles
- room for music and SFX cues

The first benchmark is a real 22.5-second reference Short supplied by the project owner. We use it as a quality target, not as training data committed to this public repository.

## Architecture

```text
voiceover
   |
   v
[probe / transcript]
   |
   v
[director -> ShotPlan JSON]
   |
   +--> [asset provider / local assets / later: stock search]
   |
   v
[renderer -> FFmpeg]
   |
   v
final.mp4
```

The `ShotPlan` is the core contract. AI providers can change later without rewriting the renderer.

## V0 scope

V0 does **not** try to be CapCut.

1. Read voiceover duration and media metadata.
2. Accept a deterministic ShotPlan JSON.
3. Render a vertical video locally with FFmpeg.
4. Support image/video scenes, fit-to-vertical crop, and simple Ken Burns motion.
5. Keep provider interfaces separate so transcription, LLM directing, and asset search can be added incrementally.

## Requirements

- Python 3.11+
- FFmpeg + ffprobe on PATH

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .

video-ai probe path/to/voiceover.mp3
video-ai validate examples/shot_plan.example.json
```

Rendering will be enabled in the next milestone once the scene asset contract is locked against the reference Short.

## Product milestones

- **V0** local engine: audio metadata -> shot plan -> MP4
- **V0.1** local transcription + timed captions
- **V0.2** AI director: transcript -> semantic scenes
- **V0.3** stock / archive asset search + ranking
- **V0.4** SFX/music rules + automatic QC
- **Alpha** Windows desktop: `Upload voiceover -> Create Short`

## License / upstream

We are studying and may reuse compatible MIT-licensed ideas/components from `browser-use/video-use`, while keeping this repository's product architecture independent. Any copied substantial MIT-licensed code must retain its required copyright/license notice.
