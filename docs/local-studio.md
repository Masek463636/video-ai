# Local Video AI Studio

This is a **local-only** browser interface with two editing styles. Classic mode wraps the existing two-pass CLI workflow; the opt-in story mode builds comparisons, photo/video panels and meme inserts. It does not provide public hosting/payment/accounts.

## Windows setup

From the project folder with the existing dependencies, FFmpeg and API keys installed:

```powershell
python -m pip install -e ".[story]"
python -m video_ai.web
```

The browser opens at http://127.0.0.1:8765. Keep the terminal open while rendering; Ctrl+C stops the application. On subsequent launches you can use `start-studio.bat` (prefers `.venv` if present). That environment must contain the same video-ai dependencies. The alternative installed command is `video-ai-web`.

Saved Windows User environment keys are inherited by a new terminal (restart the terminal application if necessary). The UI reports **presence**, not live API validity. It never displays or persists key values. Classic mode requires Gemini plus at least one of Pexels/Pixabay. Story mode requires Gemini; stock keys are optional for its image-only fallback. Keys are passed only to the renderer child process; they are redacted from the UI log. API requests still go to the configured external providers, as in the CLI.

Choose a style, upload an MP3 (100 MiB max, up to 180 seconds), optionally turn off reactions, then create. Processing is one job at a time. Progress represents actual stages, not an estimated percentage. Classic mode uses the same presets as the successful voice4/voice5 commands, with stickers from `stickers/`. Story mode uses `memes/` and `elements/`; see [pack preparation, supported scenes and limitations](story-style.md).

Results, source audio, work files and the last 120 log lines are kept in `work-web/<job-id>/`. Download the final MP4 from the UI. Existing outputs are never overwritten. Reloading the page preserves jobs; interrupted jobs are marked as errors on restart. There is no automatic cleanup yet: remove old job directories with the app stopped to reclaim space. A failed job can be rerun by uploading the audio again.

## Boundary

The HTTP listener is fixed to IPv4 loopback. Host/Origin checks and a required custom upload header prevent browser cross-origin uploads/DNS rebinding. Upload names are server-generated, lengths bounded, and audio inspected with ffprobe before starting the CLI. No shell interpolation is used. Only known static assets, job state and completed outputs are exposed. Do not proxy this server onto the internet: production needs authentication, per-user storage, encrypted BYOK, resource isolation, quotas and a persistent queue.

## Validation

`python -m pytest tests/test_web.py` exercises real HTTP uploads, invalid media, busy-job rejection, cross-origin protection, two-stage command wiring with a controlled renderer, download and persisted history. The actual AI pipeline requires the user's API keys and models; a mocked workflow test is not evidence of a full AI render.
