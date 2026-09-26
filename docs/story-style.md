# Second style: stories and memes

`story` is an opt-in editing pipeline. The local studio offers **Классический**
and **Истории и мемы**. The existing `create`/`render` workflow keeps its presets.
This is a first version of the second style, not an exact reproduction of the
reference videos or a public SaaS deployment.

## Start on Windows

From your existing video-ai installation, with FFmpeg and the transcription
dependencies already installed:

```powershell
python -m pip install -e ".[story]"
python -m video_ai.web
```

Use the Python environment that already renders your videos. If using `.venv`,
substitute `.\.venv\Scripts\python.exe` for `python` in both commands.
Open **Истории и мемы**, select the MP3 and click **Создать ролик**.
The studio runs on your computer at http://127.0.0.1:8765. Keep its terminal open.

The second style needs `GEMINI_API_KEY` for planning and visual selection.
Pexels/Pixabay keys enable stock video searches. Without them, generic footage
can fall back to illustrations. Commons/Openverse image search needs no extra
key. API limits and availability still apply; this mode has no local CLIP-only
replacement for Gemini.

## Your two folders

Put packs inside the project:

- `memes/`: reaction clips and images, e.g. MP4, GIF, PNG, JPEG, WebP.
- `elements/`: arrows, pointing hands, symbols and other overlays. Transparent
  PNG/WebP works best. MP4, GIF, WebM, MOV and MKV are also supported.

Subfolders and Cyrillic filenames work. Descriptive filenames help when you
inspect the pack, but Gemini also describes the actual image or three sampled
video frames. It does not identify a person by their face.

Start with 20–40 useful files per folder. A run indexes at most the first 120
files in sorted order per folder. First indexing uses extra Gemini requests;
successful descriptions are cached in `.video-ai-index` within the pack and
reused until a file's path, size or modification time changes. Preview images
are sent to Gemini for indexing and selection. Unverified files are skipped if
indexing fails. Empty or missing folders are allowed: there will simply be no
local memes/elements. The existing `stickers/` folder belongs to classic mode.

Turning off reactions skips the packs and generated effect sounds. It still
allows explanatory comparison arrows.

## What it does

- Plans scenes from the whole voiceover, preserving every word and its timing.
- Searches Pexels/Pixabay for generic actions and Commons/Openverse for images.
  It judges a bounded shortlist with Gemini rather than accepting the first hit.
- Requires source titles/descriptions to mention a planned named entity or one
  of its aliases. Generic stock is not used for that subject. This is a metadata
  guard, not proof of historical authenticity; inspect factual videos before use.
- Builds two-object comparisons, photo/video panels, local meme reactions and
  collages with a local element. Adds short captions, labels and optional sounds.
- Keeps images with genuine transparency as cutouts. Other images appear as
  cards; a picture with a painted checkerboard is not treated as transparent.
- Samples longer clips for an action window. Short clips loop instead of holding
  their final frame. Scene cuts are rounded on one shared frame timeline.

Search is limited to those providers. Google-wide image crawling, exact archival
video retrieval, semantic background-removal QC and arbitrary animated map
graphics are not implemented. The fixed layouts also cannot yet guarantee a
clear face or subject underneath every overlay. Reference-level quality needs
real voiceover/pack trials and further tuning.

If no candidate passes, the job stops with an explanation instead of silently
inserting an unrelated picture. `sources.json` retains source links, creators
and reported license metadata; check the source's reuse/attribution terms when
publishing. A pack's inclusion does not establish rights to its contents.

## Command-line testing

```powershell
video-ai story ".\voice5.mp3" `
  -o ".\voice5-story-v1.mp4" `
  --work-dir ".\work-voice5-story-v1" `
  --meme-dir ".\memes" `
  --elements-dir ".\elements" `
  --language ru
```

To reuse timings, add `--transcript ".\work-voice5-v1\transcript.json"` **only
if it belongs to that exact MP3**. Otherwise let the app transcribe it again.

For optional CPU background removal:

```powershell
python -m pip install -e ".[cutouts]"
```

Then add `--cutouts` to the CLI command. The first use may download a model.
If removal is unavailable or its alpha mask is unusable, the original image
card is retained. Background removal is not enabled by the browser UI in v1.

Work files include `transcript.json`, `packs.json`, `story.plan.json`, a partial
`story.materialized.json` checkpoint, `sources.json`, downloaded assets and
render files. In the studio these are in `work-web/<job-id>/story-work/`.
After a failed search you can edit the scene's queries in `story.plan.json` and
rerun with `--story-plan <that-file>` and the same transcript/work directory.
Planning can be reused this way, but there is no automatic resume of completed
materialized scenes yet. Existing downloads and pack descriptions are cached.

## Verification

`python -m pytest tests/test_story.py tests/test_web.py` covers invalid plans,
named-entity filtering, pack caching, interrupted downloads, HTTP job routing,
and a complete controlled-provider story pipeline with real FFmpeg rendering.
Pixel checks cover both comparison objects, the final scene, animated reactions
and transparent overlays; frame counts check accumulated timing drift.
These tests do not validate real API availability or the creative quality of
Gemini's decisions.
