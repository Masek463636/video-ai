# Semantic editing: first increment

This change addresses the gap between topic-related stock footage and footage
that demonstrates the narrated action.

- Preserve concrete director queries before generic Material Brain rescue rules.
- Allow consecutive scenes to explain the same subject with different actions.
- Rank candidates primarily against director intent, not their own search query.
- Limit automatic foreground accents to roughly one per twelve seconds, with no
  minimum. Respect an empty Gemini effects plan; local stickers cannot override it.
- Optionally choose a source window after footage has been downloaded.

## Try on existing downloaded assets

From the repository directory after installing the package, with GEMINI_API_KEY
already configured in your environment:

```powershell
video-ai render .\work\shot_plan.materialized.json -o .\semantic-test.mp4 --work-dir .\work-semantic-test --reference-framing --select-moments
```

The original plan is not overwritten. The selected offsets are saved to
`work-semantic-test/shot_plan.moments.json`; decisions and failures are recorded
in `work-semantic-test/moments/moments.json`. To repeat the render without more
Gemini calls, render the saved moments plan without `--select-moments`.

The `make` and `create` commands also accept `--select-moments`. The query fixes
apply when assets are searched again; rendering an existing plan cannot replace
an unrelated downloaded clip with better footage.

Moment selection samples up to eight windows, three frames each, then makes at
most one additional Gemini request per eligible scene. It uses existing Gemini
configuration and quota. It accepts only a valid window with fit >= 70. Missing
credentials, failed extraction, API errors, or weak matches leave the current
source timing unchanged. Sparse frame sampling can miss brief events; this is
not full video understanding. Short sources and meme clips are left alone.

Old plans default to source_start=0. Render seeks to the selected source offset
in both full-frame and blurred-background layouts. Narration and subtitle timing
remain independent. Source offsets are reset when new assets are downloaded.

## Verification

New regression tests cover preserved comparison queries across repair passes,
empty effects plans, old/new JSON compatibility, invalid offsets, bounded source
windows, rejected model selections, and actual FFmpeg renders from a later
blue section of a red-then-blue source in both framing modes.

No live Gemini/stock-provider quality benchmark was run: this environment has no
provider keys, local transcription/CLIP models, or original run's source assets.
The reference and previously rendered MP4s are not clean source projects.
The original baseline already has 11 failing tests; this increment does not
claim to repair those unrelated legacy failures.
