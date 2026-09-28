# Premium v3

Premium v3 is an isolated whole-story editor.

Core rule: plan the story first, then search and assemble.

Pipeline:

1. Local transcription with word timings.
2. One Gemini batch call plans the complete story arc and all beats.
3. Media retrieval runs locally against the existing providers without per-candidate Gemini calls.
4. One Gemini multimodal batch call reviews the complete chosen sequence and requests only genuinely weak replacements.
5. Repairs are searched locally; duplicate checks and moment selection stay local.
6. Intentional callbacks may reuse an earlier familiar visual.
7. Existing low-level Premium renderer, captions and audio mixer are reused, but the editorial decisions come from the v3 story blueprint.
8. Final overlays run without Gemini.

This keeps Gemini use bounded and makes neighboring scenes part of one edit rather than independent decisions.
