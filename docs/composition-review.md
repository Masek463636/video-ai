# Classic composition review

The studio's classic final render now enables `--composition-review`. This is
also available with `render --overlays-file PATH --composition-review` to reuse
existing source clips and effects. It requires Gemini and adds bounded requests:
one per visual span, one additional request if that span needs repair, and one
per non-text insert. A provider failure disables subsequent review requests for
that render; unverified optional image/GIF inserts are omitted.

Review samples actual cropped renderer output at three times. For an unsuitable
video span it compares up to four alternatives: left/right focus and middle/end
source windows. Only a valid explicitly approved alternative replaces the clip.
No new footage is downloaded by this pass. If none is suitable, the source stays
unchanged and the report marks it `unresolved`; this is not a passed quality gate.
Three sampled frames cannot guarantee visibility between samples.

Insert review checks narration relevance and protected face/object/action boxes
in the final cropped background at three times during the insert. Conservative
slots reserve subtitles and overlapping quantity labels. Reviewed inserts appear
in place without travel/rotation, so they do not cross protected areas. An insert
with no available slot is skipped. Duplicate insert labels are suppressed; the
main narration subtitles and standalone quantity callouts remain. The latter
retain their existing placement. Detection is model-based, not a tracking system.

`WORK/composition/review.json` records source decisions, reasons, repairs,
unresolved shots, rejected inserts, and placement boxes. Rendered previews are
kept beside it. `overlays.reviewed.json` can be reused with `--overlays-file`.
The input materialized plan is not overwritten; scene repairs are recorded in
`review.json` and baked into the rendered span files, not into a new shot plan.

This is not yet an automatic replacement search or frame-by-frame final-video
QC. Review authentic API behavior on real jobs; unit tests use controlled model
responses and FFmpeg integration checks cover crop and overlay geometry.
