# Classic editing: readable reactions and source replacement

The studio classic final step enables `--composition-review`. `--shorts-fx`
now builds local reactions only: no text callouts, quantities or stock-photo cards.
Narration subtitles remain. The reaction planner may choose zero accents. Its
automatic maximum is duration/8 seconds rounded, clamped to 1–4; `--max-overlays`
can override the maximum. Studio uses automatic mode. Reactions are anchored to
spoken words, last at most 1.7 seconds, and are spaced at least four seconds apart.

The planner selects `stickers` or sibling `memes` packs. Local CLIP proposes a
file; visual review must independently approve immediate readability and relevance.
PNG/JPEG/WebP/GIF and MP4/WebM/MOV reactions are supported. Video audio is ignored.
An unavailable provider never triggers a text/stock-photo fallback. Reviewed emoji
use 34–40% of canvas width, memes 70–80%, subject to free space and aspect ratio.
No small fallback slot is used. This is an overlay implementation, not a separate
full-screen meme scene. Crowded/text-dependent/ambiguous reactions are rejected.

Composition review samples three actual cropped frames per span. Missing actions
can trigger new stock-video retrieval instead of futile recropping. Up to six
spans per render can search replacements, each with at most two short queries and
four inspected candidates per query. The original core action and narration are
used in retrieval and actual full-resolution crop verification. Named/locked
entities and non-stock source modes are excluded from generic replacement.
No match preserves the original and explicitly reports `unresolved`, not success.

Framing-only failures still compare up to four crops/windows. Requirements remain
fixed across candidates; full-resolution replacements are verified before an
atomic file replacement. Processing failures preserve the original rendered clip.
Mood and actor details may differ for anonymous illustrative stock footage.
Stock footage is not required to contain literal narrated dialogue as screen text.

Reaction relevance and background placement are separate requests. Text and photo
inserts from old saved overlay lists are omitted in composition-review mode.
Faces/actions/objects and subtitle space are protected. Invalid coordinate responses
get one correction attempt. Previews explicitly convert to full-range JPEG to avoid
the observed Windows FFmpeg MJPEG error.

`WORK/composition/review.json` records status, requirements, retrieval attempts,
replacement source attribution, validation and FFmpeg failures. It also stores
placement responses. `overlays.reviewed.json` contains actual surviving reactions.
The input shot plan is not overwritten; replacements are baked into rendered spans
and recorded in the report. Console output summarizes actual statuses and accents.

Verification includes real FFmpeg render→replacement→video reaction→captions→audio
muxing with controlled model/retrieval responses, plus failed verification, source
identity, timeline, range and UI job tests. Live Gemini, stock availability and
subjective resemblance to references require real user footage/API runs; synthetic
integration success does not certify those. Three sampled frames are not tracking
or frame-by-frame QC. Provider rate limits can still delay requests.
