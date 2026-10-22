# Standing author instructions for the performance notes

After resolving each review point, update the note sources and review status,
rebuild the affected PDFs and the combined review document, and verify them.
Then synchronize the notes with:

`/afs/cern.ch/work/l/lichengz/private/MTD/TB_2025/LecroyTranslator4CMSTimingDetector`

The author explicitly authorizes a scoped commit and push to its online
repository after each accepted review point. Do not treat the older "no
commits or push" analysis-work instructions as prohibiting this note workflow.
Do not stage unrelated changes, force-push, or claim synchronization until the
push is verified. Report a push failure rather than silently leaving it local.

Read `REPOSITORY_WORKFLOW.md` for scope and verification. Keep code references
consistent with the online version; preserve scientific results unless the
review decision explicitly changes them. Repository upload is not official CMS
approval or authorization to submit the notes to CMS.

In rendered notes, captions and implementation footnotes, cite code through
clickable GitHub links, not local filesystem or unlinked repository-relative
paths. Prefer immutable commit links. Before citing, verify the committed code
matches the local version; synchronize any intended changes first. Do not cite
or link results directories, audit folders or local directory names in rendered
text. Preserve provenance metadata internally; do not delete saved artifacts.
