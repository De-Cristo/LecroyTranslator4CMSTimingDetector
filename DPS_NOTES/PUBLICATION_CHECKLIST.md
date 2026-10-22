# Publication candidates: not approved for submission

N1-6 local work completed on 22 October 2026. Official template, DP metadata,
cover handling and collaboration approval remain pending. No submission was made.

## Files for convener review

Each note is packaged independently in `note1_mcp_reference/` and
`note2_dual_threshold/`:

- `publication_candidate_slides.pdf`: title, CMS author/contact, abstract and
  content; **no additional A4 cover**. Use this if the publication system adds it.
- `cover.pdf`: separate local A4 cover, sharing the note's title and abstract.
- `publication_candidate_with_cover.pdf`: local preview with exactly one A4
  cover followed by the candidate slides. Do not upload this if the system
  generates the cover itself.
- `candidate_report.json`: page counts, checksums, build checks and pending status.
- `source_bundle.zip`: editable **internal-review** sources, including both
  `main.tex` and `publication_candidate.tex`; not a sanitized public-source release.

The combined `both_notes_review_v1.pdf` is for discussion only, not a two-note
submission. Existing `slides.pdf` and `note_with_cover.pdf` remain full review
versions. All candidate PDFs retain the unapproved-draft notices.

Note 1's candidate omits the review queue and four internal historical backups:
end difference, bar-minus-MCP, clock-phase illustration and clock-estimator
comparison. Their internal review versions and source files are preserved.
The validated residual-segment-offset diagnostic remains in the candidate.
In particular, the unvalidated clock-estimator image cannot appear in that PDF.

Note 2's candidate omits its closing review-status slide but retains permanent
scientific qualifications beside the relevant results. The current Note2
comments are closed by explicit author decisions for their stated scope;
N2-3 and N2-4 are temporarily closed for the exploratory version pending a
page-by-page review. This does not validate a precision model, establish total
calibration uncertainties or authorize release. N1-6 packaging itself closes
no scientific item.

## Completed locally

- Verified and linked detector/readout and method citations in both notes;
  see `review_2026-10-22_publication/citation_audit.md`.
- Kept exact `CMS Preliminary Phase 2` labels on all plot exports.
- Kept separate titles/abstracts for the two notes, shared with their own covers.
- Added separately numbered candidate builds, with excluded backup pages absent.
- Preserved all calibration inputs, fits and plotted data unchanged.

## Convener decisions still needed

1. Confirm that the supplied working adaptation is acceptable, or provide the
   approved CMS DP template. Its own README says it is not officially verified.
2. Assign the two DP identifiers and confirm the publication date and metadata.
   No number is invented; the drafts explicitly have none assigned.
3. Confirm slides-only versus externally supplied A4 cover for the submission
   workflow. Approve the author/contact line and final publication wording.
4. Complete collaboration review and authorize release. Remove internal notices
   only after approval and resolution of remaining scientific review items.

Rebuild with `python DPS_NOTES/draft_support/build.py --candidates`.
No ROOT processing or figure regeneration is needed for this packaging change.
