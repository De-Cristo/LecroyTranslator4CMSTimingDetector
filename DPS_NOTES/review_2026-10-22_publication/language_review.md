# Language-only review

Applied the author's requested
[hep-text-review skill](https://github.com/De-Cristo/hep-skills/blob/1c1800886839b3800d64fc7593b0f4023db5fdb1/skills/hep-text-review/SKILL.md)
at revision `1c1800886839b3800d64fc7593b0f4023db5fdb1`.
The comparison baseline is analysis-repository commit `37eded5`.

## Scope

The pass edits slide prose, captions and slide abstracts for concise sentences,
direct descriptions and consistent terminology. The A4 cover abstracts remain
byte-for-byte unchanged in the source. Original bibliographies, equations,
selection boundaries, numerical values, fit inputs and figure assets are
preserved. Scientific caveats about conditional errors, model validity,
cross-run transfer and reference-free calibration remain explicit.

This is a language pass, not a new physics validation or bibliography audit.
Existing numerical verification is reused. The skill's broader recommendations
to change bibliography entries or restructure the scientific argument are
outside the author's language-only scope.

## Terminology

- Profile mean: the mean observable in a selected charge bin.
- SD: the spread of selected hits; SEM: the uncertainty on the profile mean.
- Bootstrap band: the conditional interval of the fitted mean curve.
- Measured charge support: populated charge intervals within the outer profile means.
- Archived reference: the original calibration using measured channel-192 timing.
- Clock-corrected reference: the calibration using the reconstructed trigger time.
- Shape comparison: agreement between calibration functions, not event timing resolution.

## Additional author decisions

During this pass, the author also requested these changes to Note 2:

- Rename the former slide 8/30 to "Reference version shared with the MCP
  reference note" and cite the existing MCP-note entry, Ref. [6].
- Remove "Sensitivity to the refreshed reference" (formerly 9/30).
- Remove "AWC versus inverse slope coefficient: all completed fits"
  (formerly 20/30). Retain the ten-channel figure and channel-status table.
  Update the claim that all completed fits are displayed.
- Rename "What the present uncertainties do and do not include" (formerly
  23/30) to "Uncertainties".
- Remove "Review queue before publication" (formerly 30/30). Outstanding
  publication-format and approval matters remain in the publication checklist.

These explicit slide changes are separate from language polishing. Removed
slides' underlying results and assets remain available; no fits are rerun.

## Verification

The preservation check compares retained inline/display mathematics and numeric
tokens against the baseline, excluding only the three explicitly removed slides
and prose cross-references to note numbers. Cover text and bibliographies are
compared directly. Figure PDFs, figure manifests, numerical macros and included
numerical tables are checked byte-for-byte. Both note variants are compiled and
checked for overflow and unresolved citations, with rendered-page inspection.
Results are recorded in `language_review_verification.json` after completion.
