# Two CMS performance-note review drafts

Prepared 20 October 2026. These are discussion drafts, not approved CMS results
or submitted notes. The original `Note1.pdf` and `Note2.pdf` are unchanged.

## Open the drafts

- `note1_mcp_reference/slides.pdf`: MCP-reference calibration slide note.
- `note2_dual_threshold/slides.pdf`: dual-threshold/inverse-slope slide note.
- Each directory also contains `cover.pdf` (A4), `note_with_cover.pdf`
  (cover followed by slides), and `source_bundle.zip` (editable self-contained
  LaTeX, figure PDFs, numerical inputs and manifest).
- `preview/` holds rendered slides and contact sheets.

The title slide uses CMS and `cms-dpg-conveners-mtd@cern.ch`. Slide 2 and the
A4 cover read the same abstract from `metadata.tex`. Every exported plot
contains `CMS Preliminary Phase 2`; the enclosing note remains conspicuously
marked as an internal review draft. No official DP number or approval is implied.

## How this first version handles the known gaps

Note1 retains the **legacy four-channel calibration** and adds the four
**clock-corrected references used by Note2**, with their selection, fit definition
and numerical comparison. The current clock-corrected references were refitted
on 22 October 2026 after clock-quality fixes. Existing waveform and
synchronization illustrations remain labelled legacy rasters. Historical timing
images are backups; reproducible before/after timing distributions replace them
in the main result.

Note2 uses the **newer extended, measured-overlap study**, whose clock-corrected
reference is now also documented in Note1. Profile fits, spectra, curve comparisons and AWC summaries are redrawn
from frozen numerical outputs. The old ambiguous residual illustration is
replaced with an explicitly defined model-minus-model residual. The internal
AWC predictor is exploratory backup material. The ten-channel subset is
explicitly identified as an interactive selection; the full 21 completed
fits are shown as well.

These choices make both drafts readable now without pretending that analysis
versions, selections and uncertainties have already been reconciled.

## Review sequence

1. N1-4 is closed: Note1 now includes the clock-corrected reference used by Note2.
   N2-1 is closed under the author's same-nominal-setup assumption, with MCP
   reference timing in run 4399 and dual-threshold timing in run 4076.
   Cross-run stability is not independently verified; response-transfer and
   selection effects remain unquantified.
2. N2-2 is closed: all 22 frozen profiles replayed from the run4076 ROOT input.
   Exact timing/threshold decoding and cuts are documented, including the
   histogram-derived charge boundary and absence of the newer 100 ps cut.
   See `review_2026-10-22_n2_2/`; numerical results are unchanged.
3. N1-3 is closed: the note now explicitly follows the code's minus-sign convention.
4. N2-3: eight-variant diagnostics are saved in `review_2026-10-22_n2_3/`.
   At the author's request, Note2 keeps only the original linear-slope versus
   clock-corrected MCP comparison; the two audit slides are removed. Nominal
   results and conclusions are unchanged. N2-3 is temporarily closed by author
   approval for this exploratory version: linearity is a working approximation;
   model/selection dependence remains outside the quoted errors. Further
   validation is required before precision calibration claims.
   N2-4: reference-covariance propagation and four-weighting audit complete;
   see `review_2026-10-22_n2_4/`. A separate table reports expanded conditional
   intervals without changing the original curves/bands. Acceptance of this
   statistical scope and separate weighting-sensitivity reporting was approved
   by the author, temporarily closing N2-4 for this version. Original plot bands
   remain slope-only statistical intervals, not total calibration errors.
   Both decisions can be revisited in the next page-by-page review.
5. N1-5 is closed for this version: same-event timing validation with conditional
   statistical intervals. Intrinsic bar resolution and synchronization-jitter
   extraction are outside scope. N2-5 is closed for documentation-only scope:
   all four Note2 comparison captions and lower axes explicitly identify a
   model difference, not event timing resolution; no numerical changes.
6. N2-6 is closed for exploratory reporting: retain all 21 completed fits and
   the unchanged, interactively selected ten-channel illustration. The new
   status table documents missing/narrow overlap and poorly constrained cases;
   no new quality cuts or population-level performance claim are introduced.
   See `review_2026-10-22_n2_6/` for the inventory and limitations.
7. N2-7 is closed for exploratory backup scope. Both predictor slides remain
   after the main conclusion, explicitly MCP-trained and not a demonstration
   of reference-free calibration. Independent validation and full predictive
   uncertainties remain future work; figures and numerical results are unchanged.
8. N1-6: local references and candidate packaging completed; official template,
   DP metadata and cover/submission handling await convener confirmation.

Before publication, synchronize the relevant local code and documentation with
the intended online repository. Verify that every code path cited in the slides
exists there and matches the version used for the reported results. Record the
repository URL and immutable commit or release identifier in the final notes.
Local figure/source snapshots alone do not establish that the online repository
is up to date. The author has now authorized committing and pushing the note
updates after each accepted review point to
`De-Cristo/LecroyTranslator4CMSTimingDetector`, using the local checkout
`/afs/cern.ch/work/l/lichengz/private/MTD/TB_2025/LecroyTranslator4CMSTimingDetector`.
See `REPOSITORY_WORKFLOW.md` for the standing synchronization workflow.

Review IDs appear beside the affected slides and in each closing review queue.
The longer evidence and writing plan remain in `review_2026-10-20/`.

Resolved: N1-1 was validated by the author. Its questions and provenance caveats
have been removed from Note1 slides 4 and 5 and its open review queue; the
underlying results are unchanged.

N1-2 comments were replaced with relevant code paths on Note1 slides 6 and 10.
The code locations are verified; this change does not independently recover
the historical runtime configuration. N1-2 is no longer in the slide review queue.

N1-3 was closed on 21 October 2026 after checking the implemented reference-time
subtraction and LOWESS path. Slides 7-9 no longer carry its review comments;
slide 9 shows the explicit minus sign and code references. No physics results
or analysis code were changed.

N1-4 investigation: `review_2026-10-21_reference_comparison/` compares the four
saved reference shapes and controlled binning/weighting changes. The shape
differences are 1.65-4.88 ps RMS after offset alignment on joint support.
The legacy profiles reconstruct from the raw-channel-192 `walk_before`
samples; the newer reference uses the expected trigger time. At the author's
request, N1-4 was closed on 21 October 2026 by adding the clock-corrected method,
four profiles and a comparison summary to Note1 (slides 15-20). The archived
curves remain explicitly labelled. Neither fit version nor the Note2 analysis
was changed. Note2's reference-version caveat is removed; its cross-run transfer
review remains open. The comparison directory preserves the investigation
record preceding this decision.

N1-5 update, 22 October 2026: `review_2026-10-22_bar8_validation/` preserves
the initial investigation, including missed resets and likely association
offsets. `review_2026-10-22_clock_quality/` contains the subsequent production
validation. Missing timestamps no longer hide resets; a clock-only 1 ns
scaled-MAD group threshold rejects gross failures without shifting MCP rows.
All 22 reference functions and their overlap-only AWC comparisons are refreshed
in `Q_Walk/results/2026-10-22_clock_quality/`. Both notes use that version.
Historical outputs are unchanged. The author subsequently closed N1-5 for this
version under the narrower timing-validation claim: the same-event bar--MCP
core width improves from 77.69 to 64.16 ps, with explicitly conditional
statistical intervals. Its slide comment and review-queue entry are removed.
Intrinsic-resolution and synchronization-jitter extraction are outside scope;
the existing position, covariance, segment-offset and calibration-systematic
limitations remain explicit. This closure changes no numerical results.

N1-6 local update: each note has separate unapproved
publication-candidate PDFs, with and without the optional local A4 cover.
The Note1 candidate excludes the four unvalidated historical timing/clock
backups, including the clock-estimator comparison; full internal review decks
retain them. Subsequent author-approved Note2 closures replace review comments
with permanent scientific qualifications. The packaging itself changes no fits,
plots or scientific review decisions. See `PUBLICATION_CHECKLIST.md`
and `review_2026-10-22_publication/citation_audit.md`. N1-6 is not fully closed:
the conveners must still confirm official format, metadata and cover handling.

## Edit and rebuild

Edit each note's `main.tex` and `metadata.tex`. Shared layout and cover source:
`draft_support/slides_preamble.tex` and `draft_support/cover.tex`.
The build copies these shared files into each standalone note directory.
The cover avoids the unavailable `everysel` dependency using ordinary LaTeX
paragraph justification; the supplied template remains untouched.

From the repository root:

```bash
python3 DPS_NOTES/draft_support/build.py --assets
```

Omit `--assets` when only slide text or layout changed. Required tools:
Python with NumPy, pandas, SciPy, Matplotlib, Pillow and pypdf; pdfLaTeX; Poppler `pdftoppm`.
The figure builder reads existing analysis outputs; it does not open ROOT files,
refit physics models, change selections or modify `Q_Walk`.

The build checks title/abstract placement, one-page covers, figure labels and
LaTeX overflow. Rendered previews still need visual inspection after layout
changes. `figure_manifest.json` records source paths, copies and SHA-256 hashes;
numerical sources are retained under `data/`.

Add `--candidates` to build the separately numbered publication-candidate
variants. Each note owns its `references.tex`; both builds reject unresolved
LaTeX citations. The combined review PDF is not a
submission package. Candidate PDF hashes and pending approval status are in
each note's `candidate_report.json`.

The draft builder produces only files under `DPS_NOTES/`; repository
synchronization and the authorized commit/push are a separate step after each
accepted review point. This is not authorization to submit an official CMS note.
Source archives can compile on their own using
`pdflatex main.tex` twice and `pdflatex cover.tex`.

## Original abstracts and bibliographies restored (22 October 2026)

At the author's request, the A4 cover abstracts now reproduce the supplied
`Note1.pdf` and `Note2.pdf` abstracts. `DPCoverAbstract` is separate from
`DPAbstract`; the slide abstracts remain unchanged. The shortened titles and
all previously agreed scientific qualifications in the slides are retained.
The original abstract claims are restored verbatim, not newly validated here.

The bibliographies now reproduce the originals' entries and order: 8 for Note1
and 12 for Note2, each split across two slides. Typography and line wrapping
are adapted to the slide format. Added bibliography entries absent from the
originals were removed, with dependent inline citations adjusted. Original
GitHub repository citations are retained alongside the direct pinned script
links used elsewhere. No data, fitted values or plotted results changed.

The author also removed Note1's "Review queue before publication" slide
(27/32 before removal). Pending template, DP metadata and CMS approval matters
remain tracked in the publication checklist; removing the slide does not close them.
The "Backup: residual segment offsets" slide (28/32 in the same version)
was also removed from both Note1 variants. Its underlying saved data are retained.
The "Backup: historical end-difference illustration" slide (29/32) was removed
from the review deck as well; it was already excluded from the candidate.
The "Backup: historical bar--MCP illustration" slide (30/32) was likewise
removed from the review deck. The validated main MCP result is retained.
