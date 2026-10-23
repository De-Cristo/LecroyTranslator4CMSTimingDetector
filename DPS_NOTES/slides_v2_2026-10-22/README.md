# Short slide drafts: version 2, draft 1

Note 1 has nine content slides and one reference slide. Note 2 has eight main
slides, one reference slide and one exploratory backup slide,
with no A4 cover pages. These are separate from the full
notes, which remain unchanged. Each deck retains the CMS/contact title slide,
an abstract on Note 1 slide 2 and a purpose slide on Note 2 slide 2, labelled plots,
and concise procedural captions.
The titles follow the author's wording, including TOFHIR2C in Note 2.

- `note1_mcp_reference/slides.pdf`: synchronize MCP/FEBd records, construct
  the reference time, fit inverse-charge profiles, and check timing correction.
- `note2_dual_threshold/slides.pdf`: motivate rapid calibration, measure the
  internal charge-to-slope mapping, benchmark the single-amplitude walk form,
  ending on corrected timing residuals. One post-reference backup slide
  retains the empirical AWC trend with inverse slope coefficient.
- `both_notes_slides_v2.pdf`: both decks, 20 landscape pages, no covers.

Each reference slide uses two columns and preserves the original
bibliography entries and numbering: eight entries in Note 1 and twelve in Note 2.
The build checks that these entries match the full notes exactly.

Existing numerical macros are copied unchanged from the full notes. Figures
retain their data and fits; the Note 1 terminology-only adaptations are described below.
The new threshold-separated figures are copied from the separate, saved
`Q_Walk/results/2026-10-22_bar_pair_by_threshold/` validation, with its inputs and
uncertainties documented there. The build does not rerun that analysis.
`asset_manifest.json` records each source and hash. No ROOT inputs are read
and no fits are rerun. The full notes retain the complete bibliographies,
selection details and uncertainty studies. No claim of intrinsic detector
resolution or demonstrated reference-free AWC calibration is added.

Rebuild from the workspace or repository root:

```bash
python3 DPS_NOTES/slides_v2_2026-10-22/build.py
```

Requires pdfLaTeX, Poppler, Pillow, pypdf and Matplotlib. Previews are generated locally
under each note's `preview/` directory. The builder rejects layout overflow,
missing fonts, unresolved citations, unexpected page counts and portrait pages.
These are first discussion drafts, not approved CMS submissions.

## Note 1 structure review

### 23 October: abstract and audience-facing terminology

Note 1 slide 2 is now an abstract, adapted from the earlier abstract and the
purpose-page logic. The full-note cover abstract is unchanged. Public slide text
uses reference module, module under test, and left/right channels; bar 8 and bar 3
remain secondary identifiers. Internal mapping: reference left/right = 137/150;
module-under-test left/right = 234/243, verified against
`TimeCalibration/channel_mapping.py`.

`prepare_note1_labels.py` changes only figure labels. It replaces selected PDF
text objects in the profile and histogram figures and overlays label-only regions
of the retained setup and matching rasters. No data, fits, selections, or full-note
figures change. Source hashes and changes are recorded in
`assets/note1_public_labels/provenance.json`. The schematic's two module labels
now distinguish the reference module and module under test; further schematic
simplification remains for the page-by-page review.

The author's six-point structure comment is implemented in the short deck:
abstract with the calibration purpose; parallel MCP/scope stream; FEBd synchronization; correlated events
with low-frequency clock drift; corrected upstream reference bars; final
downstream-minus-reference-bar timing comparison. The setup slide explicitly
distinguishes this test-beam synchronization from the final HL-LHC scheme.
One combined, six-panel threshold-separated bar-to-bar histogram figure
ends the narrative on slide 9, immediately before references on slide 10.
The earlier MCP-check slide is absent; the separate width-summary slide is
removed to keep one final result slide. Its saved analysis artifacts are retained.

The former 8,737-event, 141.8 ps mixed-threshold result is superseded. Each of
six thresholds now has independent four-channel fits and same-event validation;
8,672 selected events are distributed across the six settings. Corrected widths
range from 62.1 to 69.4 ps. No combined event histogram or per-setting alignment
is used. The optional event-weighted RMS of fitted widths (65.7 ps after
correction) is saved as a descriptive summary, not an operating-point resolution.
Slide 8 now uses the lowest-threshold bar-8 profiles rather than mixed-threshold
bar-7/bar-8 examples; slide 9 shows the six timing comparisons with a compact
width-range summary. In-sample and single-bar-resolution limitations remain
explicit; bootstrap intervals are retained in the saved analysis outputs.
The final caption relates the approximately 65 ps pair width to the quadrature
sum of the two bar widths, explicitly assuming independent timing fluctuations.
It does not assign individual bar resolutions or assume equal performance.
The 65 ps notation is an approximate scale for the 62--69 ps per-setting results,
not an additional pooled fit. Correlations would add a covariance term and
invalidate the simple quadrature interpretation.
The full original notes are unchanged in content.

## Note 2 structure review

The short deck follows the authors' eight-point narrative. The goal is a
repeatable internal-data calibration, not a catalogue of MCP-fitted amplitudes.
Slides 2--6 cover the HL-LHC calibration need, variation of the original
quadratic coefficients, the dual-threshold physical model, internal slope
profiles and the one-amplitude inverse-slope correction. The internal slope-profile
example on slide 5 is channel 150 (19,312 selected hits). Slide 7 benchmarks
the functional form with the existing MCP comparison curves. Slide 8 reproduces
original Note 2 Figure 5, channel 137 residuals after the AWC correction.
The main talk ends there. Slide 9 contains the compact original bibliography.
The former separate future-procedure slide is removed; its independent-test
requirements remain in the backup discussion.

Slide 10 is internal backup on the original note's final observable,
AWC versus inverse slope-to-charge coefficient (Figure 6), using the currently
saved ten-channel figure. It distinguishes an informative trend from a
validated precision predictor. The original Figure 6 caption incorrectly
describes the points as charge-bin means: here they are explicitly channels,
with slope-only conditional AWC bootstrap intervals.

The new shared-response model and its prediction plot (former slide 11) are
deferred and removed from the current deck. Its discussion document and saved
analysis artifacts are retained for future consideration; no data are deleted.
The remaining backup is labelled internal discussion only, not for public presentation.
Its plot drawing operations and numerical results are unchanged; only the
CMS Preliminary text object is removed from short-deck figure copies by
`prepare_note2_backups.py`. The full-note figures are untouched. The backup
headline also omits the CMS logo. The build checks reference placement and
the absence of Preliminary labels on the remaining backup page.

`AWC_PREDICTOR_DISCUSSION.md` separates the derivative identity from the
empirical shared-response hypothesis and provides a suggested coauthor reply.

The residual figure is the unaltered embedded JPEG from PDF page 12 (printed
page 11) of the supplied `DPS_NOTES/Note2.pdf`, not a replacement refit.
`assets/note2_figure5/provenance.json` records source and image hashes.
`prepare_note2_figure5.py` extracts the image and adds the CMS label outside it;
it can also rebuild from the saved JPEG when the supplied PDF is unavailable.
Its original points, error bars and residual linear fit are preserved. The
supplied caption does not specify the statistical definition of these error
bars, so no SD/SEM or uncertainty claim is added. It is a residual-versus-charge
illustration, not a single-hit resolution or a new reference-free validation.

The original four-channel coefficient table is retained as historical context
from Table 1 of the supplied detector note and `Q_Walk/empirical_walk_models.md`;
it is not presented as the input to the current comparison curves. The latter
retain their existing provenance. Note 1, cover abstracts, full notes and all
numerical analysis results are unchanged by this narrative update.
