# Short slide drafts: version 2, draft 1

Two notes: Note 1 has nine content slides and Note 2 has eleven, each followed by one compact reference slide,
with no A4 cover pages. These are separate from the full
notes, which remain unchanged. Each deck retains the CMS/contact title slide,
a purpose slide on slide 2, labelled plots,
and concise procedural captions.
The titles follow the author's wording, including TOFHIR2C in Note 2.

- `note1_mcp_reference/slides.pdf`: synchronize MCP/FEBd records, construct
  the reference time, fit inverse-charge profiles, and check timing correction.
- `note2_dual_threshold/slides.pdf`: motivate rapid calibration, measure the
  internal charge-to-slope mapping, benchmark the single-amplitude walk form,
  and explore AWC prediction from internal observables. One backup slide
  combines the physical predictor and its excluded-bar test.
- `both_notes_slides_v2.pdf`: both decks, 22 landscape pages, no covers.

Each final reference slide uses two columns and preserves the original
bibliography entries and numbering: eight entries in Note 1 and twelve in Note 2.
The build checks that these entries match the full notes exactly.

Existing figures and numerical macros are copied unchanged from the full notes.
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

Requires pdfLaTeX, Poppler, Pillow and pypdf. Previews are generated locally
under each note's `preview/` directory. The builder rejects layout overflow,
missing fonts, unresolved citations, unexpected page counts and portrait pages.
These are first discussion drafts, not approved CMS submissions.

## Note 1 structure review

The author's six-point structure comment is implemented in the short deck:
purpose; parallel MCP/scope stream; FEBd synchronization; correlated events
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
Slide 9 explains
the proposed internal-data procedure and the independent tests still needed.

Slide 10 ends the main narrative with the original note's final observable,
AWC versus inverse slope-to-charge coefficient (Figure 6), using the currently
saved ten-channel figure. It distinguishes an informative trend from a
validated precision predictor. The original Figure 6 caption incorrectly
describes the points as charge-bin means: here they are explicitly channels,
with slope-only conditional AWC bootstrap intervals.

Slide 11 is the only backup. It combines the median-charge candidate
`A = K S(Q*)^2 / (c1 Q*)` with the saved leave-one-bar-out comparison:
3.22 uA RMS AWC error, 14.1% RMS fractional error, ten channels and six bars.
These are the same frozen clock-quality results used in the full note's
backup, not new fits. Learning K uses reference-calibrated training bars;
prediction inputs for an excluded bar are internal. Transfer to other
configurations and full predictive uncertainty still need validation.
Slide 12 retains the original compact bibliography.

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
