# Short slide drafts: version 2, draft 1

Note 1 has ten main slides, one reference slide and one diagnostic backup slide. Note 2 has eight main
slides, one reference slide and one exploratory backup slide,
with no A4 cover pages. These are separate from the full
notes, which remain unchanged. Each deck retains the CMS/contact title slide,
an abstract on Note 1 slide 2 and a purpose slide on Note 2 slide 2, labelled plots,
and concise procedural captions.
The titles follow the author's wording, including TOFHIR2C in Note 2.

## Note 1 Backup Placement (23 October)

The main presentation ends on the calibration-route closure (slide 10).
References are slide 11. The left-minus-right diagnostic is now backup slide 12,
marked internal discussion only, without a CMS Preliminary plot label. Its
points, widths and assumptions are unchanged. The original labelled analysis
figure remains archived. `note1_transfer_review.pdf` contains slides 9, 10 and 12.
The private meeting checklist has been reconciled locally and is not uploaded.

- `note1_mcp_reference/slides.pdf`: synchronize MCP/FEBd records, construct
  the reference time, fit inverse-charge profiles, and check timing correction.
- `note2_dual_threshold/slides.pdf`: motivate rapid calibration, measure the
  internal charge-to-slope mapping, benchmark the single-amplitude walk form,
  ending on corrected timing residuals. One post-reference backup slide
  retains the empirical AWC trend with inverse slope coefficient.
- `both_notes_slides_v2.pdf`: both decks, 22 landscape pages, no covers.
- `note1_transfer_review.pdf`: the three new Note 1 closure slides only.

Each reference slide uses two columns. Note 1 has seven entries after removing
the Saclay source at the author's request and renumbering; the remaining entry
text is unchanged. Note 2 retains its original twelve entries and numbering.
The build checks these against the full notes and requires every remaining
Note 1 reference to be cited in the slide body or its caption.

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

Requires pdfLaTeX with TikZ, Poppler, Pillow, pypdf, Matplotlib, NumPy, pandas
and SciPy; the reused clock-analysis module also requires uproot and awkward.
Previews are generated locally
under each note's `preview/` directory. The builder rejects layout overflow,
missing fonts, unresolved citations, unexpected page counts and portrait pages.
These are first discussion drafts, not approved CMS submissions.

## Note 1 structure review

### Body citations and restricted-source removal

The Saclay presentation (previous reference 2) is removed from the short Note 1
deck only. All seven remaining entries have explicit citations at the relevant
discussion: detector design (slide 3), software (4), LOWESS (5), MCP timing,
waveform sampling and amplitude walk (6), and the slope-based alternative (8).
The source-to-claim audit is recorded in `note1_mcp_reference/citation_audit.json`.
The original full-note bibliography and Note 2 are unchanged.

### 23 October: reference-bar transfer closure replaces old slide 9

Slides 9, 10 and backup 12 show the MCP -> reference -> module-under-test calibration
path, direct-versus-transfer closure, and left-minus-right timing checks.
The references are on slide 11. Only T1=3, T2=60 DAC is displayed, chosen
to match slide 8 rather than by timing performance. The other five settings
and their historical figures are retained but not shown on these slides.

The separate `DPS_NOTES/review_2026-10-23_reference_transfer/` analysis refits
both downstream paths on identical input events with the frozen upstream
calibration. It saves the common-event sample, fitted profiles, curves, timing
distributions and 400 paired block-bootstrap replicas. The slide build only
copies these saved closure figures and numerical macros; it does not rerun
this analysis. Results support in-sample closure for the tested bar, not an
independent validation for all modules or a single-bar resolution measurement.

Slide 10 now shows only DUT minus reference, enlarged, with legend labels
identifying the calibration of each bar. The auxiliary DUT-minus-MCP comparison
is retained in the saved analysis but is not presented in the deck. No numerical
results or uncertainty definitions were changed for this presentation update.

Backup slide 12 uses the bar-average convention explicitly: histograms show
(tL-tR)/2, with widths 27.0 ps for the calibrated reference and 44.3 ps for
the downstream bar calibrated via reference. Their quadrature is 51.9 ps.
The independent-end and negligible-position-spread assumptions remain visible;
the original unscaled analysis outputs are preserved.

### 23 October: consistent LOWESS drift panels on slide 5

`prepare_note1_drift.py` redraws three panels from the same frozen matched-event
snapshot as slide 4, with the same subset colors, common ns residual limits,
and elapsed scope time in seconds. Each subset has a centered least-squares
clock line; the residuals are smoothed using the existing
`TimeCalibration/ch192_vs_trigger_lowess.py::_lowess_smooth` implementation
(fraction 0.15, three robust iterations). All 7,206 points are retained. This
illustrative redraw does not rerun or replace detector calibration results.
The curves, residuals and settings are saved in `assets/note1_drift/`.
Slide 5 briefly defines LOWESS, cites reference [8], and links directly to
Cleveland's 1979 JASA paper, DOI `10.1080/01621459.1979.10481038`.
The original bibliography entries and the full notes remain unchanged.

### 23 October: restore interval ratio minus one on slide 4

The author subsequently requested retaining the original observable to distinguish
slide 4 from the time residuals on slide 5. The current figure therefore shows
`(delta_t_FEBd / delta_t_scope - 1) * 1e6`, a dimensionless quantity, not ns.
`prepare_note1_matching_ratio.py` uses the saved timestamp snapshot to construct
7,203 consecutive intervals within the three subsets (never across boundaries).
No linear clock alignment or LOWESS is applied. All intervals appear in the
scatter; the histogram displays the central 98% to resolve its narrow core.
A Gaussian fit describes the central ratio spread, not intrinsic timing noise.
The caption defines segments/subsets and distinguishes the following drift study.
The new figure, interval table, histogram and fit report are in
`assets/note1_matching_ratio/`. The earlier ns artifacts remain archived below;
they are no longer regenerated by the slide build or used on slide 4.

### Superseded: matching residuals in ns

Slide 4 replaces the dimensionless interval-ratio figure with timing residuals
in ns, as explicitly selected by the author. `prepare_note1_matching.py` reads
the frozen timestamp snapshot in `assets/note1_matching_ns/`. It uses the same
three matched subsets as the old ratio figure: 2,405, 2,396 and 2,405 events
(one fewer consecutive interval per subset in the original ratio plot).
An independent linear fit aligns each subset's clocks; no LOWESS, topology,
charge cut, or residual clipping is applied to the displayed 7,206 events.

The 80-bin histogram retains all events. A Poisson-error-weighted Gaussian fit
uses bins within median +/- 1.5 scaled MAD; its effective core width is about
0.066 ns. This is not a pure intrinsic synchronization-noise measurement:
the pre-LOWESS residuals still contain slow drift. The caption defines a segment
as a dataset subset and links the tails to the following drift-correction slide.
A saved 201-event rolling-mean diagnostic supports a slow-structure contribution
to the tails; it is not applied to the displayed data and does not establish
that every tail event is caused by drift. Data, histogram counts, fit parameters,
and input provenance are saved alongside the figure. Detector calibration fits
and the other slides are unchanged.

### 23 October: simplified beam-test schematic

Slide 3 now uses the editable vector drawing `assets/note1_setup/setup.tex`.
Only the beamline, upstream reference module, downstream module under test, and
MCP are shown. The reference module is perpendicular to the beam; the module
under test is rotated by approximately 52 degrees from that orientation, as
specified by the author. At the author's request, the slide does not annotate
the angle: no numerical angle, arc, or dashed angular baseline is shown.
Distances and dimensions are not to scale.
The build compiles this TikZ source and records the resulting figure hash.
The caption no longer refers to the removed readout-path highlighting.
The previous setup assets remain archived but are not used on the slide.

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

# Slide 7 wording review (23 October 2026)

Note 1 describes the selected coincidence in physical terms: both ends of the
upstream reference bar and downstream bar under test, plus the dedicated
trigger channel. This is a wording-only change; the five-hit selection and
unique MCP matching in the analysis remain unchanged.

## Slide 8 time-difference labels (23 October 2026)

The two profile axes now identify left/right bar-end time minus the clock-aligned
MCP time, in ns. The caption defines the MCP clock mapping and retains the
existing +26 ns display-offset explanation. Labels only: plotted data, fits,
and numerical results are unchanged.
