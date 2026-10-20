# Writing plan for two CMS detector-performance notes

Status: proposed, following template/PDF/source inspection on 2026-10-20.
See `gap_analysis.md` for evidence and the full figure inventory. This is a
writing and figure-reproduction plan, not authorization to submit a note.

## Intended deliverables

Two independent, reviewable slide notes, each with:

- Slide 1: scientific title, author `CMS`, contact `cms-dpg-conveners-mtd@cern.ch`.
- Slide 2: abstract identical to the publication-cover abstract.
- A short sequence of results with detailed, standalone captions.
- Physics figures carrying `CMS Preliminary Phase 2` in the exported PDF/PNG.
- A shared metadata file for that note's slides and A4 cover.
- Reproduction inputs, commands, numerical tables and a plot manifest.
- Source archive suitable for Overleaf and a compiled local review PDF.

Keep the existing long-form notes as supporting material. An A4 publication
cover is a separate product, not an automatic new long-form manuscript.
Record any corresponding corrections needed in the existing long-form notes.

Proposed working titles, subject to the authors' final scope:

1. **MCP-referenced timing calibration of CMS BTL prototype bars in the 2025 beam test**.
2. **Dual-threshold time-walk calibration for CMS BTL prototypes**.

The second title avoids claiming that the present reference AWC extraction
uses no MCP. Its abstract can distinguish externally benchmarked development
from the intended reference-free deployment.

## Milestone 1: freeze the scientific content

Establish one per-note run/configuration table with beam conditions, detector
operating points, thresholds, channel/bar mapping, event selection and input
files. Link every plotted number to a saved table or a reproduction command.

For Note1 choose whether the released reference is the legacy four-channel
calibration or the updated weighted version. Preserve the old result if that
is the intended public reference; describe its actual fit procedure. If the
updated version is selected, regenerate its plots and coefficient table and
use the same version throughout Note2. Do not transplant old coefficients
into a newer selection or clipping prescription.

For Note2 use the corrected separation of roles: run4076 for S(Q), run4399
for the MCP walk curve, then reference-AWC extraction and internal prediction.
Freeze the overlap definition, free-offset alignment and channel eligibility
criteria. Treat previously hand-selected 10/11-channel summaries as exploratory
until a documented quality policy supports the chosen sample.

Acceptance: a source/selection/units table with no unresolved contradictions
for any result included in the main slides. Unsupported results move to
explicitly labelled work in progress or are omitted, rather than guessed.

## Milestone 2: reproduce the main figures

Use publication-specific plotting scripts under `DPS_NOTES/` that read frozen
analysis outputs. Avoid changing original analysis selections merely to alter
plot style. Save PDF, PNG and the numerical plot data together. For each figure,
record source files, analysis version, exact selections, fit function, weighting,
uncertainty meaning, display offsets and whether it is a training or held-out
result. Keep bootstrap bands at their true scale.

Priority order:

1. Note1 reference walk profiles and coefficient table; synchronize versions
   with Note2.
2. Note2 slope profiles, Q spectra and supported-overlap curve comparisons.
3. Note2 AWC summary and precisely defined residual plot.
4. Note1 scope waveform/reference resolution and final bar timing validation.
5. Note1 compact matching/clock correction diagnostics and shared setup diagram.
6. Optional clock/baseline backups and optional internal AWC predictor.

Acceptance: every included plot can be regenerated from its documented inputs,
contains the required label, and remains legible at its final slide size.
Histogram widths are labelled as fitted sigma or sample RMS as appropriate.
No uncertain photo-extracted points are presented as original measurements.

## Milestone 3: write Note1 slides

Suggested core sequence: 11-12 slides, with detailed diagnostics in backup.

| Slide | Purpose and proposed content |
| --- | --- |
| 1 | Title, CMS, required contact. |
| 2 | Abstract: reference-detector role, synchronization problem, adopted calibration and only verified performance numbers. |
| 3 | Setup and sample: two BTL modules, downstream MCP, independent DAQs, restricted MCP acceptance; explain why reference bars are calibrated. |
| 4 | MCP time estimator: waveform template fit and scope-only timing comparison. Distinguish intrinsic MCP estimate from end-to-end reference. |
| 5 | Coincidence matching: channel192 indicator and unique matched events; show validation and retained counts. |
| 6 | Adopted clock alignment: linear clock relation and residual drift, followed by LOWESS correction. Define all signs and time units. |
| 7 | Calibration sample and variable definitions: reference channel pairs, numeric cuts, Q [ADC], t_SiPM - t_MCP, profile/fit prescription. Include a spectrum if useful. |
| 8 | Bar7 reference calibration: ch133 and ch154 profiles with quadratic functions. Caption explains points, SEM, fit weights and valid support. |
| 9 | Bar8 reference calibration: ch137 and ch150 in the same convention. Coefficients provided as a table or linked numeric supplement. |
| 10 | Timing validation: corrected end difference and bar-MCP difference, with event selection, fit uncertainties and covariance/position assumptions. |
| 11 | Result and applicability: what the reference calibration enables, support range, residual synchronization limitation and transfer conditions. |
| 12, if needed | Short public references and reproducibility information; alternatively distribute concise citations on the relevant slides. |

Backup candidates: clock-waveform estimators, baseline study, three-system
resolution algebra, matching implementation details and full coefficient table.
If both synchronization approaches are retained as headline results, give each
an explicit method/results pair rather than mixing them in one caption.

## Milestone 4: write Note2 slides

Suggested core sequence: 11-13 slides depending on inclusion of prediction.

| Slide | Purpose and proposed content |
| --- | --- |
| 1 | Title, CMS, required contact. |
| 2 | Abstract: dual-threshold mapping, cross-run MCP benchmark and demonstrated result; separate future reference-free deployment. |
| 3 | Physics idea: two threshold crossings, finite-interval S = Delta I/(t2-t1), correction AWC/S(Q), one AWC per channel. |
| 4 | Samples and roles: run4076 versus run4399, different events, operating conditions and transfer assumptions; reference Note1 for MCP details. |
| 5 | Q spectra and selection: peak-right cut and representative channels, Q as ADC proxy. Avoid calling all retained events a pure MIP sample. |
| 6 | S(Q) calibration for bar7: profile means, explicitly labelled SD/SEM and true-scale fit uncertainty. |
| 7 | S(Q) calibration for bar8, same definitions. State linear-model adequacy quantitatively. |
| 8 | Extract reference AWC: weighted/unweighted objective as actually used, free offset, shared measured Q support, and no event matching across runs. |
| 9 | Curve comparison and residual: selected representative channels; shared-range shading, clear gap/extrapolation style, and actual residual definition. Split if captions cannot remain readable. |
| 10 | Channel summary: reference AWC versus 1/c1, inclusion policy, correct errors and fit adequacy. This is a correlation diagnostic, not proof of a universal rule. |
| 11 | Optional internal predictor: K S(Q*)^2/(c1 Q*) hypothesis, fixed Q* definition, whole-bar held-out prediction and baseline comparison. Include only after the validation gate below. |
| 12 | Conclusions and limits: what is established about inverse-slope shape, what remains in predicting AWC, and the separately required channel offsets. |
| 13, if needed | Public references and definitions, or move these into slide captions and backup. |

Do not use a second detector module to derive AWC as a replacement for the
single-layer method. The role of MCP here is development/benchmarking; the
proposed deployed predictor takes only single-layer internal observables.

## Optional predictor validation gate

Keep the recent physical candidate out of the main abstract until:

- A channel eligibility rule is defined without selecting on the desired trend.
- The Q* convention and one headline metric are fixed; mean, median and peak
  alternatives are sensitivity studies, not opportunities to choose the best score.
- Reference-fit and slope-fit uncertainties, including their induced feature/
  target correlations, are propagated or explicitly bounded.
- Predictions are evaluated with whole bars held out, with model-selection
  effects disclosed; an independent run/configuration is preferable if available.
- The uncertainty claim distinguishes statistical fit uncertainty, prediction
  scatter, charge-definition sensitivity and cross-run transfer effects.
- Correction-curve RMS is not described as measured detector timing resolution.

The present exploratory results can be shown in backup with those limitations.
Do not promote them to a universal 10% AWC calibration claim.

## Caption standard

For each plot write one standalone paragraph answering: what is measured and
why; which sample and operating conditions; which event/channel selection;
what the axes and units mean; what each marker/line/band represents; what
uncertainties are included; and what conclusion the plot actually supports.
Define BTL, SiPM, MCP, Q, LOWESS and AWC when first used, and avoid unexplained
internal run numbers as the only sample description.

For a slope profile, explicitly separate event-to-event SD from SEM used in
fitting and from the bootstrap band. For an AWC summary, state that each point
is one channel and identify the bootstrap interval and fixed-reference
approximation. For a residual plot, distinguish profile residuals from
individual timestamp distributions. Use uncertainty values and units directly
beside quoted resolutions.

## Milestone 5: package and review

Proposed layout:

```text
DPS_NOTES/
  cms_dp_note_template/          # supplied scaffold retained
  review_2026-10-20/             # gap analysis and this plan
  note1_mcp_reference/
    main.tex                    # slides
    metadata.tex                # title/abstract/contact/assigned number
    cover.tex
    figures/
    captions.tex
    plot_manifest.csv
    scripts/
    tables/
    build/
  note2_dual_threshold/
    ...                         # same layout, independent metadata
```

Resolve the local `everysel.sty` dependency or use the established Overleaf
toolchain, then compile both decks and covers. Inspect every rendered slide
for readable labels, captions, clipping, overlap, undefined references and
placeholders. Check that slide2 and the cover use identical abstract text.
Confirm the publication workflow before concatenating an A4 cover, since the
submission service may generate the first page itself. Use assigned DP numbers
and release dates only when supplied; do not invent them.

Acceptance: two independently buildable source bundles, two reviewed slide PDFs,
two cover previews, labelled standalone figures, traceable numbers and a short
remaining-issues list. External submission is a subsequent explicit action.

## Decisions to settle during writing

1. Which reference-calibration version is the public baseline for both notes?
   Recommended: one frozen version with fully documented selections and fits;
   do not mix the legacy four-channel coefficients with the extended pipeline.
2. Is Note2 limited to inverse-slope validation, or does it include internal
   AWC prediction as a headline result? Recommended first draft: validated
   shape comparison in the main story; exploratory predictor in backup.
3. Is the alternate clock-waveform synchronization branch central to Note1?
   Recommended: keep trigger/LOWESS alignment as the main calibration path,
   with the alternate clock/baseline diagnostics in backup unless their
   independent performance is a primary intended result.

These decisions affect emphasis, not the ability to begin source tracing,
caption drafting and reproducible figure preparation.
