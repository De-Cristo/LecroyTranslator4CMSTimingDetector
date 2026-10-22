# Dual-threshold time-walk calibration: review draft v1

Open `slides.pdf` or `note_with_cover.pdf`. Edit `main.tex` and `metadata.tex`.
`figure_manifest.json` identifies the numerical input to every figure;
`data/` retains the source tables, coefficients and bootstrap replicas.

This draft uses the measured-overlap comparison refreshed on 22 October 2026.
It explicitly distinguishes reference-fitted AWC, model-shape closure and
exploratory internal prediction. The 4076 and 4399 samples are different events.
N2-1 is closed under the nominal-setup assumption below. Review items N2-2
through N2-7 remain open before any public release.

On 21 October 2026 the clock-corrected reference profiles were also added to
Note1, closing the reference-version gap N1-4. This left the N2-1 question of
cross-run operating conditions and calibration transfer. No Note2 fit,
figure data or AWC result changed.

On 22 October 2026 the run4399 references were refitted after finite-trigger
segmentation and clock-only group rejection. All AWC comparisons, summary
points, the exploratory predictor and the run4399 charge spectrum were
regenerated consistently. The run4076 slope fits and bootstrap replicas are
unchanged. Current inputs are `Q_Walk/results/2026-10-22_clock_quality/`; the
old September results are preserved. `reference_update_awc_impact.csv` records
the per-channel change. The quoted intervals still hold the reference fixed.
No cross-run or reference-fit systematic has been added to those intervals.

N2-1 closed on 22 October 2026 under the author's stated working assumption:
the nominal detector setup is the same, with MCP reference timing in run 4399
and dual-threshold timing in run 4076. Run configurations or logbook records
were not supplied, so this is not an independently verified equality of gain,
bias, thresholds, temperature, firmware or charge calibration. The slides
state the assumption and retain the unquantified cross-run and
selection-dependent response limitations. No fit, data, figure or numerical
uncertainty changed. The two inline N2-1 comments and queue entry were removed;
the other N2 review items remain open.

The N1-6 packaging update also adds verified context/method references here.
`publication_candidate_slides.pdf` and `publication_candidate_with_cover.pdf`
provide independent candidate deliverables for this note. They remain explicitly
unapproved and retain the scientific review comments; only the closing queue is
omitted. No N2 review item is closed by this formatting change. See
`../PUBLICATION_CHECKLIST.md` for pending convener decisions and cover handling.

The standalone source archive compiles with pdfLaTeX. Within the repository,
use the shared builder described in `../README_DRAFTS.md`. No CMS DP identifier
has been assigned in this draft.
