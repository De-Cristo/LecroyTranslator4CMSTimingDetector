# Dual-threshold time-walk calibration: review draft v1

Open `slides.pdf` or `note_with_cover.pdf`. Edit `main.tex` and `metadata.tex`.
`figure_manifest.json` identifies the numerical input to every figure;
`data/` retains the source tables, coefficients and bootstrap replicas.

This draft uses the measured-overlap comparison refreshed on 22 October 2026.
It explicitly distinguishes reference-fitted AWC, model-shape closure and
exploratory internal prediction. The 4076 and 4399 samples are different events.
See review items N2-1 through N2-7 before any public release.

On 21 October 2026 the clock-corrected reference profiles were also added to
Note1, closing the reference-version gap N1-4. The N2-1 caveat now concerns
cross-run operating conditions and calibration transfer only. No Note2 fit,
figure data or AWC result changed.

On 22 October 2026 the run4399 references were refitted after finite-trigger
segmentation and clock-only group rejection. All AWC comparisons, summary
points, the exploratory predictor and the run4399 charge spectrum were
regenerated consistently. The run4076 slope fits and bootstrap replicas are
unchanged. Current inputs are `Q_Walk/results/2026-10-22_clock_quality/`; the
old September results are preserved. `reference_update_awc_impact.csv` records
the per-channel change. The quoted intervals still hold the reference fixed.
No cross-run or reference-fit systematic has been added to those intervals.

The N1-6 packaging update also adds verified context/method references here.
`publication_candidate_slides.pdf` and `publication_candidate_with_cover.pdf`
provide independent candidate deliverables for this note. They remain explicitly
unapproved and retain the scientific review comments; only the closing queue is
omitted. No N2 review item is closed by this formatting change. See
`../PUBLICATION_CHECKLIST.md` for pending convener decisions and cover handling.

The standalone source archive compiles with pdfLaTeX. Within the repository,
use the shared builder described in `../README_DRAFTS.md`. No CMS DP identifier
has been assigned in this draft.
