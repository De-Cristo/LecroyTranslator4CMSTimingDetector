# N2-3 diagnostic study definition

This is an exploratory adequacy/sensitivity study of the frozen 22-channel
run4076 sample, not a new nominal calibration or an optimized channel selection.
The following choices were fixed before running this scan.

- Baseline: the frozen prompt selection, histogram-derived charge boundary,
  ten equal-width charge bins, SEM-weighted linear fit.
- Model alternative: quadratic on the same selected hits and ten bins.
  Use bin-averaged polynomial moments, including mean(Q^2), to avoid confusing
  finite-bin averaging with curvature. Linear predictions remain at mean Q.
- Binning alternatives: 8 and 12 equal-width bins on the baseline selected hits.
- Boundary alternatives: multiply the frozen charge threshold by 0.95 and 1.05,
  keeping all other cuts fixed. These are sensitivity probes, not calibrated
  uncertainties on the threshold.
- Rise-time alternatives: require delta-t >= 100 ps, first with the frozen
  charge threshold, then with the histogram peak boundary recomputed. These
  separate timing-selection effects from an associated charge-boundary change.
- No additional slope clipping, range tuning or channel removal based on fit
  quality is allowed. All failures are reported.

Fit diagnostics use profile residuals with SEM bars, chi-square/ndf, and the
linear-to-quadratic reduction in chi-square. Their error model conditions on
the reconstructed times/calibration and treats profile means as independent.
They are diagnostics, not independently calibrated hypothesis-test probabilities.

For AWC propagation, reuse the saved run4399 reference and uniform-Q AWC-plus-
offset least-squares objective. First reproduce the original 21 completed AWC
fits on their saved grids. Channel 238 is retained as a no-reference-overlap
case. Then restrict each channel's saved grid to the intersection of measured
support of all successfully fitted alternatives. Refit the baseline and every
alternative on exactly these points. Never extrapolate to recover support.
Report retained-grid fraction and the baseline change from narrowing support.

Only positive finite slope functions on this common support can be inverted.
Non-positive, flat-inverse-slope or missing-support cases are failures, not
discarded quietly. Negative AWC values remain visible, with no positivity clamp.

Propagate the saved baseline slope-bootstrap coefficients to the common grid
to retain the original conditional statistical scale. Alternative AWC shifts
and their maximum absolute envelope are reported separately; they are not
independent statistical significances or a new 68%/95% uncertainty. No alternative
is selected merely because it agrees best with the MCP reference. Final model
choice, reference uncertainty and channel eligibility remain explicit decisions.
