# Reference-bar transfer closure

## Purpose

Test the chain MCP -> calibrated reference bar -> module under test, against
direct MCP calibration of the same module under test. The previous 64.7 ps
example independently calibrated all four bar ends against the MCP; it did
not demonstrate transfer through the reference bar.

## Procedure

1. Use only T1=3, T2=60 DAC (the lowest scanned setting, already used on slide 8).
2. Retain the frozen MCP-derived reference-bar corrections shown on slide 8.
3. Apply the existing four-channel quality and charge cuts and reference-fit
   support. Both downstream fitting paths use the same input events and 20
   uniform inverse-charge bin edges, with existing per-bin 5 scaled MAD clipping,
   at least five entries per bin, and SEM-weighted quadratic fits.
4. Fit each downstream end against either MCP or the corrected reference-bar
   average. The latter target is formed by subtracting the two MCP residuals;
   the event's MCP time cancels exactly. Only the frozen reference calibration
   needs the MCP; downstream transfer fitting does not use its event time.
5. Evaluate both paths on identical events in their common occupied-bin and
   fitted domains, with no extrapolation. Compare correction shapes, bar-pair
   widths, downstream-minus-MCP widths, and both left-minus-right widths.
6. Reproduce the old 1461-event result separately, distinguishing any changed
   sample from a changed calibration path.

Widths use the existing truncated-normal core fit in median +/- 2.5 scaled MAD.
400 paired file/segment bootstrap replicas keep cuts and calibrations fixed.
Their 16th--84th percentiles are conditional statistical intervals, not total
errors. This is an in-sample closure check with a shared reference calibration,
not an independent validation or a demonstration for every module.

## Interpreting widths

The measured bar-pair width includes both bar-average timing fluctuations and
their correlations. It is not the resolution of either individual bar.
The quantity 0.5 sqrt(sigma(L-R,ref)^2 + sigma(L-R,DUT)^2) is an illustrative
independent-end prediction only: it assumes negligible position spread and
end-to-end/common timing correlations. Left-minus-right cancels common timing
fluctuations and includes propagation-time variation from hit position.
Therefore it cannot establish the individual bar-average resolution alone.

The report also checks the exact sample-variance identity:

Var(D-R) = [Var(L-R,ref) + Var(L-R,DUT)]/4
           + Cov(ref L,ref R) + Cov(DUT L,DUT R) - 2 Cov(ref mean,DUT mean).

This identity applies to sample variances, not automatically to separately
fitted Gaussian core widths. No square-root-of-two single-bar claim is made.

## Reproduction

Run `python DPS_NOTES/review_2026-10-23_reference_transfer/run.py` from the
workspace root. The first run selects a compact input snapshot from the frozen
clock-aligned cache; subsequent runs use that snapshot. No ROOT replay or
clock refit is required. Numerical CSVs, fits, provenance, and a report are
saved beside the script. This does not modify the original six-setting results.

Run `python DPS_NOTES/review_2026-10-23_reference_transfer/prepare_slides.py`
afterward to render the workflow and bar-pair timing closure, and export
the numerical LaTeX macros. Then rebuild the short slide decks.

## First results

1512 events enter both downstream calibration paths; 1456 events remain in
their common fitted domains for every displayed before/after comparison.
The bar-pair core width is 91.03 ps before any correction, 81.09 ps with only
the reference corrected, 64.77 ps with direct MCP downstream calibration,
and 65.01 ps via the reference bar. The paired conditional bootstrap gives
a central 68% interval [-0.30, +0.44] ps for the transfer-minus-direct width
difference (observed +0.24 ps). The downstream-minus-MCP widths are 74.27
and 74.44 ps, respectively. Their agreement supports the calibration transfer
on this sample, without constituting an independent general validation.

The reference left-minus-right width changes from 54.63 to 53.90 ps; the
downstream width changes from 88.92 to 89.31 ps (direct) or 88.62 ps (transfer).
Thus the independent-end core-width estimate is 51.86 ps, not the observed
65.01 ps bar-pair width. The exact sample-variance check records a nonzero
covariance remainder; it does not isolate a particular source of noise.
Position spread and non-Gaussian components also limit interpretation of the
separately fitted core widths. Do not claim 65/sqrt(2) as a measured bar resolution.

The charge-dependent correction shapes differ by event-weighted RMS values
of 2.87 ps (left) and 5.14 ps (right) after constant alignment. Differences reach
12.88 and 23.05 ps at the common fitted-domain boundaries. These are shape
differences, not timing resolutions or statistically independent deviations.

## Slide 10 presentation update

The displayed closure now contains only DUT minus reference, with explicit
legend labels for the three calibration cases. The DUT-minus-MCP panel is
not needed for this comparison and does not isolate the origin of the 65 ps
width. Its original two-panel PDF is retained as `closure_timing_with_mcp.pdf`;
all numerical results and the DUT-minus-MCP histogram CSV remain unchanged.
