# N2-4: reference uncertainty and charge-weighting sensitivity

This protocol is fixed before running the audit. Preserve the nominal linear
slope model, MCP quadratic, original measured-overlap grid (including gaps),
channel set, and free AWC-plus-offset fit. No ROOT extraction or model selection
is performed. The original comparison figures and conclusions are not replaced.

For all 22 references, reproduce the saved quadratic coefficients and full
covariance from the saved profile and existing fitter. The covariance is the
unscaled inverse information using profile SEM; it is not inflated by residual
chi-square and does not include clock-calibration or model systematics.

For the 21 channels with overlap, first reproduce the nominal AWC, offset and
1000-replica slope-only percentile interval. Treat the two runs' statistical
samples as independent, conditional on their fixed calibrations, selections and
support. For each slope replica, the alignment is a linear transformation of
the reference coefficient vector (a,b,c). Use this transformation to propagate
the entire correlated reference covariance. The reference approximation is
multivariate Gaussian, not an event bootstrap or a Bayesian posterior.

Report reference-only Gaussian intervals, the original slope-only percentiles,
and combined intervals from the equal-weight mixture of the conditional
Gaussians over valid slope replicas. Compute mixture percentiles numerically,
without Monte Carlo sampling noise. Preserve negative AWC and report invalid
replicas. A common reference-time offset must affect the fitted offset but
not AWC. Never interpret the 3000-point grid as independent measurements.

As objective sensitivity only, fit the same two central curves on exactly the
same supported Q points with four positive weight functions:

1. Uniform Q (nominal).
2. Uniform inverse Q, implemented as d(1/Q)/dQ magnitude = 1/Q^2.
3. Run4076 selected population density: profile count / Q-bin width.
4. Run4399 selected population density: profile count / Q-bin width, after
   converting the inverse-Q bin edges to Q.

The latter densities use saved bin counts, not unbinned events. Normalize
weights over retained support. Report each AWC/offset and its shift; keep the
maximum shift separate from statistical intervals. These choices answer
different charge-averaging questions and are not competing likelihood fits.
No claim that any weighting is uniquely optimal, or that this envelope is a
calibrated systematic uncertainty, is made.

Do not add slope-model alternatives to Note2. The combined statistical result
is still conditional: cross-run response changes, shared calibration effects,
model/selection choices and cross-channel correlations remain outside it.
