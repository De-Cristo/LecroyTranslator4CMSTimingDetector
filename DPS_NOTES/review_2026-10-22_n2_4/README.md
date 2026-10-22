# N2-4: statistical propagation and charge-weighting check

22 October 2026. The original linear-slope versus clock-corrected MCP
comparison is retained. Central AWC values, curve shapes, selections, measured
support and conclusions are unchanged. No quadratic slope model is introduced.

## Result

All 22 saved MCP reference coefficient vectors and covariances reproduce from
their profiles. All 21 original AWC fits and slope-only intervals reproduce;
channel 238 still has no measured overlap. All 1000 slope replicas per fitted
channel remain usable. The following numbers are in microamperes:

| Channel | Nominal AWC | Slope-only 16--84% | Both-run 16--84% | Maximum weighting shift |
|---|---:|---|---|---:|
| 133 | 36.03 | [32.73, 44.75] | [32.48, 45.72] | 4.11 |
| 154 | 24.25 | [23.92, 24.86] | [22.98, 25.84] | 0.85 |
| 137 | 18.33 | [18.12, 18.61] | [17.91, 18.80] | 2.05 |
| 150 | 20.89 | [20.65, 21.15] | [20.41, 21.38] | 1.13 |

For channel 137, replacing uniform Q by uniform inverse Q shifts AWC by
+0.68 uA, weighting by the run4076 population shifts it by +1.72 uA,
and weighting by the run4399 population shifts it by +2.05 uA.
These exceed the combined statistical half-width in this example. Charge
weighting is therefore not interchangeable at the quoted statistical precision.
The same fixed models disagree in shape, so different charge priorities can
give different best-matching amplitudes. This is not a new slope-model test.

The reference contribution is especially important for channel 136: its
combined interval [-13.66, 4.93] uA crosses zero. Poorly constrained channels
130, 236 and 239 retain broad/non-Gaussian slope-replica distributions; their
nominal fit can lie outside the percentile interval. No channel is removed to
improve the summary. A small weighting shift on very narrow overlap (e.g.
channel 235) is not proof of a well-tested broad-range calibration.

## Statistical definition

On the unchanged supported Q grid, let x(Q)=1/(c0+c1 Q) and
W(Q)=a+b/Q+c/Q^2. For fixed slope coefficients and weights w, solve
W(Q)=A x(Q)+B by weighted least squares. Its result is exactly linear in
the reference coefficients beta=(a,b,c):

```text
(A, B) = M beta
Cov(A, B | slope replica) = M C_reference M^T
```

The implementation uses weighted centered sums, with the AWC response to
the constant coefficient a set to zero. Thus a common time-zero fluctuation
changes B but not A. Off-diagonal covariance terms are retained. The dense
grid is an integration grid, not a collection of independent measurements.

The reference coefficients are approximated by a multivariate Gaussian with
the saved covariance. For each of the 1000 run4076 slope-bootstrap replicas,
calculate the conditional Gaussian AWC mean and variance. The combined CDF
is the equally weighted mixture of those 1000 Gaussian CDFs. Root-finding
gives its 16th, 50th and 84th percentiles, without Monte Carlo noise. This is
not an event-level MCP bootstrap or a Bayesian posterior. The two runs contain
different events; their statistical errors are assumed independent conditional
on the fixed calibration and selection. Shared systematic effects are not
made independent by this assumption.

The reference covariance is the unscaled inverse information of the
SEM-weighted profile fit. It is not multiplied by chi-square/ndf. The audit
records reference chi-square (0.71--1.65 per degree of freedom here), but
does not redefine the reference fit or its uncertainty convention.

The old slope-only intervals are empirical interpolated bootstrap percentiles.
The combined intervals are Gaussian-mixture quantiles; their endpoints need
not widen monotonically in every channel, particularly for irregular empirical
distributions. The mixture variance obeys the law of total variance and
includes a nonnegative additional reference contribution.

## Weighting definition

`study_definition.md` fixes four objectives before execution. All use identical
central models and exactly the same retained Q points, including support gaps:

- Uniform Q: current nominal shape-matching definition.
- Uniform inverse Q: weight 1/Q^2 on the uniform-Q grid.
- Run4076 population: saved selected count divided by each Q-bin width.
- Run4399 population: saved selected count divided by each converted Q-bin
  width, not divided by its original inverse-Q width.

Weights are normalized on retained support. The population estimates are
piecewise constant and are not unbinned event reweighting. Their estimation
errors are not propagated. `weighting.csv` reports both uniform-Q and weighted
RMS; weighted RMS values optimize different objectives and should not be used
to select the "best" objective by their size.

## Recommendation and review status

Keep uniform-Q matching as the declared nominal definition: it gives equal
weight per charge interval rather than using either run's population weights. Report the
combined conditional statistical interval and the weighting sensitivity
separately. Do not add the maximum weighting shift in quadrature as though
it were a measured one-sigma error. Do not assert that weighting is negligible.

The requested reference propagation and weighting audit are complete.
N2-4 remains open for acceptance of this reporting scope; this is not a complete
calibration uncertainty budget. Clock/LOWESS calibration, fine-time lookup,
cross-run response transfer, model/selection choices, random support boundaries,
and cross-channel correlations remain unpropagated. The original Note2 curve
bands and summary bars retain their clearly labelled slope-only definition;
the new table separately reports the expanded conditional intervals. No
selected-subset trend fit or internal predictor is refitted in this audit.

## Files and verification

- `uncertainties.csv`: all 21 central values and separate uncertainty intervals.
- `weighting.csv`: all 84 fits, signed shifts and RMS definitions.
- `reference_checks.csv`: all 22 reference coefficient/covariance replays.
- `channel_*_propagation.npz`: grids, coefficient maps, full covariance,
  slope replicas, mixture components and weight functions.
- `report.json`: input and code hashes, assumptions and exclusions.
- `figures/`: four-channel uncertainty budget and full weighting sensitivity.
- `uncertainty_table.tex`: generated four-channel table used in Note2.

From the analysis root, with the existing scientific Python environment:

```bash
python -m pytest DPS_NOTES/review_2026-10-22_n2_4/test_uncertainty.py -q
python DPS_NOTES/review_2026-10-22_n2_4/study.py
python DPS_NOTES/review_2026-10-22_n2_4/plot.py
python DPS_NOTES/review_2026-10-22_n2_4/verify.py
```

No ROOT data or GPU is required. Six focused tests cover direct weighted
least squares, offset invariance, correlated covariance propagation against
sampling, mixture quantiles, bin-density units/support and invalid slopes.
The independent verifier checks all 22 reference covariances, 84 weighted
fits, all combined-interval CDF probabilities, representative bootstrap maps,
frozen inputs and both labelled PDF figures.
