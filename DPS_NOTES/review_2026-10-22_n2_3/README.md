# N2-3: linear-profile adequacy and AWC sensitivity

22 October 2026. **Temporarily closed by author approval for exploratory scope.**
Linearity remains a working approximation, not a validated precision model;
model and selection dependence are outside the quoted errors. The author
will revisit decisions in a page-by-page review. No nominal calibration,
selection, bootstrap band or reference
curve has been replaced. This directory records an exploratory audit, not a
new recommended calibration or a total-uncertainty prescription.

## Method

`study_definition.md` fixes the variants and interpretation before execution.
`study.py` replays the original run4076 `23_e.root` file with safe second-time
reconstruction and the frozen pair/prompt/charge selection. All 22 nominal
profiles reproduce their saved counts, bin statistics and coefficients. All
21 available nominal MCP alignments also reproduce on their original grids.
Channel 238 has no saved measured overlap with the run4399 reference.

Each channel has eight fits: nominal ten-bin linear, quadratic on identical
hits/bins, 8 and 12 bins, charge boundary multiplied by 0.95 and 1.05, and a
100 ps minimum rise time with either the frozen or recomputed histogram
boundary. The boundary is the smoothed-spectrum half-height edge, not a
Landau-Gaussian fit parameter. There is no new event clipping or channel removal.

Profile means are weighted by SEM (sample SD divided by sqrt(count)). For a
quadratic, its prediction in a bin uses the actual mean of Q squared, not
the square of mean Q. Residual chi-square/ndf conditions on the reconstructed
sample and these SEM estimates; it is a diagnostic, not a calibrated p-value.
The original fine-time calibration and its correlations are not resampled.

For AWC, all eight alternatives use the same run4399 reference and the same
uniform-Q grid restricted to support common to every alternative and the
saved measured-overlap domain. Gaps remain excluded; no extrapolation is
introduced. Fit reference(Q) = AWC / S(Q) + offset by least squares. Negative
AWC is retained and flagged, not hidden. Original-to-common-grid shifts are
recorded separately from model/selection shifts. The saved 1000 baseline
coefficient replicas are propagated on this common grid for a conditional
16th--84th percentile interval. No alternative-model bootstrap is claimed.

## Findings

176 profile fits completed. 20 channels retain all-variant common support.
Channel 235 originally had only 339.23--342.33 ADC of reference overlap;
the 8-bin and raised-boundary profiles start beyond it. Its comparisons are
therefore unavailable, distinct from channel 238's absent original overlap.

| Channel | Linear chi-square/ndf | Quadratic chi-square/ndf | Common-grid linear AWC (uA) | Quadratic AWC (uA) | Maximum absolute AWC shift (uA) |
|---|---:|---:|---:|---:|---:|
| 133 | 17.38 | 5.63 | 35.65 | 36.92 | 3.33 |
| 154 | 8.07 | 9.01 | 24.18 | 23.59 | 1.07 |
| 137 | 24.72 | 3.52 | 18.24 | 22.99 | 4.75 |
| 150 | 8.72 | 2.29 | 21.01 | 23.46 | 2.44 |

For channel 137 the baseline conditional interval is [18.03, 18.52] uA,
but quadratic gives 22.99 uA. Its MCP model-closure RMS worsens from 6.65 to
9.14 ps despite the improved local slope fit. Channel 150 behaves similarly;
channel 133 improves in both comparisons. A quadratic is therefore not an
automatically validated replacement. Raw chi-square cannot increase for the
nested quadratic fit, but chi-square/ndf can (e.g. channel 154).

Channels 236, 239 and 244 change AWC sign across variants. Only 17.8% of the
original comparison grid remains for channel 244, so this is a restricted
diagnostic, not evidence of robustness. Channel 130 has a 425 uA maximum shift
but does not change sign. Channels 136 and 245 remain negative.

The variation envelope exceeds the baseline conditional half-interval for
18 of the 20 channels with common support. This compares scales only:
alternatives share events and are correlated. The envelope is **not** a
confidence interval, a measured systematic uncertainty, or a significance.

## Recommendation and closure condition

Do not declare the linear model generally adequate or select the replacement
solely because it resembles the MCP curve. Decide a physically acceptable
slope model and fitting range from run4076 profile behavior, then repeat the
fixed MCP comparison. Preserve unsupported and unstable-channel diagnostics.
The author chose the alternative, limited scope for this version: retain the
original comparison as an exploratory model with these observed limitations.
This closes the review comment, not the model-validation question. The audit
is preserved separately; model/range revalidation remains necessary for
precision calibration claims. N2-4 documents conditional statistical errors
and separate weighting sensitivity, and N2-6 documents exploratory channel
coverage. Neither supplies a total model/selection uncertainty.

## Outputs and reproduction

- `profiles.csv`: bin means, SD, SEM, counts, predictions and residuals.
- `models.json`, `variants.csv`: all 176 model/selection cases and alignment results.
- `channels.csv`: common support, baseline interval and variation envelope.
- `common_grid.csv`: exact retained coordinates and fixed MCP reference.
- `report.json`: original input/code hashes, replay flags and per-channel summary.
- `figures/`: residual overview, AWC sensitivity map and 22 channel diagnostics;
  all plots carry the CMS Preliminary Phase 2 label.

Using Python with numpy, pandas, scipy, matplotlib, uproot, awkward, pypdf
and pytest, from the analysis root:

```bash
python -m pytest DPS_NOTES/review_2026-10-22_n2_3/test_study.py -q
python DPS_NOTES/review_2026-10-22_n2_3/study.py
python DPS_NOTES/review_2026-10-22_n2_3/plot.py
python DPS_NOTES/review_2026-10-22_n2_3/verify.py
```

The study requires the documented EOS ROOT input; plotting uses saved tables.
Verification independently recomputes all 160 common-grid alignments and
profile chi-squares, checks support and input/code hashes, and reads all
25 figure PDFs. The focused four-test suite also checks finite-bin quadratic
averaging, support gaps, signed alignment and interior quadratic minima.
