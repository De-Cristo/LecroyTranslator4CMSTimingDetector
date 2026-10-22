# Same-event bar-to-bar walk-correction validation

**Superseded as a resolution result:** this sample mixes six first-threshold
settings. Their displaced timing means broaden the combined distribution.
Use `../2026-10-22_bar_pair_by_threshold/` for independently selected, calibrated
and fitted operating points. The original numerical artifacts below are kept
unchanged for traceability, not as a fixed-operating-point resolution.

22 October 2026. Separate diagnostic result; no note slides or prior calibration
files changed. Use `bar_pair_before_after.pdf` for the histogram and
`charge_dependence.pdf` for the four charge profiles. PNG versions are included.

## Observable and selection

The observable is downstream bar 3 minus reference bar 8:

`B = ((t234 + t243) - (t137 + t150)) / 2`.

The corrected difference is
`B_corr = B - (W234(Q234) + W243(Q243) - W137(Q137) - W150(Q150)) / 2`.
Each W is the per-channel MCP-derived quadratic in inverse Q, in ps.
The calculations subtract timestamps before averaging to preserve precision.
Computing B from the saved MCP residuals gives identical values: the shared
MCP and clock mapping cancel in this observable, but their effects on the
fitted calibration curves do not thereby disappear.

Both states use the same 8,737 events from the preceding four-channel
compatibility check: exact five-hit topology {137,150,234,243,192}, unique MCP
match, refreshed clock quality, all four channels passing signal quality and
peak-left charge cuts, and all four charges inside populated fit support.
No final timing-tail cut or segment-by-segment recentering is applied.

## Results

| Statistic | Before walk correction | After walk correction |
| --- | ---: | ---: |
| Fitted Gaussian-core sigma [ps] | 161.70 | 141.79 |
| Conditional block-bootstrap 16th-84th percentile [ps] | 156.91-165.61 | 138.00-145.60 |
| Full-sample standard deviation [ps] | 159.87 | 139.46 |
| Fraction outside core-fit window | 0.767% | 0.217% |

The fitted core decreases by 19.91 ps, or 12.31%. The paired-bootstrap
16th-84th percentile interval for the decrease is 18.30-20.98 ps
(11.46-12.90%). All 400 replicas fit successfully in both states.

The fit uses the same truncated-normal maximum-likelihood implementation as
the prior Note 1 validation, within each state's median +/- 2.5 scaled MAD.
The 400 paired bootstrap replicas resample 174 file/segment blocks, using the
same sampled blocks before and after. Calibrations and selection stay fixed.
These are conditional statistical intervals, not total uncertainties.

Both histograms use identical 20 ps bins. Each distribution is independently
median-centered only for display. The raw timing shift is not a jitter
improvement. Events outside the displayed range remain in the saved event
table and full-sample statistics; counts are recorded in `report.json`.

## Interpretation and limitations

The correction improves the bar-to-bar timing spread and strongly reduces
the charge dependence. Charge profiles use ten equal-population bins per
channel and all selected events, without timing clipping; errors are SEM.
Their zero is the corresponding global median, not the global mean, so a
nonzero constant level is not by itself evidence of residual walk.

The corrected distribution remains asymmetric. Varying the core half-window
from 2 to 4 scaled MAD gives widths of 148.78 to 138.46 ps after correction
(165.40 to 159.04 ps before). Every tested window shows improvement, but the
absolute fitted width is model/window dependent. It should not be presented
as a precisely Gaussian detector resolution.

This is in-sample validation: the fits used overlapping calibration events.
No calibration refitting is performed in the bootstrap. Independent validation,
residual segment/charge dependence, geometry and inter-bar correlations still
need consideration before a performance claim. Dividing by sqrt(2) would
require independent, equal-resolution bars and negligible flight-time spread;
this study does not establish those assumptions.

## Saved evidence

- `report.json`: widths, conditional intervals, window sensitivity and definition.
- `selected_events.csv.gz`: event IDs, file/segment, charges and timing differences.
- `histogram_bins.csv`: identical before/after histogram bin edges and counts.
- `paired_block_bootstrap.csv`: paired width and improvement replicas.
- `charge_profiles.csv`: bin means, SEM and populations.
- `segment_statistics.csv`: per-file/segment medians, SD and populations.
- `input_manifest.json`: hashes of code, cached events and fixed fit inputs.

Reproduce from the workspace root in the existing analysis environment:

```bash
OPENBLAS_NUM_THREADS=1 python3 Q_Walk/results/2026-10-22_bar_pair_timing/validate.py
```
