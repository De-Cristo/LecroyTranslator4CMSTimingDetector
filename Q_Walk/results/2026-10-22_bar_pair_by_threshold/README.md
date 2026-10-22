# Threshold-separated bar-to-bar timing

22 October 2026. This supersedes the mixed-threshold timing interpretation in
`2026-10-22_bar_pair_timing`. The older artifacts are preserved as diagnostics,
not as a fixed-operating-point detector resolution.

## Results

| First threshold DAC | Second threshold DAC | Selected events | Before core sigma [ps] | After core sigma [ps] | After: conditional 16th-84th percentile [ps] |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 60 | 1461 | 91.21 | 64.70 | 63.00-66.10 |
| 5 | 60 | 1383 | 92.37 | 62.13 | 60.59-63.89 |
| 7 | 60 | 1514 | 92.42 | 63.05 | 61.57-64.26 |
| 10 | 60 | 1377 | 96.24 | 66.24 | 64.55-67.35 |
| 15 | 60 | 1407 | 103.30 | 68.22 | 66.72-69.27 |
| 20 | 60 | 1530 | 112.65 | 69.44 | 68.16-70.70 |

All six operating points improve. The original note's 58.2 ps figure used
1,011 events and has not been exactly reproduced here; no assertion of matching
its full selection, calibration or fit procedure is made.

## Selection and calibration

The input is the clock-quality-refreshed four-channel event cache from all
60 run4399 files. Exact five-hit topology is {137,150,234,243,192}, with a unique
matched MCP record. The upstream reference is bar 8 and the downstream bar is
bar 3. The code first groups by the complete (step1, step2) scan setting; step1
is 1.2 throughout this cache. Each event and source file is verified to contain
only one setting. Existing clock reconstruction operates within file/segment
groups, which also contain only one setting.

For each setting, independently:

1. Combine its ten source files, not files from other threshold settings.
2. Require finite positive charge, slope, Delta I and threshold-rise time,
   and finite MCP-referenced timing after the recorded clock-quality gate.
3. Find each channel's charge-spectrum left half-maximum boundary using the
   existing histogram peak finder. Keep charge at or above that boundary.
   This is not a newly fitted Landau window or an additional MCP amplitude cut.
4. Construct 20 uniform inverse-Q bins across that channel's selected range,
   clip timing at five scaled MAD per bin and require at least five entries.
   Fit the bin means with a SEM-weighted quadratic in inverse Q.
5. For timing validation, require all four channels to pass quality, charge
   cuts and populated fit support, bounded by the outer profile mean positions.
   No walk-curve extrapolation or final timing-tail selection is used.
6. Compare downstream-minus-reference bar averages before and after applying
   that setting's four corrections to exactly the same events.

The 8,672 selected events are not forced to equal the previous 8,737-event
sample: charge boundaries, binning, fits and support were recomputed within
each setting. Before/after membership is identical within every setting.
No per-file, per-segment or per-threshold offset correction is applied to the
timing data. Median subtraction is only for histogram display, separately
before and after within each setting. No combined timing histogram is made.

## Widths and uncertainties

Core widths use the previous truncated-normal maximum-likelihood fit in each
state's median +/- 2.5 scaled MAD window. The full-sample SD, tail fraction and
window variations (2, 3 and 4 scaled MAD) are stored in each report.

There are 400 paired file/segment-block bootstrap replicas per setting.
The same sampled blocks are used before and after. Calibrations and selections
are fixed. These are conditional statistical intervals, not full calibration
uncertainties. Fits and evaluation use overlapping events; this is in-sample
validation. The result is a two-bar difference, not an individual bar resolution.

## Combining measured widths only

The optional descriptive summary uses the already fitted widths:

`sigma_summary = sqrt(sum((N_k - 1) * sigma_k^2) / sum(N_k - 1))`.

This averages within-setting fitted variances, without including the offsets
between settings. It is not a new histogram fit, an inverse-variance estimate
of a universal resolution, or a claim that all thresholds have equal resolution.
Weights use the selected event populations and are fixed in the bootstrap.
The per-setting bootstrap streams use distinct random seeds; their paired
replicas are combined with the same formula.

- Before: 98.46 ps; conditional interval 97.62-99.48 ps.
- After: 65.71 ps; conditional interval 65.06-66.26 ps.

Threshold dependence remains visible in the primary plot and table. Shared
calibration/systematic effects are not represented by these combined intervals.

## Outputs

- `before_after_all_thresholds.pdf`: six independent before/after histograms.
- `width_vs_threshold.pdf`: per-setting widths and bootstrap intervals.
- `width_summary.csv`: numerical table, also including reference-bar-minus-MCP.
- `combined_width_summary.json`: optional width-only combination and weights.
- `reference_mcp_vs_threshold.pdf`: independent reference-bar/MCP check.
- `vth1_*_vth2_60/`: four calibration profiles and coefficient/covariance JSONs,
  selected event tables, bootstrap replicas, core-window checks and individual plots.
- `input_manifest.json`: hashes of the cache and analysis code.

The short Note 1 uses these threshold-separated results. Its calibration
example is the lowest scanned threshold, not a threshold chosen for the best
measured width. Full legacy notes and Note 2 are not reanalysed by this change.

Reproduce in the existing analysis environment from the workspace root:

```bash
OPENBLAS_NUM_THREADS=1 python3 Q_Walk/results/2026-10-22_bar_pair_by_threshold/run.py
```

The source extraction cache remains local; saved numerical outputs and figures
are synchronized for review and slide recompilation. Full regeneration requires
that cache or its documented ROOT extraction inputs.
