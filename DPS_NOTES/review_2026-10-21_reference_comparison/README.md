# Comparing the two MCP-reference versions

21 October 2026. Investigation of N1-4, not a closure of that review point.
Neither production fit nor either note PDF was changed by this comparison.

## Result

The four fitted shapes are close over jointly populated measured charge
support. They are not numerically identical, and the changes are not limited
to binning.

| Channel | Common outer Q range [ADC] | Shape RMS [ps] | Max absolute difference [ps] | Binning-only control RMS [ps] |
| --- | --- | --- | --- | --- |
| 133 | 208.2-404.8 | 2.20 | 7.44 | 1.41 |
| 154 | 303.4-546.6 | 3.91 | 8.75 | 0.14 |
| 137 | 187.8-400.7 | 1.65 | 3.97 | 1.04 |
| 150 | 312.8-646.9 | 4.88 | 10.42 | 0.70 |

The walk variation itself spans about 190-249 ps over these intervals. The
shape RMS is a descriptive difference between calibration functions, not a
detector timing resolution or an uncertainty on AWC. Large differences in
individual polynomial coefficients can cancel in the evaluated function.

## Definition of the comparison

The old functions are exactly the saved binned coefficients used by Note1.
The new functions are exactly the run-4399 snapshot coefficients used by Note2.
Both are quadratic in inverse charge, not in charge.

1. Express times in ns; add 26 ns to the newer reference to undo the known
   difference in historical display convention.
2. Restrict Q to the intersection of the two outer profile-mean ranges.
3. Use a uniform 20,001-point Q grid and require membership in a retained
   profile bin from each version. Missing bins are excluded, not bridged.
4. Subtract the mean new-minus-old difference over that grid, fitting one
   constant offset only. No multiplicative scaling or slope alignment is used.
5. Report RMS and maximum absolute residual over the supported grid.

The extra offset after accounting for 26 ns is -2.12, +0.84, +1.26 and -2.21 ps
for channels 133, 154, 137 and 150 respectively. Missing-bin gaps remove 2578
grid points for channel 133 and 961 for channel 137. The other two comparisons
have no gaps in their common interval. These exclusions explain small changes
from a quick initial comparison of the full outer interval.

`reference_comparison.pdf` has one page per channel: saved profile means with
SEM and the two aligned functions above; their difference in ps below.
Profile points within the outer interval remain visible even where the other
version has a gap. The plotted curves and reported metrics use joint support.

## It is not only a binning change

The archived profiles can be reconstructed to better than 4e-12 ps from the
archived `*_walk_before_walk_fit_data.csv` samples, after converting ps to ns
and adding 26 ns. The recovered requested/retained bin counts are:

| Channel | Old requested / retained | New requested / retained | Old / new profiled hits |
| --- | --- | --- | --- |
| 133 | 40 / 26 | 20 / 19 | 2456 / 2417 |
| 154 | 80 / 20 | 20 / 19 | 2526 / 2324 |
| 137 | 100 / 48 | 20 / 20 | 11703 / 11071 |
| 150 | 100 / 25 | 20 / 19 | 12120 / 10843 |

An unweighted quadratic fit to the archived profile means reproduces all four
saved old coefficient sets. The newer coefficient sets are reproduced by
SEM-weighted fits to their saved profiles. NumPy uses `w=1/SEM`, equivalent
to a squared-residual objective weighted by `1/SEM^2`.

Crucially, `ch192_vs_trigger_lowess.py` defines the `walk_before` sample using
the measured channel-192 time. Its `walk_expected` sample instead uses the
clock-reconstructed expected channel-192 time. Despite the `_expected_refit`
suffix in the old profile filename, the numerical profile reconstruction
matches the archived **before** sample. The new pipeline uses
`reconstruct_reference_delta_t` and the expected trigger time. Both subtract
the MCP peak-minus-trigger term; the N1-3 sign conclusion remains valid.

The newer pipeline also applies its own exact-five-hit topology, positive
finite quality checks (including the slope-related variables), a peak-left
charge threshold, and per-bin timing clipping. These are additional changes
to the pipeline, not effects isolated by this comparison. All historical
upstream command-line selections have not been recovered here.

## Controlled checks

For the binning-only control, take exactly the events represented in the old
retained bins, regroup them into 20 uniform inverse-Q bins, and fit their
means without weights or additional clipping. All those events are retained;
only the grouping changes. The resulting shape differences from the original
old fit are in the final column of the first table (0.14-1.41 ps RMS).

Changing only the weights of the old saved profile points from unweighted to
SEM-weighted gives 0.73, 1.80, 0.29 and 0.38 ps RMS, respectively. These control
RMS values are not additive and do not allocate the full old/new difference
among causes. No controlled timing-reconstruction or selection-only rerun was
performed.

This supports the expectation that binning alone produces small changes here.
It does not establish statistical equivalence of the two production versions:
the event samples can overlap, their covariance is not known, and no full
systematic uncertainty has been assigned to the difference.

## Reproduce and review

Run `python3 DPS_NOTES/review_2026-10-21_reference_comparison/compare.py`.
The script prefers the included input snapshots, so it can run in the online
repository without ROOT files. `input_manifest.json` records their hashes.
`summary.csv`, per-channel detail JSON and curve CSVs contain the numerical
results. Source-code snapshots document the inspected definitions.

Before choosing one common reference for both notes, explicitly decide which
timing-reconstruction branch is intended. The current result does not support
calling the old/new change a severe disagreement, nor calling it merely a
rebinning. N1-4 stays open until that version choice is agreed.
