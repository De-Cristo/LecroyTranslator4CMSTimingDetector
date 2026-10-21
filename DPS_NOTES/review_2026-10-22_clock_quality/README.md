# N1-5 implementation and validation, 22 October 2026

The accepted modifications replace historical timing claims with reproducible
same-event comparisons. N1-5 is closed by author decision for this version's
conditional timing-validation scope. Intrinsic bar resolution and separate
synchronization-jitter extraction are outside scope; their unresolved physical
and systematic inputs have not been validated by this closure.

## Production changes

`Q_Walk/qwalk_t0_data.py` now detects resets between consecutive finite trigger
timestamps, including across missing values. Invalid timestamps receive segment
-1. Before LOWESS, groups with clock-line scaled MAD above 1000 ps are rejected
wholesale; the statistic uses all finite matched trigger events, independent of
SiPM arrival time. The bad associations are not repaired by shifting MCP rows.

All 60 run4399 files and 22 requested reference channels were reprocessed.
References, clock-group audits, updated overlap fits and their input snapshots
are under `Q_Walk/results/2026-10-22_clock_quality/`. All reference fits complete;
21 AWC comparisons have measured overlap. Channel 238 still has none.
Run4076 fits and bootstrap replicas are unchanged. Earlier results are preserved.

## Bar-8 validation

The exact topology is {137,150,234,243,192} with a unique MCP match. There are
12,481 matched events; 11,826 pass both-channel quality after clock rejection,
10,517 additionally pass both peak-left cuts, and 10,127 lie in both populated
reference supports within the outer profile means. Both distributions use
exactly those events before/after correction, without a SiPM timing-tail cut.

| Quantity | Before walk [ps] | After walk [ps] | After: block-bootstrap 16-84% [ps] | After: full SD [ps] | After: outside core window |
| --- | ---: | ---: | --- | ---: | ---: |
| End-difference core sigma | 55.16 | 54.82 | 54.32-55.22 | 61.26 | 1.57% |
| Bar-minus-mapped-MCP core sigma | 77.69 | 64.16 | 63.27-64.99 | 70.33 | 1.56% |

Core fits maximize a truncated-normal likelihood inside median +/-2.5 scaled
MAD. The 400-replica file/segment-block bootstrap resamples groups with the
clock and walk calibrations fixed. These intervals are conditional statistical
uncertainties, not total errors. Full SD uses all selected events, including
outside the plotted range. Median subtraction is for display only.

Six finite-trigger groups fail the clock threshold; four additional groups
contain nonfinite triggers. There are 174 retained groups, with 12,094 events
before signal quality and Q selection. Thresholds from 1 ns to 1 microsecond
give the same retained groups. No segment-by-segment centering is applied to
the headline result; residual segment offsets remain visible in backup.

Holding out segment ordinals for walk fitting gives D=54.87 and R=64.32 ps,
versus D=54.84 and R=64.16 ps with the frozen fits on the same 10,124 events.
Holding out alternate trigger events gives R=63.92 versus 63.80 ps on the same
9,844 events. These are separate conditional checks, not a joint independent
clock-and-walk validation.

## Effect on Note 2

Channel 137 AWC changes from 18.513 to 18.332 uA; its shape RMS is 6.80 ps.
Channel 136 changes from +27.899 to -4.240 uA, and channel 151 shifts from
7.649 to 9.354 uA. These are material stability warnings, not new validated
calibrations. The complete impact table is `reference_update_awc_impact.csv`
in the new Q_Walk result directory. Sparse groups retain the existing line-only
fallback below ten alignment inliers; their small fitted residual scale alone
does not validate association. Eligibility remains N2-6, and reference/transfer
uncertainties remain outside the slope-only AWC intervals.

## Note changes and reproduction

The main timing plots now state the sample, fit definition, bootstrap interval,
SD and tails. Historical 52.87/46.8 ps images are explicitly historical backups.
The text no longer infers intrinsic 26 ps bar resolution or a synchronization
term from a quadrature difference of separate Gaussian-core fits. Both notes
use the same new reference version, including all affected AWC plots and captions.

Run `Q_Walk/results/2026-10-22_clock_quality/refresh.py` after its documented
reference-extraction command. That script reuses the previous validation and
comparison functions, directs outputs here and records new input snapshots.
`validation_report.json` is the numeric authority; `validation.pdf` contains
the diagnostic panels. `reference_comparison/` holds the updated archived-vs-new
reference check. `input_manifest.json` records source and calibration hashes.
