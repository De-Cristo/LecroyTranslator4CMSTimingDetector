# N2-6 closure: exploratory channel coverage

Closed with author approval on 22 October 2026 for this version's reporting
scope. The author explicitly requested retaining the ten-channel results.
No fit, bootstrap interval, figure, selection or conclusion is changed.

## Coverage and inclusion

The historical explored set contains 22 channels, not a statistically
representative or newly quality-selected detector sample:

- Upstream: 130, 133, 134, 135, 136, 137, 139, 149, 150, 151, 154, 155, 156, 159.
- Downstream: 235, 236, 238, 239, 240, 242, 244, 245.

The existing matching implementation requires intersecting outer profile-mean
ranges, at least three jointly populated points on its 3000-point charge grid,
a positive central slope on that support, a nonflat inverse-slope predictor,
and valid bootstrap replicas with finite positive slopes. These are numerical
completion conditions, not a demonstrated calibration-quality criterion.
The grid points sample fitted curves; they are not independent data points.
The implementation is `Q_Walk/results/2026-09-24_measured_q_overlap/recompute.py`,
reused by the 22 October clock-quality refresh.

Twenty-one fits completed and remain in the full summary. Channel 238 has no
measured profile overlap and therefore no AWC estimate. Its saved failure is
copied verbatim to `failures.json`. No channel is newly excluded because of its
AWC, interval, trend agreement or model mismatch.

The ten upstream channels 133, 134, 135, 137, 150, 151, 154, 155, 156 and 159
remain an interactively selected illustration. Their points, errors and saved
weighted straight-line fit are unchanged. Membership is not a reproducible
quality rule and cannot establish a population-level law.

## Documented limitations

The new slide reports diagnostic examples, not an exhaustive quality ranking:

- Channels 235 and 244 have overlap spans of 3.10 and 16.99 ADC. Each has one
  profile mean from each run inside the outer overlap. Their fit parameters
  use the full input profiles; the many curve-grid points do not increase
  the measured information inside these narrow intervals.
- Channels 130, 236 and 239 have slope-only 16th--84th percentile intervals
  spanning zero and hundreds of microamperes. A central estimate need not lie
  inside this interval. They remain visible as poorly constrained cases.
- Channels 136 and 245 have negative central AWC, as do 236 and 239. These
  signed fit results are retained, not interpreted as validated positive
  correction amplitudes.
- Reference changes and sparse clock groups remain limitations on the
  existing reference-sensitivity slide. A small model mismatch, especially
  over narrow charge support, does not validate an event-level calibration.

The complete numerical inventory is already packaged in Note2:
`../note2_dual_threshold/data/sandbox_test/Q_Walk/results/2026-10-22_clock_quality/channels.csv`.
It records all 21 support ranges, supported/gap grid counts, in-overlap profile
mean counts, signed AWC and conditional percentiles. The adjacent
`awc_summary_special_10_channels_no_ch240_linear.csv` and `_linear_fit.json`
preserve the ten-channel points and fitted line. Figure-manifest hashes verify
these snapshots. The source failure record is
`Q_Walk/results/2026-10-22_clock_quality/failures.json`.

## Scope and verification

The inline N2-6 comments and queue entry are replaced with permanent scope
statements and the status table. This closes documentation for an exploratory
comparison, not the design of an unbiased performance sample. Independent
eligibility criteria, reference stability and adequate clock statistics remain
necessary before claiming detector-wide calibration performance.

The publication verifier checks channel membership, the diagnostic values,
the retained full and ten-channel slides, and absence of N2-6 review markers.
The existing frozen-data verification is reused. Both Note2 variants and the
combined review PDF are rebuilt, and the changed slides are inspected.
