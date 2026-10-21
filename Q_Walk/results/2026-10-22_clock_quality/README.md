# Clock-quality reference refresh, 22 October 2026

The run4399 references are refitted after finite-timestamp segmentation and a
whole-clock-group quality requirement. Consecutive finite trigger times define
resets even when missing timestamps intervene. Nonfinite triggers receive
segment -1 and cannot supply a valid reference time.

Reject an entire topology/file/segment when the scaled MAD of the robust
ch192-versus-trigger line residuals exceeds 1000 ps. This scale uses all finite
matched trigger events, before LOWESS, without selecting on signal-channel
timing. The threshold is motivated by the separated populations documented in
`DPS_NOTES/review_2026-10-22_bar8_validation/RECOMMENDATION.md`. It is not a
per-event timing cut. No timestamp rows are shifted or rematched.

All other reference-fit selections and the run4076 slope fits are unchanged.
AWC and a free offset are refitted only on jointly populated measured Q support.
Bootstrap AWC intervals still vary only run4076 slope coefficients, holding the
new reference fixed; they do not include reference-fit or cross-run systematics.

Reproduce from sandbox_test (numpy, pandas, scipy, matplotlib, awkward, uproot):

```bash
python Q_Walk/qwalk_extended_t0_reference.py \
  --out-dir Q_Walk/results/2026-10-22_clock_quality/reference_plots \
  --reference-dir Q_Walk/results/2026-10-22_clock_quality/references \
  --workers 4 --cache-dir Q_Walk/results/2026-10-22_clock_quality/cache
python Q_Walk/results/2026-10-22_clock_quality/refresh.py
```

The refresh script reuses the previous overlap/predictor implementation and
timing-validation implementation, redirecting their output into new directories.
Its recorded inputs identify the new reference version. Historical results are
preserved. `reference_update_awc_impact.csv` compares old and new AWC results.
Timing outputs are in `DPS_NOTES/review_2026-10-22_clock_quality/`.
