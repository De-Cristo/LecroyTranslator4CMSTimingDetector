# Downstream/reference-bar calibration compatibility check

22 October 2026. This is a separate audit, not a change to the note or its
frozen calibrations. The candidate fits are sufficient to proceed to a
bar-to-bar timing validation; no timing-resolution result is established here.

## Findings

- The reference pair is upstream bar 8, channels 137 and 150. The coincident
  downstream pair is bar 3, channels 234 and 243. The source sample requires
  the exact five-hit topology including channel 192 and a unique MCP match.
- The older full-run downstream fits use 60 input files and the same topology,
  but were not included in the October clock-quality refresh. That refresh's
  downstream list covers bars 1, 2, 4 and 6, not bar 3.
- The old downstream runner uses the default IQR-based inverse-Q window;
  the current reference runner uses 20 uniform bins over the full selected
  inverse-Q range. Both use bin-wise 5 scaled-MAD timing clipping, at least
  five retained events per bin, and SEM-weighted quadratic fits.
- The refreshed upstream-bar-8 cache already contains all four signal
  channels: 12,481 matched events from 60 files, of which 12,094 pass the
  recorded clock-quality gate. A new ROOT extraction is unnecessary for this
  compatibility check.

## Numerical check

The audit reruns the current profile/fit procedure on the cached sample for
all four channels. Upstream bin occupancies match the saved current profiles
exactly; profile means agree within 0.001 ps and coefficients within numerical
tolerance. Refitting each old downstream CSV also reproduces its saved
coefficients. This checks the saved numerical provenance, not the original
ROOT matching independently.

| Channel | Current quality hits | After peak-left cut | Retained profile bins | Current fit support [ADC] | Old/new shape RMS [ps] |
| --- | ---: | ---: | ---: | --- | ---: |
| 137 | 12028 | 11053 | 20 | 189.38-585.69 | upstream reproduced |
| 150 | 11843 | 10863 | 19 | 314.78-665.60 | upstream reproduced |
| 234 | 12094 | 11195 | 17 | 51.36-157.51 | 0.213 |
| 243 | 12093 | 11242 | 16 | 116.23-275.66 | 0.581 |

Shape RMS is evaluated on a uniform-Q grid within common old/new fit support,
after subtracting the mean curve difference. It is a diagnostic difference,
not an uncertainty or a timing resolution. Maximum absolute centered
differences are 0.522 ps and 1.218 ps for channels 234 and 243, respectively.
Keeping the old cut and bin edges while using the refreshed clock sample gives
shape RMS differences of 0.203 ps and 0.200 ps, respectively. This separates
clock/sample changes from the additional effects of recomputed cuts/binning;
it is not a full systematic-error decomposition.

Requiring all four channels simultaneously leaves:

- 11,825 events passing channel quality and clock quality;
- 9,415 additionally passing all four peak-left cuts;
- 8,737 additionally within all four fit supports and occupied profile bins.

The last sample uses neither extrapolated corrections nor a cut on the final
bar-to-bar time difference. Its size differs from the note's 10,127-event
bar-MCP sample because the latter imposes signal quality and charge support
on the upstream pair only. Timing clipping used in fitting profiles is not
applied as an event selection for this common validation sample.

## Recommendation and remaining work

Use the refreshed downstream candidate fits with the current upstream fits,
not a mixture of legacy and current reference versions. Proceed to a same-event
before/after comparison of the downstream bar-average time minus the upstream
bar-average time. Check residual dependence on charge, segment stability and
fit-sample dependence before interpreting its width as detector performance.
The shared MCP/clock term cancels algebraically from the difference of the two
bar averages; calibration uncertainty and detector correlations do not thereby
vanish. A single-bar resolution needs additional independence/equal-resolution
and negligible flight-time-spread assumptions, or a known reference resolution.

No slides, original fits or event caches were overwritten. The new JSON files
are candidate coefficients/covariances; CSV files contain means, SEM and counts.
`common_supported_event_ids.csv` identifies the common candidate sample.
`input_manifest.json` records hashes of the cached input, original fit products
and directly used analysis scripts.

Reproduce from the workspace root with the existing analysis environment:

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python3 Q_Walk/results/2026-10-22_bar_pair_reference_check/check.py
```
