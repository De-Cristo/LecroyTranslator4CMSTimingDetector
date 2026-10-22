# N2-2: frozen reconstruction and selection audit

Closed on 22 October 2026 after replaying all 22 run-4076 channel profiles from
the original ROOT input. This closes the implementation/provenance question,
not the selection-optimization, model-validity or uncertainty questions in
N2-3, N2-4 and N2-6. No fit, bootstrap replica, figure data or AWC was replaced.

## Input and scope

The input is
`/eos/cms/store/group/dpg_mtd/comm_mtd/TB/MTDTB_H8_Sep2025/TOFHIR/reco/4076/23_e.root`:
230113 events and 1589882 flattened hits. The frozen targets are the 22
`*_profile_linear_summary.json` files under
`Q_Walk/results/2026-10-22_clock_quality/inputs/channel_*/`, their
`run4076_profile.csv` tables and coefficient NPZs. This includes channel 238,
whose slope profile exists even though it has no measured overlap with its
run-4399 reference and therefore no AWC fit in the current 21-channel summary.

`audit.py` replays selections, bin statistics and linear fits without producing
plots or overwriting the input artifacts. `audit.json` records checks, exact
input/code hashes, per-channel counts, threshold codes and timing ranges.
`channel_counts.csv` is a compact export of that report.

## Timestamp and current definitions

`Q_Walk/qwalk_core.py` implements:

```text
dc = (int64(t2coarse) - int64(t1coarse)) modulo 1024
t2 = time + float(dc)*6250 ps - f1 + f2
S  = 1000 * delta_I_uA / (t2 - time)_ps
```

`time` is the reconstructed first-threshold timestamp. The signed integer and
floating-point operations avoid the old uint16 coarse-time multiplication
overflow. This is the explicit `time2_mode="safe"`, not the legacy notebook
compatibility mode. The slope uses the difference of the reconstructed float
timestamps, exactly as in the saved analysis.

The fine-time lookup is built from **all hits in this file**: the mean of
`time modulo 6250 ps` for each `(channelID, tacID, t1fine)` key. The same table
is then queried at `t1fine` and `t2fine` to supply `f1` and `f2`. This is the
implemented internal lookup, not an independent second-threshold calibration.
The reader has a zero fallback for missing keys, but there are **zero missing
keys in either the prompt or the selected samples of all 22 audited channels**.

All prompt/selected hits have `step2=113621`. The decoder is:

```text
v1 = floor(step2 / 10000) - 1                       = 10
v2 = floor((step2 - 10000*(v1+1)) / 100) - 1        = 35
delta_I = (v2-v1) * 0.3125 uA                      = 7.8125 uA
```

This verifies the register decoding and implemented current scale, not an
independent measurement of the hardware threshold currents. Q is the ROOT
`energy` branch in ADC units, not calibrated deposited energy in MeV.

## Frozen event and hit selection

1. The target module must contain exactly two hits, one at each end of the
   specified bar. Duplicate hits and a third hit in that module fail. Module
   membership and bar partners follow `qwalk_channels.py`.
2. Hits in the other module and channel 192 are ignored. Neither their presence
   nor their absence is required. This is not an exactly-two-hits-total cut.
3. After the pair selection, apply the quality and charge cuts to the target
   channel's hit. The partner must be present but need not independently pass
   the target's timing or charge cuts.
4. Require finite Q and S, positive delta-I, positive reconstructed delta-t,
   and `dc == 0`. No DBSCAN, slope-MAD cut or fixed slope window is used here.
   The same-coarse-bin requirement defines this prompt population; genuine
   rises straddling a coarse-clock boundary are outside this selection.
5. Find the prompt-charge peak with 80 histogram bins. The range is the
   0.5th to 99.5th percentile, extended on each side by 5% of
   `max(p99.5-p0.5,20 ADC)`. Smooth counts using `[1,4,6,4,1]/16`.
   Starting from the maximum, follow adjacent bins at or above half height.
   Q_left is the left edge of that connected region. Require `Q >= Q_left`.
6. The Landau-Gaussian fit is a spectrum description; it does **not** define
   Q_left. The search/fit ranges are not additional cuts on the selected data.
   There is no extra upper-Q cut and no train/validation split.

## Profile and bootstrap

Use ten equal-width bins spanning the minimum to maximum selected Q; include
the maximum in the last bin. The point coordinates are mean Q and mean S.
Hit SD uses `ddof=1`; SEM is SD/sqrt(N). A singleton has stored SD=0 and SEM=NaN.
The linear fit uses weights `1/SEM^2` and omits nonfinite or negligible SEM
(`SEM <= sqrt(machine epsilon)*max(1,max(abs(mean S)))`). Channel 135 has nine
fitted bins because its final bin is a singleton. All other channels have ten.
An independent least-squares solve reproduces the saved coefficients.

The saved bootstrap contains 1000 coefficient pairs per channel, with seed
4076. Hits are resampled with replacement, using fixed Q-bin edges, and the
profiles and SEM-weighted fits are rebuilt. The audit checks NPZ counts,
finiteness, edges and stored 16th/50th/84th coefficient percentiles; it does
not regenerate all 22000 bootstrap fits. Pointwise bands use the saved paired
coefficients, preserving their correlation.

## Newer defaults are not the frozen selection

Current `qwalk_two_hit_pair_profile.py` adds a default minimum rise of 100 ps.
The frozen study predates that cut and is reproduced with only delta-t > 0.
Selected hits below 100 ps number 2161 (ch130), 3845 (ch139), 1626 (ch151),
1 (ch159), 1 (ch236) and 41 (ch238); all other audited channels have none.
They remain in the frozen results. Their physical acceptability and influence
are not established by reproducing the selection: these remain N2-3/N2-6 work.
The current spectrum fitter also includes a uniform-background component absent
from the saved older Landau-Gaussian summaries. It is not rerun in this audit
and does not participate in the histogram-boundary or profile replay.

## Checks and reproduction

All 22 pair/prompt/selected counts, histogram boundaries, profiles and linear
coefficients/covariances match. Profile CSVs agree with JSON summaries, allowing
the expected singleton SEM NaN. Frozen input hashes are retained in `audit.json`.
The channel-137 sequence is 25979 pair hits, 20789 prompt hits and 19126
peak-right hits, with Q_left = 184.91628556995408 ADC.

Seventeen focused tests passed: timing wrap/overflow and mode dispatch,
threshold conversion, calibration lookup, existing selection checks, the
optional rise-time mask, and added exact-module-pair/singleton-bin checks.
One existing spectrum-fit test initially failed because NumPy 1.26 lacks
`np.trapezoid`; its test-only integrator now supports both NumPy APIs. No
production reconstruction or selection code was changed.

```bash
OPENBLAS_NUM_THREADS=1 python DPS_NOTES/review_2026-10-22_n2_2/audit.py
python -m pytest Q_Walk/tests/test_qwalk_core.py \
  Q_Walk/tests/test_qwalk_two_hit_pair_profile.py \
  DPS_NOTES/review_2026-10-22_n2_2/test_semantics.py -q
```

The ROOT replay requires EOS access and the usual NumPy/Pandas/SciPy/Awkward/
Uproot/Matplotlib environment. The check used a single process; the 64 MB input
fits comfortably in memory. `verify.py` checks the saved audit and rebuilt
notes and exports the compact counts table without another ROOT replay.
