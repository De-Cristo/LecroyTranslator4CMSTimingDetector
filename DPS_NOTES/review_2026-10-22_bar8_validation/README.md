# Bar-8 timing validation for N1-5

This directory contains a new validation, not a reproduction of the unidentified
event samples behind `sipdif.jpg` and `bar-mcp.jpg`. The note PDFs and the frozen
production calibration functions are not modified. N1-5 remains open pending
review of the results and the proposed note wording.

## Outcome

Read `RECOMMENDATION.md` first and open `validation_review.pdf` (six pages).
All 60 files were processed: 12,481 matched bar-8 events, of which 10,577 pass
both-channel quality, peak-right and measured-support requirements. Both frozen
reference profiles were reproduced, including their retained-bin counts; the
largest profile-mean discrepancy is below 0.00005 ps.

The end-difference Gaussian core is 54.88 ps; the bar-minus-MCP core is 64.20 ps.
However, the original clock processing has pathological groups hidden by
profile clipping. `clock_audit.py` identifies missed resets across NaN trigger
values and tests an event-row-offset hypothesis. It changes only copies in
memory. No production clock code, ROOT file, calibration curve or note is edited.
The finite-segmentation plus clock-quality diagnostic retains 10,282 events,
with a 64.18 ps bar-minus-MCP core and 70.31 ps full-sample standard deviation.
These are conditional diagnostics, not a validation of the legacy 46.8 ps claim.

## Inputs and processing

- All 60 synchronized run4399 ROOT files under
  `/eos/user/l/lichengz/MTD/TB2025/sync_out_4399_example/`.
- Exact five-hit topology: channels 137, 150, 234, 243 and 192, with a unique
  matched MCP entry. Target bar 8 is read out by channels 137 and 150.
- Existing `Q_Walk.qwalk_t0_data` ingestion and segment-local robust line plus
  LOWESS reference reconstruction. No new timing convention is introduced.
- Exact clock-corrected ch137/ch150 quadratic snapshots used in Note1/Note2,
  from `2026-09-24_measured_q_overlap/inputs`. `input_manifest.json` preserves
  numerical inputs and source snapshots. `root_inputs.json` records input
  paths, sizes, modification times and event counts (not content hashes).
- Four extraction workers; numerical libraries use one BLAS thread each.
  See `resources.json` and `environment.json` for the execution environment.

Before interpreting the timing distributions, the script reconstructs both
20-bin profiles and checks counts and means against the saved profiles.

## Selection and observables

Both channels must have finite timing and finite positive Q, slope, threshold
current difference and crossing-time separation, matching the frozen reference
quality selection. Both must pass their saved peak-left Q thresholds. Both must
lie inside retained profile-bin support and the outer fitted profile means.
No cut on the validation end difference or bar-minus-MCP residual is applied.
The reference-profile timing clipping is used only when rebuilding or refitting
the calibration profiles, not to select the validation histograms.

For each channel, form `r_i = delta_t_ref_ps - 1000 W_i(Q_i)`, where the saved
W_i is in ns and `delta_t_ref_ps = t_i - expected_ch192 - (MCP_peak - MCP_trigger)`.
The observables are `D = r_137 - r_150` and `R = (r_137 + r_150)/2`.
The two corrections are applied separately before taking the average; a fit to
the average charge is not substituted. Arbitrary constants do not affect widths.

The pre-walk distributions use the same events and clock mapping. The raw-trigger
diagnostic instead uses measured channel-192 timing, keeping the same frozen walk
functions and event sample; it is not a separately refitted raw-trigger analysis.

## Widths and uncertainty

The headline Gaussian-core width uses the existing `gaussian_core_width` helper:
the window is the sample median +/- 2.5 times 1.4826 MAD, and a truncated-normal
likelihood includes the window normalization. It is not the standard deviation
of the truncated sample. The full-sample spread is also reported as the sample
standard deviation about the sample mean, with ddof=1; it is labelled full-sample
RMS in the plots. Fractions outside the fit window are reported separately.

Four hundred event-bootstrap replicas and four hundred file/segment-block
replicas re-estimate the window and width. The block bootstrap resamples entire
observed file/segment groups with replacement. Percentile endpoints are the 16th
and 84th percentiles of successful fits. These are conditional uncertainties:
the saved clock and walk calibrations are fixed, so these intervals do not
include full calibration, model or operating-condition systematics. Block
resampling measures sensitivity to the observed group mixture; it assumes groups
can serve as resampling units and is not an independent proof of that assumption.

Additional diagnostics vary the core half-window (2, 2.5, 3 and 4 scaled MAD),
select central charge populations without timing cuts, inspect charge-asymmetry
dependence, and inspect widths and medians by file/scope segment. Charge asymmetry
is an internal position-sensitive proxy, not an independent position measurement.
The group-recentered widths are diagnostics only; headline histograms are not
recentered separately by group.

Walk cross-fitting holds out one scope-segment ordinal across all files and
refits the 20-bin weighted quadratics on other ordinals. Clock cross-fitting
alternates trigger events within each segment, fitting line plus LOWESS on one
half and evaluating the other half only within training-trigger bounds. These
are separate checks: walk cross-fitting keeps the original clock map; clock
cross-fitting keeps the saved walk curves. They are not a fully independent
joint clock-and-walk validation.

## Reproduce

From the workspace root, with the packages in `environment.json` installed:

```bash
OPENBLAS_NUM_THREADS=1 python DPS_NOTES/review_2026-10-22_bar8_validation/validate.py extract
OPENBLAS_NUM_THREADS=1 python DPS_NOTES/review_2026-10-22_bar8_validation/validate.py analyze --bootstrap 400
```

The cached `matched_hits.csv.gz` avoids rereading ROOT when rerunning the
statistical checks. `selected_events.csv.gz` and `all_events_selection.csv.gz`
retain event identity and selection flags. `validation_report.json` is the
numerical record; `validation.pdf` contains the review plots. No coefficients
or thresholds are tuned to reproduce the historical 52.87/46.80 ps numbers.

Run `python DPS_NOTES/review_2026-10-22_bar8_validation/clock_audit.py` after
`analyze` to reproduce the separate clock investigation. The seven initially
suspect groups are selected by clock-line scaled MAD > 1 microsecond, not by
SiPM timing. In those files, segmentation is recalculated from finite MCP
trigger timestamps. Clock-quality thresholds from 1 ns through 1 microsecond
then select the same 10,282 validation events. This wide stability interval
reflects a large separation between ordinary clock residuals and failed groups;
it does not establish a final approved selection. Five one-row-shift diagnostics
identify likely association errors but no remapping is applied.
