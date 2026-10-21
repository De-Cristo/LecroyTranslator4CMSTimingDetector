# N1-5 validation and modification recommendation

22 October 2026. Status: investigation completed; N1-5 should remain open.
Production code, calibration outputs, ROOT inputs and note PDFs are unchanged.

## Main conclusion

Do not simply approve the historical 26 ps bar-resolution and 46.8 ps bar-MCP
claims. The current reproducible run4399 sample supports an end-difference core
near 55 ps and a bar-minus-MCP core near 64 ps, with the latter affected by
segment offsets. The current reference processing also contains independently
identified clock/matching failures. A small Gaussian-core width alone does not
validate the complete timing sample.

## Quantitative results

All 60 ROOT files were read with the existing reconstruction. There are 12,481
matched bar-8 events, 12,175 with both-channel quality, 10,992 passing both
peak-left charge cuts, and 10,577 in both saved measured calibration supports.
The ch137/ch150 profile occupancies exactly reproduce the saved inputs. The
maximum profile-mean differences are 0.0000419 and 0.0000233 ps respectively,
consistent with small numeric serialization/precision differences.

| Observable, same 10,577 events | Before walk correction | After walk correction | After: segment-block bootstrap 16th-84th percentile |
| --- | ---: | ---: | ---: |
| End difference D, Gaussian-core sigma | 55.08 ps | 54.88 ps | 54.42-55.31 ps |
| Bar minus mapped MCP R, Gaussian-core sigma | 77.67 ps | 64.20 ps | 63.16-65.07 ps |

Both before/after versions already use the same reconstructed clock map.
The two saved channel corrections are applied independently. The D correction
is small because common charge-dependent timing shifts largely cancel in the
end difference for this population; this is not evidence that walk correction
is ineffective for the bar-average time.

The D full-sample standard deviation is 61.16 ps. The original R full-sample
standard deviation is approximately 1.25e11 ps (0.125 s), demonstrating gross
reference-reconstruction failures. There are 395 selected events with |R| >
1 microsecond. Those events were not removed from the baseline validation.

The Gaussian windows span +/-2.5 scaled MAD around the median, with a
truncated-normal likelihood. The bootstrap has 400 successful replicas for
each observable and each of the event/block schemes. The intervals condition
on fixed calibration functions; they are not total measurement uncertainties.

## Robustness checks

- Changing the Gaussian-core half-window from 2 to 4 scaled MAD gives
  D widths 54.28-55.48 ps and R widths 63.92-64.34 ps.
- Held-out walk fitting gives D = 54.93 ps and R = 64.26 ps, versus 54.88 and
  64.19 ps with frozen curves on the same 10,571 eligible events. Thus no large
  walk-fit optimism is visible in this test.
- Holding out alternate clock-trigger events gives R = 63.69 ps, compared
  with 63.61 ps on the same 10,124 events using the full-data clock map. This
  tests local interpolation, not independent acquisition-segment prediction.
- The central 50% of charge asymmetry gives D = 54.68 ps, versus 54.88 ps for
  the full selected sample. This is an internal sensitivity check, not an
  independent tracking measurement of the illuminated position range.
- Subtracting each file/segment's own median reduces the diagnostic R core to
  54.30 ps. This is an extra data-derived correction and is not used in the
  headline result. It shows that segment offsets contribute appreciably.

## Confirmed clock problems

### Missed resets across nonfinite timestamps

`Q_Walk/qwalk_t0_data.py:assign_scope_segments` differences adjacent trigger
timestamps directly. A NaN between the last timestamp of one acquisition and
the first timestamp of the next hides the reset: neither adjacent difference
compares the two finite endpoints, so both receive the same segment label.

This happens in `4399_56_e` between entries 23331 and 64556 (one intervening
NaN), and in `4399_5_e` between entries 84064 and 129138 (98 intervening NaNs).
The original sparse bar-8 clock fits have unphysical slopes -1.945 and -1.521.
Explicit evidence is saved in `missed_reset_evidence.json`. The local test
segments finite timestamps before mapping labels back; production is untouched.

### Likely one-event association offsets

For file 26 segment 0 and files 38, 41, 44, 52 segment 2, the existing mapped
scope trigger and TOFHIR trigger disagree by far more than clock drift.
Using the next MCP row's scope timestamp in a diagnostic comparison reduces
the robust clock-line residual scale dramatically:

| File / segment | Original clock scale | Next-MCP-row diagnostic scale |
| --- | ---: | ---: |
| 26 / 0 | 1.61e8 ps | 102.90 ps |
| 38 / 2 | 1.35e8 ps | 265.99 ps |
| 41 / 2 | 1.40e8 ps | 99.06 ps |
| 44 / 2 | 1.38e8 ps | 145.85 ps |
| 52 / 2 | 1.49e8 ps | 68.38 ps |

These figures use all available matched trigger rows in the affected groups,
not just the bar-8 coincidence sample. This is strong evidence for an association
offset, but does not authorize blindly shifting timestamps: MCP peak, trigger,
waveform identity and TOFHIR entry must be rematched consistently upstream.
The diagnostic does not determine where in the input-building chain the offset
was introduced. An additional file-5 group remains invalid after reset repair.

### Local clock-quality diagnostic

After the finite-trigger reset treatment, reject whole groups using their
clock-line residual scale, independently of SiPM timing. Six groups fail.
Thresholds of 1 ns, 10 ns, 100 ns and 1 microsecond select exactly the same
10,282 events: retained group scales are approximately 36-618 ps, while failed
ones are orders of magnitude larger.

On this diagnostic subset, with unchanged saved walk curves:

- D core sigma = 54.86 ps; full-sample standard deviation = 61.26 ps.
- R core sigma = 64.18 ps; full-sample standard deviation = 70.31 ps.
- R outside the core-fit window = 1.56%.

The R medians of the 174 retained file/segment groups span approximately
-74 to +71 ps, with 16th-84th percentiles -37 to +40 ps. This confirms the
need to address segment offsets before extracting a synchronization term.
The diagnostic selection is not yet adopted as the production selection.

## Recommended changes to the note

1. Keep N1-5 open. Move the unidentified historical 52.87/46.80 ps images to
   backup, or explicitly mark them as historical illustrations whose original
   event selection and fit definition have not been recovered.
2. Add a clock-quality validation step before the resolution figures. Require
   finite-timestamp reset handling and verified trigger association; reject
   invalid clock groups with a documented clock-only rule. Do not repair this
   by clipping SiPM timing until the desired resolution is obtained.
3. Rerun the reference calibration and timing validation after that production
   decision. Check the effect on Note1's reference curves and Note2's AWC
   comparison before freezing their common reference version again.
4. Replace the main timing panels with a reproducible same-event before/after
   comparison, showing Gaussian-core sigma, fit-window definition, statistical
   interval, full-sample spread, event count and tail fraction. Show segment
   stability alongside it or in backup.
5. Describe half the end-difference width as a conditional estimate, not a
   demonstrated intrinsic bar resolution. At fixed position, for end timing
   errors e_L and e_R, Var((e_L+e_R)/2) = Var(e_L-e_R)/4 + Cov(e_L,e_R).
   Position broadening and shared timing fluctuations invalidate a naive
   identification. The measured r_i also share the MCP/reference error, so
   their observed covariance does not isolate detector covariance.
6. Do not extract a synchronization jitter by subtracting these separately
   fitted Gaussian-core widths in quadrature. Their tails, fit windows,
   common-mode covariance and segment offsets have not been brought under a
   single variance model.

Suggested interim review-slide wording:

> In a reproducible run4399 bar-8 sample, channel-wise walk correction reduces
> the bar-minus-MCP Gaussian-core width from 77.7 to 64.2 ps. The corrected
> end-difference core is 54.9 ps. These are conditional diagnostic widths:
> clock-reset and event-association failures require treatment before a final
> timing-resolution or synchronization-jitter claim.

The numerical diagnostics do not establish that the historical 46.8 ps result
was wrong; its original sample and processing are unknown. They establish that
it is not reproduced or justified by the current documented full-run pipeline.
