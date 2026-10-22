# N2-5 closure: model-shape closure is not event timing resolution

Closed on 22 October 2026 with the author's approval for this version, as a
documentation-only clarification. No new event analysis or resolution claim.

The four Note2 comparison plots (channels 137, 154, 159 and 242) use the saved
linear-slope Q method and clock-corrected MCP reference. Their lower panel is

```text
1000 * [AWC / S_4076(Q) + offset - W_MCP,4399(Q)]  [ps].
```

The reported RMS is the root mean square of this model difference on the
uniform-Q matching grid, with unsupported intervals removed. It describes
shape mismatch. It is not the width of an event-level time-difference
distribution, a single-channel resolution, or a measured improvement after
event-by-event correction. The two runs contain different events.

All four lower-axis labels now read "Model difference [ps]". Captions state
the interpretation explicitly; the channel-137 caption also names the RMS
shape mismatch and its Q weighting. The N2-5 inline review comment and queue
entry are removed, while the physical limitation remains in permanent text.

The existing curve data, fitted AWC/offset, bootstrap bands, N2-4 uncertainty
audit and conclusion text are unchanged. Only the four comparison figures
are re-exported, to change their axis labels. The underlying implementation
in `draft_support/build_assets.py::comparison` still subtracts the saved
`mapped_ns` and `reference_ns` columns without another residual fit.

Verification uses the existing clock-quality numerical/snapshot checks and
the publication-package verifier, which now requires the four explicit axis
labels and captions and the absence of N2-5 from the review/candidate decks.
The affected slides are visually inspected after recompilation. Event-level
timing validation remains outside this closure and is not implied by it.
