# MCP-reference calibration: review draft v1

Open `slides.pdf` or `note_with_cover.pdf`. Edit `main.tex` and `metadata.tex`.
`figure_manifest.json` identifies the input to every figure; `data/` retains
copies of those inputs and the four-channel coefficient table.

This draft preserves the legacy four-channel reference functions and adds the
four clock-corrected references used in Note2. Both versions are explicitly
identified. See open review items N1-5 and N1-6 before any public release.

N1-1 is resolved: the author confirmed validation of the MCP waveform and
intrinsic-reference timing results. The review questions and provenance
caveats were removed from slides 4 and 5 and the open review queue.
The figures and numerical results were not changed or independently revalidated
as part of this text revision.

N1-2 comments were replaced at the author's request with implementation paths
on slides 6 and 10: `FEBD_Sync/simple_sync.py`,
`TimeCalibration/ch192_vs_trigger_lowess.py`,
`TimeCalibration/ch192_vs_trigger.py`, and `TimeCalibration/bar_processing.py`.
Their matching, selection and optional-cut roles were checked in the current
code. This documentation edit does not establish the historical command-line
configuration used for the legacy figures.

N1-3 is closed at the author's request (21 October 2026). Slide 9 explicitly
uses `delta_t = t_SiPM - mapped_trigger - (t_peak - t_trigger)`, matching
`TimeCalibration/ch192_vs_trigger_lowess.py:816` and `:1102`. The expected
trigger includes the LOWESS residual once (line 661), with segment offsets
restored at lines 802-806; the validation subtracts that expected trigger once.
The separate corrected-channel diagnostic subtracts the residual once before
its unit-slope adjustment (lines 201-204). The N1-3 questions and queue entry
are removed, and slide 8 explains the expected-trigger construction.
This is a documentation correction based on code inspection; no analysis code,
fit, plotted data or historical acquisition configuration was changed or rerun.

N1-4 is closed at the author's request (21 October 2026). Slides 15-20 add the
clock-corrected definition and selection, four saved SEM-weighted profiles,
and the archived-versus-updated shape comparison. The new figures read the
exact `2026-09-24_measured_q_overlap/inputs/channel_*` snapshots used in Note2;
no ROOT processing or refit was performed. `data/clock_reference_coefficients.csv`
preserves a, b and c in the original ns convention, separately recording the
+26 ns display shift. The archived fits remain unchanged and are now labelled
as unweighted fits using measured channel-192 timing. Their reconstruction and
the comparison are documented in `../review_2026-10-21_reference_comparison/`.
The comparison summary CSV is also included under this note's data snapshots.
This closes the reference-version gap, not N1-5 timing-resolution validation or
N2-1 cross-run transfer conditions.

The standalone source archive compiles with pdfLaTeX. Within the repository,
use the shared builder described in `../README_DRAFTS.md`. No CMS DP identifier
has been assigned in this draft.
