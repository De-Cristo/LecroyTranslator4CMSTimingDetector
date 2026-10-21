# MCP-reference calibration: review draft v1

Open `slides.pdf` or `note_with_cover.pdf`. Edit `main.tex` and `metadata.tex`.
`figure_manifest.json` identifies the input to every figure; `data/` retains
copies of those inputs and the four-channel coefficient table.

This draft preserves the legacy four-channel reference functions. It does not
silently replace them with the newer reference used in Note2. See the numbered
open review items N1-4 through N1-6 before any public release.

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

The standalone source archive compiles with pdfLaTeX. Within the repository,
use the shared builder described in `../README_DRAFTS.md`. No CMS DP identifier
has been assigned in this draft.
