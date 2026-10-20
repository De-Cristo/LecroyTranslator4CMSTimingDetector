# MCP-reference calibration: review draft v1

Open `slides.pdf` or `note_with_cover.pdf`. Edit `main.tex` and `metadata.tex`.
`figure_manifest.json` identifies the input to every figure; `data/` retains
copies of those inputs and the four-channel coefficient table.

This draft preserves the legacy four-channel reference functions. It does not
silently replace them with the newer reference used in Note2. See the numbered
open review items N1-3 through N1-6 before any public release.

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

The standalone source archive compiles with pdfLaTeX. Within the repository,
use the shared builder described in `../README_DRAFTS.md`. No CMS DP identifier
has been assigned in this draft.
