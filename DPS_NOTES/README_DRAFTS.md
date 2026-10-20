# Two CMS performance-note review drafts

Prepared 20 October 2026. These are discussion drafts, not approved CMS results
or submitted notes. The original `Note1.pdf` and `Note2.pdf` are unchanged.

## Open the drafts

- `note1_mcp_reference/slides.pdf`: MCP-reference calibration slide note.
- `note2_dual_threshold/slides.pdf`: dual-threshold/inverse-slope slide note.
- Each directory also contains `cover.pdf` (A4), `note_with_cover.pdf`
  (cover followed by slides), and `source_bundle.zip` (editable self-contained
  LaTeX, figure PDFs, numerical inputs and manifest).
- `preview/` holds rendered slides and contact sheets.

The title slide uses CMS and `cms-dpg-conveners-mtd@cern.ch`. Slide 2 and the
A4 cover read the same abstract from `metadata.tex`. Every exported plot
contains `CMS Preliminary Phase 2`; the enclosing note remains conspicuously
marked as an internal review draft. No official DP number or approval is implied.

## How this first version handles the known gaps

Note1 retains the **legacy four-channel calibration**. Its profiles and
coefficients are redrawn from CSV without refitting. Existing waveform,
synchronization and resolution illustrations are retained as labelled legacy
rasters. Captions say when raw-input provenance or uncertainty is incomplete.

Note2 uses the **newer extended, measured-overlap study**, not the legacy Note1
curves. Profile fits, spectra, curve comparisons and AWC summaries are redrawn
from frozen numerical outputs. The old ambiguous residual illustration is
replaced with an explicitly defined model-minus-model residual. The internal
AWC predictor is exploratory backup material. The ten-channel subset is
explicitly identified as an interactive selection; the full 21 completed
fits are shown as well.

These choices make both drafts readable now without pretending that analysis
versions, selections and uncertainties have already been reconciled.

## Review sequence

1. N1-4 / N2-1: select one reference calibration version for both notes.
2. N2-2: freeze run configurations, time decoding and numerical cuts.
3. N1-3: settle clock mapping, time-difference signs and correction ordering.
4. N2-3 / N2-4: assess slope-model adequacy, matching objective and uncertainties.
5. N1-5 / N2-5: reproduce bar timing validation.
6. N2-6: define channel eligibility independently of the observed summary trend.
7. N2-7: decide whether internal prediction belongs in the main result.
8. N1-6: bibliography, official template, metadata and release packaging.

Before publication, synchronize the relevant local code and documentation with
the intended online repository. Verify that every code path cited in the slides
exists there and matches the version used for the reported results. Record the
repository URL and immutable commit or release identifier in the final notes.
Local figure/source snapshots alone do not establish that the online repository
is up to date. The author has now authorized committing and pushing the note
updates after each accepted review point to
`De-Cristo/LecroyTranslator4CMSTimingDetector`, using the local checkout
`/afs/cern.ch/work/l/lichengz/private/MTD/TB_2025/LecroyTranslator4CMSTimingDetector`.
See `REPOSITORY_WORKFLOW.md` for the standing synchronization workflow.

Review IDs appear beside the affected slides and in each closing review queue.
The longer evidence and writing plan remain in `review_2026-10-20/`.

Resolved: N1-1 was validated by the author. Its questions and provenance caveats
have been removed from Note1 slides 4 and 5 and its open review queue; the
underlying results are unchanged.

N1-2 comments were replaced with relevant code paths on Note1 slides 6 and 10.
The code locations are verified; this change does not independently recover
the historical runtime configuration. N1-2 is no longer in the slide review queue.

## Edit and rebuild

Edit each note's `main.tex` and `metadata.tex`. Shared layout and cover source:
`draft_support/slides_preamble.tex` and `draft_support/cover.tex`.
The build copies these shared files into each standalone note directory.
The cover avoids the unavailable `everysel` dependency using ordinary LaTeX
paragraph justification; the supplied template remains untouched.

From the repository root:

```bash
python3 DPS_NOTES/draft_support/build.py --assets
```

Omit `--assets` when only slide text or layout changed. Required tools:
Python with NumPy, Matplotlib, Pillow and pypdf; pdfLaTeX; Poppler `pdftoppm`.
The figure builder reads existing analysis outputs; it does not open ROOT files,
refit physics models, change selections or modify `Q_Walk`.

The build checks title/abstract placement, one-page covers, figure labels and
LaTeX overflow. Rendered previews still need visual inspection after layout
changes. `figure_manifest.json` records source paths, copies and SHA-256 hashes;
numerical sources are retained under `data/`.

The draft builder produces only files under `DPS_NOTES/`; repository
synchronization and the authorized commit/push are a separate step after each
accepted review point. This is not authorization to submit an official CMS note.
Source archives can compile on their own using
`pdflatex main.tex` twice and `pdflatex cover.tex`.
