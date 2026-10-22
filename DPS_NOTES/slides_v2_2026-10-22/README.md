# Short slide drafts: version 2, draft 1

Two notes, each with ten content slides and one compact reference slide,
with no A4 cover pages. These are separate from the full
notes, which remain unchanged. Each deck retains the CMS/contact title slide,
a purpose slide (Note 1) or short abstract (Note 2) on slide 2, labelled plots,
and concise procedural captions.
The titles follow the author's wording, including TOFHIR2C in Note 2.

- `note1_mcp_reference/slides.pdf`: synchronize MCP/FEBd records, construct
  the reference time, fit inverse-charge profiles, and check timing correction.
- `note2_dual_threshold/slides.pdf`: measure the slope proxy, select/profile
  hits, construct the inverse-slope correction, calibrate AWC against MCP,
  and summarize channel dependence and limitations.
- `both_notes_slides_v2.pdf`: both decks, 22 landscape pages, no covers.

Each final reference slide uses two columns and preserves the original
bibliography entries and numbering: eight entries in Note 1 and twelve in Note 2.
The build checks that these entries match the full notes exactly.

Existing figures and numerical macros are copied unchanged from the full notes.
The new threshold-separated figures are copied from the separate, saved
`Q_Walk/results/2026-10-22_bar_pair_by_threshold/` validation, with its inputs and
uncertainties documented there. The build does not rerun that analysis.
`asset_manifest.json` records each source and hash. No ROOT inputs are read
and no fits are rerun. The full notes retain the complete bibliographies,
selection details and uncertainty studies. No claim of intrinsic detector
resolution or demonstrated reference-free AWC calibration is added.

Rebuild from the workspace or repository root:

```bash
python3 DPS_NOTES/slides_v2_2026-10-22/build.py
```

Requires pdfLaTeX, Poppler, Pillow and pypdf. Previews are generated locally
under each note's `preview/` directory. The builder rejects layout overflow,
missing fonts, unresolved citations, unexpected page counts and portrait pages.
These are first discussion drafts, not approved CMS submissions.

## Note 1 structure review

The author's six-point structure comment is implemented in the short deck:
purpose; parallel MCP/scope stream; FEBd synchronization; correlated events
with low-frequency clock drift; corrected upstream reference bars; final
downstream-minus-reference-bar timing comparison. The setup slide explicitly
distinguishes this test-beam synchronization from the final HL-LHC scheme.
Threshold-separated bar-to-bar histograms and a width-versus-threshold summary
now end the narrative, immediately before references.
There are still ten content slides and one reference slide.

The former 8,737-event, 141.8 ps mixed-threshold result is superseded. Each of
six thresholds now has independent four-channel fits and same-event validation;
8,672 selected events are distributed across the six settings. Corrected widths
range from 62.1 to 69.4 ps. No combined event histogram or per-setting alignment
is used. The optional event-weighted RMS of fitted widths (65.7 ps after
correction) is saved as a descriptive summary, not an operating-point resolution.
Slide 8 now uses the lowest-threshold bar-8 profiles rather than mixed-threshold
bar-7/bar-8 examples; slides 9 and 10 show the six timing comparisons and their
width summary. Fixed-calibration intervals and in-sample limitations remain
explicit. Note 2 and the full original notes are unchanged in content.
