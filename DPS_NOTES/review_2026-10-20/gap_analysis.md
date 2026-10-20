# DPS note preparation: gap analysis

Review date: 2026-10-20. Scope: inspect the supplied template and the two
existing long-form notes, identify scientific and production gaps, and prepare
a writing plan. No physics figures or manuscripts were changed in this review.
No submission or publication action was performed.

## Inputs and template check

| Input | Finding |
| --- | --- |
| `../Note1.pdf` | 21 physical pages; MCP reference, synchronization, reference-bar walk calibration. |
| `../Note2.pdf` | 15 physical pages; dual-threshold slope mapping and AWC comparison/prediction. |
| `../cms_dp_note_template.zip` | Extracted without overwriting into `../cms_dp_note_template/`. |
| PDF page size | Both existing notes are US Letter (612 x 792 pt), although called A4 in discussion. The new template cover is actual A4. |
| `main.tex` | Three-slide, 16:9 Beamer scaffold. Title slide already uses CMS and the requested conveners' email; slide 2 reads the shared abstract. |
| `metadata.tex` | Shared title, abstract, contact, date and placeholder DP number. One independent copy is needed per note. |
| `cover.tex` | Publication cover only, not a replacement for the existing full-length note. Shares abstract with slide 2. |
| Plot label | `\DPPlot` adds `CMS Preliminary Phase 2` around a plot in the slide. It does not modify standalone figure files. Regenerated figures need their own embedded label; then use `\DPPlotEmbedded`. |
| Template status | README explicitly says this is a working adaptation, not a verified official CMS DP template. Treat the supplied user requirements as the working specification; confirm publication packaging and assigned numbers before submission. |
| Build | Slides compile locally to `../cms_dp_note_template/build/main.pdf` (3 pages). Cover build fails because the local TeX installation lacks `everysel.sty`, required by `ragged2e`. This is a build-environment gap, not evidence of invalid physics content. |

Both PDFs were text-extracted in full. Selected physics figures and equations
were also rendered and visually inspected; extraction corrupts some symbols,
so the extracted text alone must not be used to repair equations.

## Highest-priority scientific gaps

### Note 1: freeze the actual reference calibration

1. **Binning disagrees with the published plots.** Section 7 (printed page 17)
   specifies 12 inverse-Q bins. Figure 13 (printed page 18) visibly uses 26,
   20, 48 and 25 retained bins for ch133, ch154, ch137 and ch150. The archived
   profile CSV row counts confirm those numbers. Requested and retained bins
   must be distinguished and the executed commands recovered.

2. **Clipping and weighting need a provenance decision.** Section 7 describes
   per-bin MAD rejection and statistical weights. The current legacy
   `TimeCalibration/walk_fit_from_csv.py` applies its MAD mask to the scatter
   fit, builds the binned profile from the original finite input arrays, and
   calls unweighted `np.polyfit` for that profile. It displays SEM bars but
   does not use them as fit weights. This is not the same implementation as
   the newer 20-bin, SEM-weighted `Q_Walk/qwalk_extended_t0_reference.py`.
   The legacy file is evidence of a reproducibility gap; this review has not
   proved which historical code revision generated every published panel.

3. **Selection descriptions are not ready to reuse verbatim.** Section 4
   states 5-sigma alignment rejection and section 5.1 states 3-sigma; separate
   stages may explain this, but they need explicit definitions and commands.
   Section 5.2 gives no numerical amplitude window. Section 5.3 describes an
   MPV-minus-one to MPV-plus-three Landau-scale window, whereas the newer
   extended calibration uses a peak-left half-maximum threshold with no
   analogous upper bound. Resolve figure by figure, not by mixing versions.
   The MCP peak-minus-trigger cut measures pulse/trigger timing consistency;
   describing it as the particle transit time through the MCP is misleading.

4. **The timing equation needs a sign convention repair.** Visually checked
   Eq. 6.2 writes a plus sign in front of a quantity expressed as
   `t_MCP,peak - t_MCP,trigger`. For that definition, the residual is
   `t_SiPM - mapped_trigger - (t_peak - t_trigger)`. The current implementation
   in `Q_Walk/qwalk_t0_data.py::reconstruct_reference_delta_t` uses subtraction.
   An opposite definition of the correction could make a plus sign valid,
   but it must be explicitly defined. Clock-scale and LOWESS signs in
   Eq. 6.3 also need a common, testable notation.

5. **Intrinsic MCP resolution and effective reference resolution differ.**
   The approximately 7.5 ps MCP estimate comes from a scope-only comparison
   and an assumed/independently measured PHOTONIS contribution. The 53.7 ps
   term extracted in section 5.4 includes synchronization effects. Label it
   an effective reference contribution under the stated covariance model,
   not a new intrinsic MCP resolution. Audit correlations in the three-system
   variance subtraction before claiming individual intrinsic resolutions.

6. **Bar-resolution claims need their assumptions and fit metadata.** Figure
   14 shows a fitted width of 0.0529 ns, while its caption says 52 ps, and the
   other panel shows 46.8 ps. State whether widths are Gaussian fit sigma or
   sample RMS, selection, fit range, uncertainty and event count. Converting
   sigma(tL-tR)/2 to bar-average resolution assumes negligible end covariance
   and controlled propagation-position broadening. A common-mode contribution
   cancels in tL-tR and is not measured by that estimate.

7. **Clock claims are too strong without further evidence.** The
   single-edge/sqrt(N) expression assumes independent edge errors. Correlated
   waveform or baseline noise does not average down the same way. Neither a
   6000-event rolling mean nor a finite-sample maximum shift establishes a
   sub-picosecond absolute reference. The proposed causes of a baseline tail
   and attribution of drift to TOFHIR are hypotheses, not measurements.

8. **Presentation and citations need cleanup.** Figure `??`, a Figure 15
   reference pointing to Figure 14, missing units, spelling errors and
   private meeting links appear. The archived +26 ns display shift is encoded
   in `walk_fit_from_csv.py`; an arbitrary offset is permissible, but both
   notes and all coefficient tables must use a documented convention.

### Note 2: distinguish model validation from reference-free prediction

1. **The reference-free claim exceeds the demonstrated procedure.** The
   abstract, introduction and Table 2 suggest that internal slope-Q data
   determine the full correction. Section 3 and Figure 4 instead describe
   AWC being fitted to the MCP calibration. Separate three stages: internal
   run4076 S(Q) extraction, MCP-derived reference AWC using run4399, and
   prediction of AWC from internal features for future deployment. These
   runs contain different events. No event-level cross-run pairing exists.

2. **Cross-run transfer assumptions need explicit evidence.** Record gain,
   overvoltage, threshold/baseline calibration, temperature, irradiation state,
   firmware/reconstruction and charge scales for both runs. Run4076 supplies
   dual-threshold information; run4399 supplies MCP reference timing. Do not
   imply simultaneous measurement of both in the same event sample. Quantify
   transfer stability where the data allow it, and bound the claim otherwise.

3. **Figure 6 has a demonstrably wrong uncertainty caption.** Its points are
   per-channel AWC estimates, not mean slopes in Q bins. The repeated Figure 3
   SD description is incorrect here. For the current saved summaries, bars
   are 16th-84th bootstrap percentiles from refitting S(Q) and AWC with the
   t0 quadratic fixed. The exact provenance of the older PDF's replotted
   points must be confirmed before assigning those intervals to that figure.
   The caption also references Eq. 3.1 for c1, which is defined in Eq. 3.2.

4. **Figure 5 must be identified precisely.** The visible plot is a binned
   residual versus Q, labelled after walk correction and a linear fit. It is
   not an event-resolution histogram. Recover the input points, uncertainties,
   correction and role of the additional line. A flat residual after fitting
   to those same points is closure on the calibration sample, not independent
   timing validation. Do not convert this plot directly into a 30 ps timing
   performance claim.

5. **The quantitative prediction claim is not supported by the latest test.**
   The text claims approximately 10% precision for most channels. In the
   saved 10-channel leave-one-bar-out tests the fractional RMS errors are
   17.22% (median Q*), 14.65% (histogram peak) and 21.85% (mean Q*). Absolute
   RMSE values are respectively 3.019, 3.310 and 2.972 uA. The simple 1/c1
   predictor gives 5.970 uA; its full-sample chi2/ndf is 607.73/8. These are
   different selections and evaluation procedures from the old Figure 6;
   they cannot silently replace or validate its 10% statement.

6. **Reference AWC uncertainty is incomplete.** Existing bootstrap bars omit
   reference-quadratic uncertainty and cross-run/model effects. The same
   slope parameters enter both predictor inputs and extracted reference AWC,
   inducing correlations. Propagate their joint uncertainty before assigning
   a prediction uncertainty or interpreting the cross-channel fit chi2.

7. **Channel eligibility must be defined independently of desired trends.**
   The extended scan has 22 channels; the strict overlap comparison completes
   21, with ch238 rejected for no overlap. The 10-channel display was chosen
   interactively. Note2's old 12-channel figure and its two excluded outliers
   are another selection. Publish an eligibility table with counts, common Q
   support, fit quality and reasons; show exclusions rather than hiding them.
   The particularly narrow support of ch235 and near-flat/negative slope
   channels require explicit treatment.

8. **Physical candidate is an optional new result, not established theory.**
   AWC = K S(Q*)^2/(c1 Q*) assumes similar local walk derivative across channels
   at a prescribed Q*. Define Q* once, make AWC constant per channel and
   distinguish hypothesis from identity. Median/mean use peak-right-selected
   events; the histogram maximum uses the quality sample before that cut.
   Peak Q* for ch133/ch135 is outside the outer profile means. Current tests
   use six held-out bar groups, fixed features and incomplete uncertainties;
   choosing Q* after comparing their scores is further model selection.

9. **Terminology needs tightening.** S = Delta I/Delta t12 is a finite-interval
   slope proxy, not automatically the derivative precisely at threshold 1.
   A slope table can be channel dependent without Q and S being poorly
   correlated within a channel. Q is the ADC charge proxy, not calibrated
   deposited energy. Cosmic-ray feasibility is a proposed application, not
   demonstrated with these pion-beam data. Avoid unsupported general claims
   about other experiments' triggers or calibration strategies.

## Figure inventory and required action

All retained physics plots require the exact `CMS Preliminary Phase 2` label
in their standalone exports, readable axes/units and self-contained captions.
Prefer regeneration from tables/arrays over relabelling screenshots.

| Note / figure | Current source or evidence | Action for slides |
| --- | --- | --- |
| N1 F1; N2 F1 setup | `../CMS-Note/cmsnote/setup.jpg` | Redraw a simplified setup identifying modules, MCP, scope and the two synchronization links. Reuse consistently. |
| N1 F2 waveform/MCP difference | `mcpwave.jpg`, `photonishpkdif.jpg` in the same directory | Recover waveform/fit and timing distribution data; regenerate two panels with conditions, counts and uncertainties. |
| N1 F3-F5 clock estimators | `Multi-clock-phases.png`, `t_ave_clock.png`, `clock-jitters.png`; `MCP_Timing/clock_study*.py` | Recover matching source run/arrays; combine into one illustrative method slide and one comparison slide if retained. |
| N1 F6-F7 baseline | `baseline_drift/baseline_drift_run4399_{rolling,hist}.pdf`; scripts in `baseline_drift/` | Optional backup. Reproduce from real inputs, not the dummy/test CSVs in that directory. Keep causal interpretations qualified. |
| N1 F8 event matching | `../CMS-Note/cmsnote/event_matching_validation.png`; `FEBD_Sync/` | Regenerate from matched-event output; caption must distinguish matching success from picosecond timing resolution. |
| N1 F9 pair resolutions | `U-D.png`, `U-P.png`, `D-P.png` | Recover histograms and fit metadata; label combined reference/synchronization contribution correctly. Optional if this clock branch is not central. |
| N1 F10-F12 trigger-clock correction | `spill.jpg`, `table.jpg`, `residuals.jpg`; `TimeCalibration/ch192_vs_trigger_lowess.py` and run4399 segment outputs | Regenerate with one chosen spill/segment definition and a clear linear-fit, residual, LOWESS, corrected sequence. Recreate ppm table as text, not a screenshot. |
| N1 F13 reference walk | Four archived profiles and coefficients in `Q_Walk/reference_quadratic_t0/`; legacy plotting code | Strongly reproducible at the profile level. Decide legacy versus current weighted reference first; regenerate all four panels and tables together. Split across slides if needed. |
| N1 F14 bar timing | `sipdif.jpg`, `bar-mcp.jpg` under the candidate Note1 source directory | Raw histogram provenance still unresolved. Reproduce same-event selections and fits before using the 26/46.8 ps claims. |
| N2 F2 Q spectra | `Q_Walk/results/2026-09-24_measured_q_overlap/energy_spectra/quality_selected_q.npz` plus CSVs | Replot run4076 spectra; mark cuts and any Q* definition, use Q [ADC]. Counts have been checked against saved selections in the earlier recovery. |
| N2 F3 S(Q) profiles | `Q_Walk/results/run4076_extended_profiles_v1/`, original four-channel CSVs in `reference_quadratic_t0/` | Replot from selected version; distinguish displayed SD, SEM fit weighting and bootstrap fit band. Remove any x10 uncertainty display scaling. |
| N2 F4 curve comparison | `Q_Walk/results/2026-09-24_measured_q_overlap/channel_*_comparison.csv`, input snapshots, `recompute.py` | Reproduce full-range display with shared-support shading and dashed gaps; explain that only supported overlap sets AWC and offset. No midpoint anchor in this version. |
| N2 F5 corrected residual | PDF panel inspected; exact input/replot provenance not located in this review | Reconstruct from reference profile points and chosen correction, or replace with explicitly defined model-closure residual. Clearly state whether an extra line was fitted/subtracted. |
| N2 F6 AWC summary | `channels.csv`, special-summary CSVs and `plot_special_channels.py` in the dated overlap directory | Regenerate a frozen channel set with correct x/y definitions, units, bootstrap meaning and exclusion policy. Show fit adequacy, not just a line. |
| N2 optional physical predictor | Three `physical_candidate_*_test/` directories and `try_physical_candidate.py` | Optional addition after target/feature uncertainty and validation review; label held-out predictions, not training points, as predictive evidence. |

## Source and production gaps

`../CMS-Note/cmsnote/cmsnote.tex` and associated assets are a candidate editable
source for Note1; they contain the same principal sections and figure names.
They must be version-matched to the supplied PDF, not assumed identical.
An editable Note2 source was not found in the directories inspected. Its text
and formulas can be reconstructed for a new deck, but the older figure data
must still be traced. No official DP numbers or release dates are provided.

The supplied template's caption column is small: four-panel figures from the
long notes will be unreadable there. Use one plot or one coherent two-panel
result per slide, with a companion slide for remaining channels. Define
abbreviations and distinguish measured data, fitted functions, prediction and
extrapolation on every public figure. Do not use a claim of CMS approval as
a substitute for the requested plot-status label.

This review does not verify external detector specifications, meeting results,
or bibliography records. Those need a primary-source citation audit during
drafting. The extracted text files are reading aids, not authoritative LaTeX.
