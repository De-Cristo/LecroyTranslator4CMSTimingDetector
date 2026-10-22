# N1-6 citation audit

Verified 22 October 2026. These references support context and general methods,
not the study's run-specific event selection, numerical widths or AWC values.

| Key | Verified reference | Primary evidence | Use in the notes |
| --- | --- | --- | --- |
| MTDTDR | CMS Collaboration, *A MIP Timing Detector for the CMS Phase-2 Upgrade*, CERN-LHCC-2019-003, CMS-TDR-020 (2019) | [CERN record](https://cds.cern.ch/record/2667167?ln=en), [report](https://cds.cern.ch/record/2667167/files/CMS-TDR-020.pdf?version=1) | Detector context, not evidence for the September 2025 setup or run conditions. |
| TOFHIR2 | E. Albuquerque et al., *TOFHIR2: The readout ASIC of the CMS Barrel MIP Timing Detector*, arXiv:2404.01208 (2024) | [Author manuscript record](https://arxiv.org/abs/2404.01208) | Readout architecture; does not establish the firmware, thresholds or calibration constants of these runs. |
| LOWESS | W. S. Cleveland, *Robust Locally Weighted Regression and Smoothing Scatterplots*, JASA 74(368), 829-836 (1979), DOI 10.1080/01621459.1979.10481038 | [Original article copy](https://sites.stat.washington.edu/courses/stat527/s13/readings/Cleveland_JASA_1979.pdf), [publisher DOI](https://doi.org/10.1080/01621459.1979.10481038), Crossref DOI metadata | Local weighted regression background. The exact robust iterations, bandwidth, interpolation and clock cuts are local implementation choices. |
| Bootstrap | B. Efron, *Bootstrap Methods: Another Look at the Jackknife*, Annals of Statistics 7(1), 1-26 (1979), DOI 10.1214/aos/1176344552 | [Original article copy](https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf), [publisher DOI](https://doi.org/10.1214/aos/1176344552), Crossref DOI metadata | General empirical resampling principle only. This is not a prescription or validation of the study's file/segment-group bootstrap or its coverage. |

Titles, author attribution, years, journal volumes and page ranges were checked.
The two statistics DOIs were independently resolved through Crossref's structured
metadata. TOFHIR2 is deliberately cited as the verified author preprint; no
unverified journal citation is supplied. The CDS search record and report identify
the TDR; direct CDS access also returned a bot-protection page during the audit.
The Cleveland publisher page returned HTTP 403, so the original article copy and
DOI metadata provide the accessible verification. No third-party PDF is copied
into the repository.

`draft_support/references.tex` is the shared bibliography source, copied into each
standalone note by the builder. All four entries are cited in both notes. The
builder rejects unresolved LaTeX citations after two passes. Local figure
manifests and versioned numerical snapshots remain the source for study results.

This audit does not validate the supplied template as an official CMS DP format,
assign a DP number, approve the physics or authorize submission.
