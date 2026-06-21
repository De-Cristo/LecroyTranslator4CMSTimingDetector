# `ch192_vs_trigger.py` Analysis

This document provides a detailed explanation of the `ch192_vs_trigger.py` script, situated in the `sandbox_test/TimeCalibration/` directory, based on the codebase's documentation (`README.md`, `ARCHITECTURE.md`, `TIMING_METHOD_NOTES.md`) and the script's source code.

## 1. Overview and Purpose

The `ch192_vs_trigger.py` script is a **side-branch calibration test**. Its primary goal is to compare the time recorded by channel 192 (`ch192`, which corresponds to the trigger channel) with the reference `trigger_time` provided by the Micro-Channel Plate (MCP) in the testing setup.

It aims to evaluate the correlation between these two timing measurements and attempts to derive a **residual-based calibration** (often analogous to time-walk or linearity corrections) to refine the `ch192` timing precision, ultimately validating if applying this calibration improves the overall timing resolution.

## 2. Core Operational Workflow

The script operates in three major phases, leveraging multi-processing and fast array evaluation tools (`uproot` and `awkward` arrays) inherited from the project's base architecture (`bar_processing.py`, `bar_helpers.py`, `bar_plotting.py`).

### Phase 1: Data Extraction and MCP Mapping
1. **MCP Mapping**: For each input ROOT file, it first builds a map from the event index to the MCP's `trigger_time` and `peak_time`. It allows filtering MCP events using amplitude limits (`mcp_peak_amp`) and a robust internal time difference cut (`--mcp-internal-dt-cut`).
2. **Channel Data Vectorised Extraction**: It then parses the main data tree using an accelerated approach via `awkward` arrays in chunks. It requires the selected channel (`ch192` by default) to be present and extracts its time, while simultaneously gathering data for an optional "validation channel" (default `ch137`) for cross-calibration testing.

### Phase 2: Per-File Segmentation and Analysis
The script deals with potential discontinuities in data by clustering events into **segments**.

1. **Segment Detection**: It identifies structural restarts in the dataset by looking for significant negative (backward) jumps in the MCP `trigger_time`. Consecutive jumps denote a new data segment.
2. **Alignment**: Since timestamps can have arbitrary offsets, it aligns the extracted `ch192` time for each segment by subtracting its initial starting offset against the MCP `trigger_time`.
3. **Linear Regression & Robust Fits**: For each segment, it performs an iterative linear fit (excluding outliers > ±3 Mean Absolute Deviation) to model `trigger_time = m * ch192_aligned + b`. This captures the primary scaling relation between the timings.

### Phase 3: Residual Calibration and Validation
Using the linear fits, the script calculates and models the timing **residuals** (the variation left over after the linear fit).

1. **Residual Correction**: 
   - It calculates `Residual = trigger_time - (m * ch192_aligned + b)`. 
   - It computes the median of these residuals across multiple bins along the `trigger_time` axis. 
   - Utilizing linear interpolation over these binned medians, it builds a secondary correction function.
2. **Calibration Application**: 
   - The calibrated `ch192` time (`ch192_cal`) is reconstructed by adding the interpolated residual correction back into the channel time and un-scaling it by the slope `m`.
3. **Validation**:
   - The script uses the validation channel (e.g., `ch137`) to independently measure resolution improvement.
   - It considers the double-difference width proxy: `[ch_val - ch192] - [mcp_peak - mcp_trigger]`.
   - By fitting Gaussians to this final proxy distribution before and after substituting `ch192` with `ch192_cal`, it objectively quantifies any improvement in intrinsic timing resolution (e.g., measuring standard deviation $\sigma$ drop in picoseconds).

## 3. Key Outputs
As it analyzes the ROOT files, it exports various graphical diagnostics for every segment and the overall dataset prefixed generically (e.g., `ch192_vs_trig_*`):
- **Scatter Plots (`*_scatter.png`, `*_scatter_segments.png`)**: 2D visualizations of `ch192` vs. MCP `trigger_time` with the iterative linear fits overlaid.
- **Residual Profiles (`*_resid_vs_trig_seg*.png`)**: Shows the deviation distributions and overlapping interpolated binned median fit.
- **Histograms (`*_residuals_combined.png`, `*_residuals_combined_cal.png`)**: 1D histograms of the uncalibrated vs. calibrated residuals fit with Gaussians using `bar_plotting.plot_t_diff`.
- **Validation Comparison (`*_validation.png`)**: A split visual graph showing the Gaussian width of the validation double-difference before and after calibration.

## 4. Relationship to the Rest of the Framework
- **Not in the Main Pass**: The logic documented in `TIMING_METHOD_NOTES.md` and orchestrated in `bar_analysis_main.py` performs primarily time-walk correction based on *energy peaks*. The `ch192_vs_trigger.py` acts as an auxiliary study focusing strictly on fine-tuning the base reference trigger signal itself via residual timeline drifts.
- **Shared Utils**: It heavily imports analytical tools from `bar_helpers.py` (like the Gaussian function `gauss` and dataset finder) and plotting modules from `bar_plotting.py` (`plot_t_diff`).
