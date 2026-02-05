# LecroyTranslator4CMSTimingDetector

This repository contains tools for processing and analyzing data from the CMS Timing Detector test beam experiments, focusing on LeCroy oscilloscope data, MCP (Microchannel Plate) timing, and synchronization.

## Overview

The analysis pipeline processes raw LeCroy `.trc` files, reconstructs MCP waveforms and clock edges, synchronizes data, applies mappings to ROOT files, validates results, and performs timing calibration studies.

## Prerequisites

- Python 3.8+
- Required Python packages: `uproot`, `awkward`, `numpy`, `matplotlib`, `scipy`
- ROOT (for reading/writing ROOT files)
- Access to CERN EOS storage for data files

## Directory Structure

- `FEBD_Sync/`: Tools for FEBD synchronization and peak reconstruction
- `MCP_Timing/`: MCP waveform reconstruction and clock analysis
- `TimeCalibration/`: Timing calibration plots and studies
- `TRC_Reader/`: LeCroy TRC file reader (compiled binary)
- `trc_out/`, `trc_out_MCP_clock_reco/`, etc.: Output directories for intermediate data

## Full Analysis Procedure

The complete analysis workflow is outlined below. Adjust paths and parameters as needed for your specific run.

### 1. Read LeCroy TRC Files

Use the compiled `read_lecroy` binary to convert raw `.trc` files to CSV format.

```bash
./TRC_Reader/read_lecroy /eos/cms/store/group/dpg_mtd/comm_mtd/TB/MTDTB_H8_Sep2025/LeCroy/raw/4237/raw_C2_0004237_0000001_6349.trc ../trc_out/
```

This generates CSV files in `../trc_out/` containing waveform data.

### 2. Combine MCP and Clock Data

Reconstruct MCP peaks and combine with clock edge timing.

```bash
python3 MCP_Timing/combine_mcp_clock.py \
  --mcp ../trc_out/raw_C1_0004237_0000001_6347_data.csv \
  --clock ../trc_out/raw_C2_0004237_0000001_6347_data.csv \
  --out-dir ../trc_out_MCP_clock_reco/ \
  --clock-polarity rising \
  --clock-min-edge-spacing-ns 3 \
  --clock-drop-last-edge 2 \
  --clock-plot-first 0 &
```

Outputs peak reconstruction results and diagnostic plots in `../trc_out_MCP_clock_reco/`.

### 3. Apply Mapping and Add Peaks to ROOT Files

Integrate the reconstructed peaks into the ROOT data tree.

```bash
python3 FEBD_Sync/apply_mapping_add_peaks.py \
  --root /eos/cms/store/group/dpg_mtd/comm_mtd/TB/MTDTB_H8_Sep2025/TOFHIR/reco/4237/1_e.root \
  --mapping ../trc_out_sync/mapping_results_*.csv \
  --peaks-dir ../trc_out_MCP_clock_reco \
  --peaks-pattern "peaks_raw_C1_*_data_with_t0.csv" \
  --channel 192 \
  --out out_with_peaks.root
```

Creates `out_with_peaks.root` with added MCP peak information.

### 4. Validate MCP Data

Dump and validate the MCP timing data, generate diagnostic plots.

```bash
python3 FEBD_Sync/mcp_validation_dump.py \
  out_with_peaks.root \
  --out dump.csv \
  --fit-from-csv \
  --channel 192 \
  --fit-plot fit3.png \
  --fit-lines 3 \
  --fit-amp-cut 0.0 \
  --out-mcp dump_mcp.csv \
  --workers 1 \
  --mcp-unit ps \
  --dump-unit ps \
  --channel-unit ps
```

Produces validation CSVs and plots.

### 5. Generate Time Calibration Plots

Create initial timing calibration plots.

```bash
python3 TimeCalibration/timecalib_plots.py \
  out_with_peaks.root \
  --channels 133 192 \
  --plot-channel 133 \
  --require-channel 192 \
  --dt-wrap 6250 \
  --out-dt-wrap-hist dt_wrapped.png \
  --out-dt-resid-hist dt_res.png \
  --energy-min 180 \
  --energy-max 300 \
  --verbose
```

Generates various timing plots and histograms.

### 6. Perform Detailed Timing Study

Run comprehensive timing analysis with module and bar specifications.

```bash
python3 TimeCalibration/timecalib_study.py \
  "*.root" \
  --module up \
  --lyso-bar 7 \
  --side both \
  --second-module down \
  --second-lyso-bar 7 \
  --second-side both \
  --workers 3 \
  --skip-ch192-plot
```

Produces detailed timing studies and additional plots.

## Notes

- Ensure all input paths exist and are accessible.
- Some steps may run in parallel (note the `&` in step 2).
- Adjust energy cuts, channel numbers, and other parameters based on your specific setup.
- Output files are generated in the current working directory unless specified otherwise.

## Authors

- Licheng Zhang (licheng.zhang@cern.ch), Feb-2026