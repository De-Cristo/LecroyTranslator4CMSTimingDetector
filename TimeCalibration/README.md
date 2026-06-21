# Bar-Level Timing Analysis

Modular suite for LYSO bar timing calibration and resolution studies. This toolkit processes ROOT files to extract timing, energy, and phase information, performing calibrations (including time-walk) and calculating intrinsic timing resolutions.

## Quick Start

The main entry point is `bar_analysis_main.py`. It supports multi-threaded processing, vectorized data extraction, and dual-module coincidence analysis.

### Example Usage
```bash
python3 bar_analysis_main.py \
    "/path/to/data/4405_*_e.root" \
    --module up \
    --lyso-bar 8 \
    --energy-range 0 1000 \
    --workers 10 \
    --fast \
    --strict-bar-only \
    --mcp \
    --down-lyso-bar 3
```

## Key Features
- **Fast Mode**: Uses `uproot` and `awkward` array vectorization for high-speed data processing.
- **MCP Calibration**: Synchronizes detector phases against a Micro-Channel Plate (MCP) reference.
- **Time-Walk Correction**: Automatically fits and applies polynomial corrections for energy-dependent timing shifts.
- **Absolute Resolution**: Uses the three-point method (Up, Down, MCP) to calculate the intrinsic resolution of each component.
- **Two-Pass Analysis**: 
    1. **Pass 1**: Performs Landau fits on raw energy spectra to determine optimal cut thresholds.
    2. **Pass 2**: Applies thresholds and calibrations to produce final timing distributions.

## Additional Scripts

- `ch192_vs_trigger.py`: Side-branch calibration test comparing channel 192 timing to the MCP trigger time.
- `ch192_vs_trigger_lowess.py`: Extended version of the above using LOWESS residual smoothing, with binned walk-fit companion plots and CSV exports.
- `ch192_vs_trigger_savgol.py`, `ch192_vs_trigger_wiener*.py`: Alternative smoothing/denoising variants for the residual calibration.
- `walk_fit_from_csv.py`: Standalone tool to refit a time-walk correction from a saved `*_walk_fit_data.csv`.
- `timecalib_plots.py`, `timecalib_study.py`, `MCP_analysis.py`: Older calibration plotting and MCP-referenced study scripts.

## Documentation

- `ARCHITECTURE.md` — four-layer design of the modular bar-analysis suite.
- `TIMING_METHOD_NOTES.md` — detailed summary of the current bar-level timing method.
- `CH192_VS_TRIGGER_NOTES.md` — explanation of the `ch192_vs_trigger.py` workflow.
- `ENGINEERING_NOISE_CANCELLING_NOTES.md` — filtering options for low-frequency baseline drift.

## Files
- `bar_analysis_main.py`: Main CLI and orchestration logic.
- `bar_processing.py`: Data extraction and filtering functions.
- `bar_plotting.py`: Histogramming and fitting routines.
- `bar_helpers.py`: Shared utilities and mathematical functions.
- `channel_mapping.py`: Hardware-to-software channel definitions.
