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

## Files
- `bar_analysis_main.py`: Main CLI and orchestration logic.
- `bar_processing.py`: Data extraction and filtering functions.
- `bar_plotting.py`: Histogramming and fitting routines.
- `bar_helpers.py`: Shared utilities and mathematical functions.
- `channel_mapping.py`: Hardware-to-software channel definitions.
