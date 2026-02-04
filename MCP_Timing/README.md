# MCP_Timing

Utilities for clock edge reconstruction and MCP peak finding from LeCroy-style waveform CSVs, plus a helper to combine MCP peaks with clock `t0`.

## Input Data Format

All scripts expect waveform CSVs in the “tidy” format:

- `Segment` (integer segment/event id)
- `Time_s` (sample time in seconds)
- `Voltage_V` (sample amplitude in volts)

If a matching metadata CSV exists, it should be named like `<base>_meta.csv` and include arrays `trigger_time` and `trigger_offset` (semicolon-separated inside brackets). These are used to compute absolute timing.

## Scripts

### `clock_study.py`

Finds clock edges per segment and optionally fits a template to extract a per-event `t0` and `Tclk`.

Common usage:

```bash
# Analyze a single CSV (fast debug)
python3 clock_study.py --input raw_C2_0004237_0000001_6347_data.csv --out-dir ./clock_out --method template

# Scan a directory of CSVs
python3 clock_study.py --dir ./trc_out --out-dir ./clock_out --method zero
```

Key options:

- `--method`: `zero` or `template`
- `--polarity`: `rising`, `falling`, `both`
- `--fixed-zero-line`: override zero-cross threshold
- `--template-min-corr`: reject edges with low xcorr
- `--drop-last-edge`: drop last N edges before linear fit
- `--min-edge-spacing-ns`: enforce spacing between edges
- `--high-jitter-threshold-ps`: save debug plots for high jitter events

Outputs (in `--out-dir`):

- Per-file CSVs like `clock_edges_zero_<base>.csv`, `clock_edges_template_precise_<base>.csv`, `clock_template_fit_<base>.csv`
- Summary CSVs like `clock_edges_zero_cross.csv`, `clock_edges_template_precise.csv`, `clock_template_fit_results.csv`
- Diagnostic plots in `plots/`

### `MCP_wave_reco.py`

Fits the largest MCP peak per segment using a Gaussian model and writes per-event peak parameters.

Common usage:

```bash
# Auto-detect *_data.csv in a directory
python3 MCP_wave_reco.py --dir ./trc_out --out-dir ./mcp_out

# Explicit file
python3 MCP_wave_reco.py --csv raw_C1_0004237_0000001_6349_data.csv --out-dir ./mcp_out --min-amp 0.01
```

Outputs:

- `peaks_<base>.csv`
- Optional diagnostic plots (disable with `--no-plots`)

### `combine_mcp_clock.py`

Runs `clock_study.py` (template mode) to get per-event `t0_abs_ns`, then runs MCP peak reconstruction and attaches `t0_abs_ns` to each segment.

Common usage:

```bash
python3 combine_mcp_clock.py \
  --mcp ./trc_out/raw_C1_0004237_0000001_6349_data.csv \
  --clock ./trc_out/raw_C2_0004237_0000001_6347_data.csv \
  --out-dir ./combined_out \
  --clock-plot-first 0
```

Output:

- `peaks_<mcp_base>_with_t0.csv` with `t0_abs_ns` and `t0_abs_ps` columns

## Notes

- All scripts use a non-interactive matplotlib backend and write plots to disk.
- Clock processing expects the clock channel waveform; MCP processing expects the MCP channel waveform.
