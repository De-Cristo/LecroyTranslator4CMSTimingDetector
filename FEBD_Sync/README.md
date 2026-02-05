# FEBD_Sync utilities

Tools for exporting ROOT data, synchronizing with scope triggers, adding MCP peaks, and validating timing.

**Prerequisites**
- Python 3
- `pip install uproot awkward numpy pandas matplotlib`

**Scripts**

**`read_root_explore.py`**
- Purpose: Inspect ROOT trees and export per-event lists (`channelID`, `channelIdx`, `time`, `energy`) to a CSV.
- Output format (dump-df): one row per event with JSON list strings.
- Example:
  ```bash
  python3 read_root_explore.py path/to/file.root \
    --dump-df 4237_1_e.csv \
    --max-entries 100000 \
    --branch-channel channelID \
    --branch-idx channelIdx \
    --branch-time time \
    --branch-energy energy
  ```

**`Febd_synchronizor.py`**
- Purpose: Synchronize ROOT event times to scope trigger times and produce mapping CSVs.
- Units: ROOT `time` is treated as ps; scope `trigger_time` is treated as seconds and converted to ps (`* 1e12`).
- Example:
  ```bash
  python3 Febd_synchronizor.py \
    --csv-path 4237_1_e.csv \
    --scope-dir 4237_scope \
    --output-dir sync_outputs
  ```

**`apply_mapping_add_peaks.py`**
- Purpose: Copy an input ROOT file and add a new `MCP` tree with `index`, `peak_time`, `peak_amp`, `phi_peak`.
- Input: mapping CSVs from `Febd_synchronizor.py` and peaks CSVs containing `segment`, `peak_time_ps`, `peak_amp`. Optional `t0_abs_ps` used to compute `phi_peak`.
- Units: `MCP/peak_time` is stored in ps.
- Example:
  ```bash
  python3 apply_mapping_add_peaks.py \
    --root path/to/in.root \
    --csv 4237_1_e.csv \
    --mapping "sync_outputs/mapping_results_*.csv" \
    --peaks-dir trc_out_MCP_reco \
    --peaks-pattern "peaks_raw_C1_*_data.csv" \
    --channel 192 \
    --out out_with_peaks.root
  ```

**`mcp_validation_dump.py`**
- Purpose: Dump ROOT event lists to a dump-df-compatible CSV and optionally append MCP info.
- `--out` matches `read_root_explore.py --dump-df` exactly.
- `--out-mcp` writes an additional CSV with MCP columns: `mcp_index`, `mcp_peak_time`, `mcp_peak_amp`, `mcp_peak_phase`.
- Filters events to those containing `--channel` (also supports `--require-channels`).
- Fitting uses the MCP CSV (`--out-mcp`) and the dumped channel times.
- Example:
  ```bash
  python3 mcp_validation_dump.py out_with_peaks.root \
    --out dump.csv \
    --out-mcp dump_mcp.csv \
    --channel 192 \
    --mcp-unit ps \
    --dump-unit ps \
    --channel-unit ps
  ```
- Fit example:
  ```bash
  python3 mcp_validation_dump.py out_with_peaks.root \
    --out dump.csv \
    --out-mcp dump_mcp.csv \
    --fit-from-csv \
    --fit-plot fit3.png \
    --fit-lines 3 \
    --fit-amp-cut 0.0 \
    --channel 192
  ```

**Notes**
- If you see a slope ~0.001 in fits, that indicates a 1000× unit mismatch. With ROOT time in ps and MCP peak time in ps, use `--mcp-unit ps --dump-unit ps --channel-unit ps`.
- `apply_mapping_add_peaks.py` requires `peak_time_ps` in peaks CSVs. It no longer falls back to `peak_time_ns`.
