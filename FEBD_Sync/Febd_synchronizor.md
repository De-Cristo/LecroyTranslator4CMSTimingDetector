# FEBd Synchronization Quick Guide

This workflow uses `read_root_explore.py` to export a CSV with list-like fields,
then runs `Febd_synchronizor.py` to align ROOT times with scope trigger times.
After that, `apply_mapping_add_peaks.py` injects MCP peaks into a new ROOT tree,
and `mcp_validation_dump.py` validates the result and fits time correlations.

## Step 1: Export CSV from ROOT

`Febd_synchronizor.py` expects a CSV with a `time` column that stores a list
as a string (e.g. `"[1.0, 2.0, ...]"`). The easiest way to get this is
`--dump-df` from `read_root_explore.py`.

Example:
```bash
python3 sandbox_test/FEBD_Sync/read_root_explore.py path/to/file.root \
  --dump-df 4237_1_e.csv \
  --max-entries 100000
```

If your branch names differ, add explicit overrides:
```bash
python3 sandbox_test/FEBD_Sync/read_root_explore.py path/to/file.root \
  --dump-df 4237_1_e.csv \
  --branch-channel channelID \
  --branch-idx channelIdx \
  --branch-time time \
  --branch-energy energy
```

## Step 2: Run the synchronization

`Febd_synchronizor.py` needs the CSV from step 1 and a scope meta CSV with
`trigger_time` values. By default it looks for:
`<scope-dir>/raw_C1_0004237_0000001_6347_meta.csv`.

Example:
```bash
python3 sandbox_test/FEBD_Sync/Febd_synchronizor.py \
  --csv-path 4237_1_e.csv \
  --scope-dir 4237_scope \
  --output-dir sync_outputs
```

Or point directly to the meta file:
```bash
python3 sandbox_test/FEBD_Sync/Febd_synchronizor.py \
  --csv-path 4237_1_e.csv \
  --meta-path 4237_scope/raw_C1_0004237_0000001_6347_meta.csv \
  --output-dir sync_outputs
```

## Step 3: Apply mapping and add MCP peaks to ROOT

`apply_mapping_add_peaks.py` copies the input ROOT, adds a new `MCP` tree, and
fills `index`, `peak_time`, `peak_amp` using mapping results + MCP peak CSVs.

Example (all segments):
```bash
python3 sandbox_test/FEBD_Sync/apply_mapping_add_peaks.py \
  --root path/to/in.root \
  --csv 4237_1_e.csv \
  --mapping "trc_out_sync/mapping_results_*.csv" \
  --peaks-dir trc_out_MCP_reco \
  --peaks-pattern "peaks_raw_C1_*_634*_data.csv" \
  --channel 192 \
  --out out_with_peaks.root
```

## Step 4: Dump validation CSV and fit correlations

`mcp_validation_dump.py` exports a CSV matching `--dump-df` format, filtered to
events containing channel 192, and appends MCP peak info.

Example dump:
```bash
python3 sandbox_test/FEBD_Sync/mcp_validation_dump.py out_with_peaks.root \
  --out dump.csv \
  --channel 192
```

Example fit + plot (3-line clustering on channel-time bands):
```bash
python3 sandbox_test/FEBD_Sync/mcp_validation_dump.py \
  --fit-from-csv dump.csv \
  --fit-plot fit3.png \
  --fit-lines 3 \
  --fit-amp-cut 0.5 \
  --channel 192 \
  --out dummy.csv
```

## Outputs

`sync_outputs/` will contain:
- PNG plots (histograms, scatter, fits, alignment diagnostics)
- `mapping_results.npz` with arrays and fit parameters
- `mapped_points.csv` with mapped trigger/root points
- `summary.json` with run metadata and fit values
