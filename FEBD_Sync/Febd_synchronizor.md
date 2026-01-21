# FEBd Synchronization Quick Guide

This workflow uses `read_root_explore.py` to export a CSV with list-like fields
and then runs `Febd_synchronizor.py` to align ROOT times with scope trigger times.

## Step 1: Export CSV from ROOT

`Febd_synchronizor.py` expects a CSV with a `time` column that stores a list
as a string (e.g. `"[1.0, 2.0, ...]"`). The easiest way to get this is
`--dump-df` from `read_root_explore.py`.

Example:
```bash
python3 LecroyTranslator/read_root_explore.py path/to/file.root \
  --dump-df 4237_1_e.csv \
  --max-entries 100000
```

If your branch names differ, add explicit overrides:
```bash
python3 LecroyTranslator/read_root_explore.py path/to/file.root \
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
python3 LecroyTranslator/Febd_synchronizor.py \
  --csv-path 4237_1_e.csv \
  --scope-dir 4237_scope \
  --output-dir sync_outputs
```

Or point directly to the meta file:
```bash
python3 LecroyTranslator/Febd_synchronizor.py \
  --csv-path 4237_1_e.csv \
  --meta-path 4237_scope/raw_C1_0004237_0000001_6347_meta.csv \
  --output-dir sync_outputs
```

## Outputs

`sync_outputs/` will contain:
- PNG plots (histograms, scatter, fits, alignment diagnostics)
- `mapping_results.npz` with arrays and fit parameters
- `mapped_points.csv` with mapped trigger/root points
- `summary.json` with run metadata and fit values
