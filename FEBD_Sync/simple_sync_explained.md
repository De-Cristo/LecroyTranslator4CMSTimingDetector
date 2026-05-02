# simple_sync.py — Explained

## Purpose

`simple_sync.py` synchronizes TOFHIR ROOT data with oscilloscope trigger/MCP measurements. It solves one problem: **which ROOT event corresponds to which oscilloscope trigger?**

The TOFHIR and oscilloscope record the same physical events but on independent clocks, so their timestamps are unrelated. This script matches them, then attaches the MCP peak measurements from the oscilloscope to each ROOT event.

## Pipeline Overview

```
ROOT file (TOFHIR)         Meta CSVs (scope)         Peaks CSVs (scope)
  ├─ channelID               ├─ trigger_time            ├─ peak_time_ps
  ├─ channelIdx               │  (seconds)               ├─ peak_amp
  ├─ time (ps)                                           ├─ t0_abs_ps
  └─ t1coarse                                            └─ trigger_time_ps
         │                          │                          │
         ▼                          ▼                          │
   ┌──────────┐             ┌──────────────┐                   │
   │ Extract  │             │ Read trigger │                   │
   │ ch192    │             │ values from  │                   │
   │ times    │             │ meta CSV     │                   │
   └────┬─────┘             └──────┬───────┘                   │
        │                          │                           │
        ▼                          ▼                           │
   ┌──────────────────────────────────────┐                    │
   │     Match events (per segment)       │                    │
   │  ch192_time ≈ slope × trig_time + c  │                    │
   │  → root_to_trigger mapping           │                    │
   └──────────────────┬───────────────────┘                    │
                      │                                        │
                      ▼                                        │
                ┌────────────────┐                             │
                │ Attach MCP     │◄────────────────────────────┘
                │ peak data via  │
                │ mapping        │
                └───────┬────────┘
                        │
                        ▼
              Output ROOT + CSV
              (original data + MCP tree)
```

## Step-by-Step

### Step 1: Extract ch192 Times from ROOT

Reads the ROOT file in chunks (`--step-size`, default 300k entries). For each event:

- Check if channel 192 (trigger channel) is present
- If `--require-trigger`: also require the trigger channel specifically
- Extract the ch192 time value using `channelIdx[192]` → `time[idx]`
- Also extract `t1coarse` for optional deduplication

**Output**: arrays of `(ch192_time, entry_index, t1coarse, n_channels)`.

### Step 2: Dedup (Optional, `--dedup`)

The TOFHIR sometimes double-counts trigger events (two consecutive events with nearly identical `t1coarse`). This step:

1. Sort by time
2. Find consecutive pairs where `|Δt1coarse| < threshold` (default 8)
3. Remove the duplicate (keep first)
4. Remove "orphan fakes" — events with only 1 channel and no twin

### Step 3: Cluster by Time Gaps

ROOT events from multiple oscilloscope acquisitions (segments) are concatenated in one file. To separate segments:

1. Sort all ch192 times
2. Compute consecutive time differences
3. If a gap exceeds `median(dt) × gap_factor`, split there

**Output**: clusters of `(times[], entry_indices[])`, one per segment.

### Step 4: Pair Clusters with Meta Files

Each oscilloscope acquisition produces a meta CSV containing `trigger_time` values (in seconds). Meta files are resolved from `--meta-dir` by glob pattern. Clusters and meta files are paired by sorted order (cluster 1 → meta file 1, etc.).

### Step 5: Match Events (Per Segment) — The Core Algorithm

This is the key step. For each segment:

#### The Problem

- **ROOT ch192 times**: in TOFHIR picoseconds (absolute, e.g., `144,540,807,848,615 ps`)
- **Trigger times**: in scope seconds from acquisition start (e.g., `0.000, 0.211, 0.423, ...`)
- These two clocks have different rates and offsets

#### The Model

The two clocks are linearly related:

```
ch192_time = slope × trigger_time + offset
```

where:
- `slope` ≈ 0.997–1.003 (clock rate ratio, very close to 1)
- `offset` = large constant (absolute time difference)

#### The Algorithm

1. **Convert trigger times to ps**: `trigger_ps = trigger_seconds × 1e12`

2. **Work in relative coordinates** (subtract first value from both sequences for numerical stability):
   ```
   T_root = ch192_times - ch192_times[0]    → starts at 0
   T_trig = trigger_ps - trigger_ps[0]      → starts at 0
   ```

3. **Initial slope estimate** from total time spans:
   ```
   slope = T_root[-1] / T_trig[-1]
   ```
   Both sequences cover the same physical time, so the span ratio ≈ clock ratio.

4. **Initial offset = 0** (both start at 0 in relative space).

5. **For each trigger event**, compute expected ROOT time and find nearest:
   ```
   expected[j] = slope × T_trig[j] + offset
   nearest_root[j] = searchsorted(T_root, expected[j])  → O(log n)
   ```
   (Check both searchsorted result and its left neighbor for true nearest.)

6. **Compute residuals** and reject outliers:
   ```
   residual[j] = T_root[nearest[j]] - expected[j]
   σ = 1.4826 × MAD(residuals)    # robust σ estimate
   Accept if |residual - median| < 5σ
   ```

7. **Refit slope + offset** using only accepted matches via `polyfit`, then repeat steps 5–7 (3 iterations total).

8. **Resolve conflicts**: if multiple triggers map to the same ROOT event, keep the one with smallest residual.

#### Why Relative Coordinates?

Absolute values are ~1.45×10¹⁴ ps. When computing residuals:
```
145,068,330,739,599 − 145,068,330,739,500 = 99 ps
```
Float64 has ~15.9 decimal digits precision. At 10¹⁴, only ~1 digit remains for the ~100 ps residual. This causes the `polyfit` refinement to produce garbage coefficients.

In relative space, values are ~5×10¹¹, leaving ~4 digits of precision for residuals — sufficient for the matching.

#### Why searchsorted Instead of Index Alignment?

Extra ROOT events (ch192 fires but no corresponding scope trigger) shift all subsequent indices in cumulative-time matching. With searchsorted in time space, each trigger independently finds its nearest ROOT event — extra events between triggers don't affect nearby matches.

### Step 6: Attach MCP Peaks

For each matched pair `(ROOT event i, trigger event j)`:

1. Look up `segment_number = j + 1` (peaks CSV uses 1-based numbering)
2. Read from peaks CSV: `peak_time_ps`, `peak_amp`, `peak_sigma_ps`, `t0_abs_ps`, `trigger_time_ps`, `trigger_offset_ps`, `prev_rising_edge_abs_ps`
3. Compute phase values: `φ = (peak_time - t0) mod 6250 ps` (clock period)

Unmatched ROOT events get `NaN` for all MCP fields.

### Step 7: Write Output

1. **Copy input ROOT → output ROOT**, then add an `MCP` tree with branches:
   `index`, `peak_time`, `peak_amp`, `peak_sigma_ps`, `phi_peak`, `phi_peak_from_edge`, `trigger_time`, `trigger_offset_ps`, `phi_trigger`, `phi_trigger_from_edge`, `t0_abs_ps`

2. **Write matched CSV** with per-event details (entry, peak, trigger, segment, etc.)

### Step 8: Diagnostic Plots

- **Validation plot** (`_validation.png`): ch192 vs trigger scatter + fit residual histogram
- **dt ratio plot** (`_dt_ratio.png`): Δt_ch192/Δt_trigger for consecutive matched events (should be peaked at ~1.0)
- **Per-segment diagnostics** (`_segN_diag.png`): dt comparison, cumulative alignment residual, ROOT dt histogram

## CLI Usage

```bash
python3 simple_sync.py \
  --root /path/to/input.root \
  --peaks-dir /path/to/peaks_csvs/ \
  --meta-dir /path/to/meta_csvs/ \
  --out-root /path/to/output.root \
  --out-matched-csv /path/to/output.csv \
  --fast --require-trigger --dedup
```

Key options:
| Flag | Default | Description |
|------|---------|-------------|
| `--channel` | 192 | TOFHIR trigger channel |
| `--gap-factor` | 500 | Multiplier of median dt to detect segment boundaries |
| `--match-tol` | 5.0 | Outlier rejection threshold in σ (MAD-based) |
| `--dedup` | off | Enable t1coarse-based deduplication |
| `--step-size` | 300000 | ROOT read chunk size |
