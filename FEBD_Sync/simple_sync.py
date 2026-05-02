#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Mar-2026
"""
Simplified FEBD sync + MCP attach using cumulative-time matching.

Replaces the fragile dt-ratio-walking in fast_sync_add_mcp_newv_debug.py
with a robust cumulative-time approach:
  1. Extract ch192 times from ROOT
  2. Read trigger times from meta CSVs
  3. Segment by time gaps
  4. Match within each segment by cumulative time (linear fit + nearest match)
  5. Attach MCP peaks
  6. Write output ROOT with MCP tree + matched CSV
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.optimize import curve_fit


def log(msg):
    print(f"[simple-sync] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────
# Reused utilities (from fast_sync_add_mcp_newv_debug.py)
# ──────────────────────────────────────────────────────────────────

def find_data_tree(f):
    all_keys = list(f.keys())
    data_keys = [k for k in all_keys if k.startswith("data")]
    if data_keys:
        best, best_cycle = None, -1
        for k in data_keys:
            cycle = 0
            if ";" in k:
                try:
                    cycle = int(k.split(";", 1)[1])
                except Exception:
                    pass
            if cycle > best_cycle:
                best_cycle, best = cycle, k
        return best
    tnames = [k for k, v in f.items() if hasattr(v, "num_entries")]
    return tnames[0] if tnames else None


def split_by_time_gaps_with_indices(values, indices, gap_factor=100):
    vals = np.asarray(values)
    idxs = np.asarray(indices)
    order = np.argsort(vals)
    vals_sorted, idxs_sorted = vals[order], idxs[order]
    if len(vals_sorted) < 2:
        return [vals_sorted], [idxs_sorted]
    gaps = np.diff(vals_sorted)
    threshold = np.median(gaps) * gap_factor
    split_pts = np.where(gaps > threshold)[0]
    return np.split(vals_sorted, split_pts + 1), np.split(idxs_sorted, split_pts + 1)


def read_trigger_values(meta_path):
    trigger_values = []
    with open(meta_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "trigger_time" not in line.lower():
            continue
        match = re.search(r"\[(.*?)\]", line)
        if not match:
            continue
        parts = match.group(1).split(";")
        for p in parts:
            p = p.strip()
            if p:
                try:
                    trigger_values.append(float(p))
                except Exception:
                    continue
    return trigger_values


def resolve_meta_files(meta_path):
    if any(ch in meta_path for ch in "*?["):
        return sorted(glob.glob(meta_path))
    m = re.search(r"_(\d+)_meta\.csv$", meta_path)
    if m:
        pattern = meta_path[: m.start(1)] + "*" + meta_path[m.end(1) :]
        siblings = sorted(glob.glob(pattern))
        if siblings:
            return siblings
    return [meta_path] if os.path.exists(meta_path) else []


def find_peaks_file(peaks_dir, peaks_pattern, suffix):
    peaks_dir = Path(peaks_dir)
    if "{suffix}" in peaks_pattern:
        patt = peaks_pattern.format(suffix=suffix)
        matches = sorted(peaks_dir.glob(patt))
        return matches[0] if matches else None
    candidates = sorted(peaks_dir.glob(peaks_pattern))
    for c in candidates:
        if suffix in c.name:
            return c
    if len(candidates) == 1:
        return candidates[0]
    return None


def load_peak_lookup(peaks_file):
    df = pd.read_csv(peaks_file)
    required = ["segment", "peak_time_ps", "peak_amp"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f'Missing required column "{col}" in peaks file: {peaks_file}')
    t0_col = "t_ave_ps" if "t_ave_ps" in df.columns else ("t0_abs_ps" if "t0_abs_ps" in df.columns else None)
    trig_col = "trigger_time_ps" if "trigger_time_ps" in df.columns else None
    edge_col = "prev_rising_edge_abs_ps" if "prev_rising_edge_abs_ps" in df.columns else None

    seg_to_peak = {}
    for _, row in df.iterrows():
        try:
            seg = int(row["segment"])
        except Exception:
            continue

        def _f(col_name, default=np.nan):
            v = row.get(col_name, default)
            return float(v) if v == v else np.nan

        seg_to_peak[seg] = (
            _f("peak_time_ps"),
            _f("peak_amp"),
            _f("peak_sigma_ps"),
            _f(t0_col) if t0_col else np.nan,
            _f(trig_col) if trig_col else np.nan,
            _f("trigger_offset_ps"),
            _f(edge_col) if edge_col else np.nan,
        )
    return seg_to_peak


def compute_phi(a, b):
    if a != a or b != b:
        return np.nan
    try:
        return float((a - b) % 6250.0)
    except Exception:
        return np.nan


LEGACY_MATCHED_CSV_COLUMNS = [
    "entry",
    "channelID",
    "t1coarse",
    "time",
    "energy",
    "mcp_index",
    "mcp_peak_time",
    "mcp_peak_amp",
    "mcp_peak_sigma_ps",
    "mcp_peak_phase",
    "mcp_trigger_time",
    "mcp_trigger_offset_ps",
    "t0_abs_ps",
    "prev_edge_ps",
    "phi_peak_from_trigger",
    "peak_minus_t0_ps",
    "peak_minus_prev_edge_ps",
    "trigger_minus_t0_ps",
    "root_time_ps",
    "meta_file",
    "peaks_file",
    "segment",
    "trigger_ps_from_meta",
    "phi_peak_from_edge",
    "phi_trigger_from_edge",
]


def _finite(value):
    try:
        return value == value
    except Exception:
        return False


def _delta_or_nan(a, b):
    if not (_finite(a) and _finite(b)):
        return np.nan
    return float(a) - float(b)


def build_legacy_matched_dataframe(matched_rows, root_arrays=None):
    """Return the matched CSV in the legacy downstream contract.

    The current timing source is t_ave_ps.  For compatibility with existing
    consumers, expose it in the historical CSV column named t0_abs_ps.
    """
    root_arrays = root_arrays or {}
    normalized = []

    for row in matched_rows:
        out = dict(row)
        entry = out.get("entry")

        for col in ("channelID", "t1coarse", "time", "energy"):
            out[col] = root_arrays.get(col, {}).get(entry, out.get(col, "[]"))

        t0_abs_ps = out.get("t0_abs_ps", out.get("t_ave_ps", np.nan))
        prev_edge_ps = out.get("prev_edge_ps", np.nan)
        peak_time_ps = out.get("mcp_peak_time", np.nan)
        trigger_time_ps = out.get("mcp_trigger_time", np.nan)

        out["t0_abs_ps"] = t0_abs_ps
        out["prev_edge_ps"] = prev_edge_ps
        out["phi_peak_from_trigger"] = out.get(
            "phi_peak_from_trigger",
            compute_phi(peak_time_ps, trigger_time_ps),
        )
        out["peak_minus_t0_ps"] = out.get(
            "peak_minus_t0_ps",
            _delta_or_nan(peak_time_ps, t0_abs_ps),
        )
        out["peak_minus_prev_edge_ps"] = out.get(
            "peak_minus_prev_edge_ps",
            _delta_or_nan(peak_time_ps, prev_edge_ps),
        )
        out["trigger_minus_t0_ps"] = out.get(
            "trigger_minus_t0_ps",
            _delta_or_nan(trigger_time_ps, t0_abs_ps),
        )

        normalized.append({col: out.get(col, np.nan) for col in LEGACY_MATCHED_CSV_COLUMNS})

    df = pd.DataFrame(normalized, columns=LEGACY_MATCHED_CSV_COLUMNS)
    if len(df) > 0:
        df = df.sort_values(["entry", "segment"]).reset_index(drop=True)
    return df


def fetch_root_arrays_for_entries(root_path, tree_name, entries, args):
    """Fetch original TOFHIR list branches as JSON strings for matched CSV rows."""
    entries = sorted(set(int(e) for e in entries))
    dumps = {col: {} for col in ("channelID", "t1coarse", "time", "energy")}
    if not entries:
        return dumps

    with uproot.open(root_path) as f_in:
        tree_in = f_in[tree_name]
        keys_in = list(tree_in.keys())
        branch_map = {
            "channelID": args.branch_channel or ("channelID" if "channelID" in keys_in else None),
            "t1coarse": "t1coarse" if "t1coarse" in keys_in else None,
            "time": args.branch_time or ("time" if "time" in keys_in else None),
            "energy": args.branch_energy or ("energy" if "energy" in keys_in else None),
        }
        read_cols = [branch for branch in branch_map.values() if branch and branch in keys_in]
        if not read_cols:
            return dumps

        entry_set = set(entries)
        n_entries = tree_in.num_entries
        for entry_start in range(0, n_entries, args.step_size):
            entry_stop = min(entry_start + args.step_size, n_entries)
            chunk_entries = [e for e in entry_set if entry_start <= e < entry_stop]
            if not chunk_entries:
                continue

            arrs = tree_in.arrays(
                read_cols,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="ak",
            )
            for ev in chunk_entries:
                local_i = ev - entry_start
                for out_col, branch in branch_map.items():
                    if branch not in read_cols:
                        dumps[out_col][ev] = "[]"
                        continue
                    try:
                        dumps[out_col][ev] = json.dumps(ak.to_list(arrs[branch][local_i]))
                    except Exception:
                        dumps[out_col][ev] = "[]"

    return dumps


# ──────────────────────────────────────────────────────────────────
# Extract ch192 times from ROOT (fast vectorised scan)
# ──────────────────────────────────────────────────────────────────

def extract_channel_times(root_path, channel, branch_channel, branch_idx,
                          branch_time, max_entries, step_size=200000,
                          require_trigger=True, trigger_channel=192):
    log(f"Opening ROOT file: {root_path}")
    f = uproot.open(root_path)
    tree_name = find_data_tree(f)
    if tree_name is None:
        raise RuntimeError("No data tree found in ROOT file.")
    tree = f[tree_name]
    keys = list(tree.keys())

    ch_branch = branch_channel or ("channelID" if "channelID" in keys else None)
    idx_branch = branch_idx or ("channelIdx" if "channelIdx" in keys else None)
    time_branch = branch_time or ("time" if "time" in keys else None)
    t1c_branch = "t1coarse" if "t1coarse" in keys else None
    if not ch_branch or not idx_branch or not time_branch:
        raise RuntimeError(f"Missing required branches. Found ch={ch_branch}, idx={idx_branch}, time={time_branch}")

    n_entries = tree.num_entries
    use_n = n_entries if max_entries is None else min(max_entries, n_entries)
    log(f'Data tree="{tree_name}", entries={n_entries}, processing={use_n}, channel={channel}')

    times, entries, t1c_vals, n_ch_list = [], [], [], []
    read_branches = [ch_branch, idx_branch, time_branch]
    if t1c_branch:
        read_branches.append(t1c_branch)

    for entry_start in range(0, use_n, step_size):
        entry_stop = min(entry_start + step_size, use_n)
        arrays = tree.arrays(read_branches, entry_start=entry_start,
                             entry_stop=entry_stop, library="ak")
        ch_list = arrays[ch_branch]
        idx_list = arrays[idx_branch]
        time_list = arrays[time_branch]
        t1c_list = arrays[t1c_branch] if t1c_branch else None

        mask = ak.any(ch_list == channel, axis=1)
        if require_trigger:
            mask = mask & ak.any(ch_list == trigger_channel, axis=1)
        if not ak.any(mask):
            continue

        idx_sel_py = ak.to_list(idx_list[mask])
        mapped_idx = np.array(
            [row[channel] if isinstance(row, list) and channel < len(row) else -1
             for row in idx_sel_py], dtype=int)

        time_sel_py = ak.to_list(time_list[mask])
        valid = np.array(
            [0 <= mi < len(trow) if isinstance(trow, list) else False
             for mi, trow in zip(mapped_idx, time_sel_py)], dtype=bool)
        if not np.any(valid):
            continue

        mapped_idx = mapped_idx[valid]
        time_sel_py = [t for t, ok in zip(time_sel_py, valid) if ok]
        tvals = np.array(
            [t[i] if isinstance(t, list) and i < len(t) else np.nan
             for t, i in zip(time_sel_py, mapped_idx)], dtype=float)

        if t1c_list is not None:
            t1c_py = [t for t, ok in zip(ak.to_list(t1c_list[mask]), valid) if ok]
            t1c_chunk = np.array(
                [t[i] if isinstance(t, list) and i < len(t) else np.nan
                 for t, i in zip(t1c_py, mapped_idx)], dtype=float)
        else:
            t1c_chunk = np.full(len(tvals), np.nan)

        local_idx = np.arange(entry_start, entry_stop, dtype=int)
        local_idx = local_idx[ak.to_numpy(mask)][valid]
        n_ch_arr = ak.to_numpy(ak.num(ch_list[mask], axis=1))[valid]

        for t, ev, tc, nch in zip(tvals, local_idx, t1c_chunk, n_ch_arr):
            if t == t:
                times.append(float(t))
                entries.append(int(ev))
                t1c_vals.append(float(tc))
                n_ch_list.append(int(nch))

        if entry_stop % (step_size * 5) == 0:
            log(f"  Scanned {entry_stop}/{use_n}, matched so far: {len(times)}")

    log(f"Finished ROOT scan: {len(times)} events with channel {channel}")
    return (np.array(times, dtype=float), np.array(entries, dtype=int),
            np.array(t1c_vals, dtype=float), np.array(n_ch_list, dtype=int),
            tree_name, use_n)


# ──────────────────────────────────────────────────────────────────
# Dedup double-counted triggers using t1coarse
# ──────────────────────────────────────────────────────────────────

def dedup_triggers(times, entries, t1c, n_ch, threshold=8.0):
    if len(times) < 2 or np.all(np.isnan(t1c)):
        return times, entries, t1c, n_ch

    order = np.argsort(times)
    times, entries, t1c, n_ch = times[order], entries[order], t1c[order], n_ch[order]
    dt1c = np.abs(np.diff(t1c))

    has_twin = np.zeros(len(times), dtype=bool)
    keep = np.ones(len(times), dtype=bool)
    i = 0
    while i < len(dt1c):
        if dt1c[i] < threshold:
            has_twin[i] = has_twin[i + 1] = True
            keep[i + 1] = False
            i += 2
        else:
            i += 1

    # Remove orphan fakes — events without a twin that only have 1 channel
    n_orphans = 0
    for idx in range(len(times)):
        if keep[idx] and not has_twin[idx] and n_ch[idx] <= 1:
            keep[idx] = False
            n_orphans += 1

    n_removed = len(times) - keep.sum()
    if n_removed > 0:
        log(f"Dedup: removed {n_removed - n_orphans} twin dupes + {n_orphans} orphan fakes")
    return times[keep], entries[keep], t1c[keep], n_ch[keep]


# ──────────────────────────────────────────────────────────────────
# Core: Cumulative-time matching
# ──────────────────────────────────────────────────────────────────

def match_cumulative_time(ch192_times, trigger_times_ps, tol_nsigma=5.0):
    """
    Match ch192 events to trigger events using relative-time matching.

    Works in relative coordinates (subtract first value from both) for
    numerical precision, then uses searchsorted to find nearest matches.

    Returns:
        root_to_trigger, trigger_to_root, slope, offset
    """
    n_root = len(ch192_times)
    n_trig = len(trigger_times_ps)

    root_to_trigger = np.full(n_root, np.nan)
    trigger_to_root = np.full(n_trig, np.nan)

    if n_root < 2 or n_trig < 2:
        return root_to_trigger, trigger_to_root, 1.0, 0.0

    log(f"  Matching: {n_root} ROOT events <-> {n_trig} triggers")

    # Work in relative coordinates for numerical stability
    T_root = ch192_times - ch192_times[0]
    T_trig = trigger_times_ps - trigger_times_ps[0]

    # ── Step 1: End-trim ──
    slope_tol = 1e-5
    trig_start = 0

    def _end_trim(T_root, T_trig, n_trig, slope_tol, n_root):
        """Trim triggers from end until |slope-1| < tol. Returns (slope, n_use, n_trimmed)."""
        n_use = n_trig
        slope = T_root[-1] / T_trig[n_use - 1] if T_trig[n_use - 1] > 0 else 1.0
        nt = 0
        while abs(slope - 1.0) > slope_tol and n_use > n_root // 2:
            n_use -= 1
            slope = T_root[-1] / T_trig[n_use - 1] if T_trig[n_use - 1] > 0 else 1.0
            nt += 1
        return slope, n_use, nt

    slope, n_trig_use, n_trimmed_end = _end_trim(T_root, T_trig, n_trig, slope_tol, n_root)
    if n_trimmed_end > 0:
        log(f"  End-trim: removed {n_trimmed_end} triggers (slope={slope:.9f})")

    # ── Step 2: If >10 trimmed from end, check front alignment ──
    if n_trimmed_end > 10:
        log(f"  End-trim removed {n_trimmed_end} triggers → checking front alignment")
        n_cc = min(10, n_root - 1, n_trig - 1)
        if n_cc >= 5:
            dt_r = np.diff(T_root[:n_cc + 1])
            dt_t = np.diff(T_trig[:n_cc + 1])
            max_shift = min(5, n_cc - 3)
            scores = {}
            for s in range(max_shift + 1):
                n_overlap = min(len(dt_r), len(dt_t) - s)
                if n_overlap < 3:
                    break
                valid = dt_t[s:s + n_overlap] > 1e-6
                if valid.sum() < 3:
                    continue
                ratios = dt_r[:n_overlap][valid] / dt_t[s:s + n_overlap][valid]
                scores[s] = np.median(np.abs(ratios - 1.0))
            if scores:
                best_shift = min(scores, key=scores.get)
                for s in sorted(scores):
                    log(f"    shift={s}: score={scores[s]:.6f}")
                if best_shift > 0:
                    trig_start = best_shift
                    log(f"  Front-trim: dropping {trig_start} triggers from start")
                    trigger_times_ps = trigger_times_ps[trig_start:]
                    n_trig = len(trigger_times_ps)
                    T_trig = trigger_times_ps - trigger_times_ps[0]
                    # Re-print first 10 events after trim
                    n_show = min(10, n_root, n_trig)
                    dt_r2 = np.diff(T_root[:n_show + 1])
                    dt_t2 = np.diff(T_trig[:n_show + 1])
                    log(f"  Post front-trim first {n_show} events (ms):")
                    log(f"    {'i':>3}  {'ROOT_rel':>12}  {'dt_ROOT':>10}  "
                        f"{'Trig_rel':>12}  {'dt_Trig':>10}")
                    for k in range(n_show):
                        r_rel = T_root[k] / 1e9
                        t_rel = T_trig[k] / 1e9
                        dr = dt_r2[k] / 1e9 if k < len(dt_r2) else float('nan')
                        dt = dt_t2[k] / 1e9 if k < len(dt_t2) else float('nan')
                        log(f"    {k:>3}  {r_rel:>12.3f}  {dr:>10.3f}  "
                            f"{t_rel:>12.3f}  {dt:>10.3f}")
                    # Redo end-trim with new T_trig
                    slope, n_trig_use, n_trimmed_end2 = _end_trim(
                        T_root, T_trig, n_trig, slope_tol, n_root)
                    if n_trimmed_end2 > 0:
                        log(f"  Re-end-trim: removed {n_trimmed_end2} triggers")
                    n_trimmed_end = n_trimmed_end2

    offset = 0.0
    log(f"  Initial slope={slope:.9f}, offset={offset:.1f}, "
        f"using {n_trig_use}/{n_trig} triggers (front={trig_start}, end={n_trimmed_end})")

    # Helper: find nearest T_root for each expected value
    def _find_nearest(T_root, expected, n_root):
        idx = np.searchsorted(T_root, expected)
        idx = np.clip(idx, 0, n_root - 1)
        best = np.empty(len(expected), dtype=int)
        for j in range(len(expected)):
            i = idx[j]
            if i > 0 and abs(T_root[i-1] - expected[j]) < abs(T_root[i] - expected[j]):
                best[j] = i - 1
            else:
                best[j] = i
        return best

    # Iterative matching + refinement
    for it in range(3):
        expected = slope * T_trig + offset
        best_idx = _find_nearest(T_root, expected, n_root)
        residuals = T_root[best_idx] - expected

        med_r = np.median(residuals)
        mad_r = np.median(np.abs(residuals - med_r))
        sig = 1.4826 * mad_r if mad_r > 0 else max(np.std(residuals), 1.0)

        good = np.abs(residuals - med_r) < tol_nsigma * sig
        if good.sum() < 3:
            good = np.ones(n_trig, dtype=bool)

        if good.sum() >= 2:
            slope, offset = np.polyfit(T_trig[good], T_root[best_idx[good]], 1)
        log(f"  Iter {it}: slope={slope:.9f}, offset={offset:.1f}, "
            f"good={good.sum()}/{n_trig}, σ={sig:.1f} ps")

    # Final matching with conflict resolution
    expected = slope * T_trig + offset
    best_idx = _find_nearest(T_root, expected, n_root)
    residuals = T_root[best_idx] - expected
    med_f = np.median(residuals)
    mad_f = np.median(np.abs(residuals - med_f))
    sig_f = 1.4826 * mad_f if mad_f > 0 else max(np.std(residuals), 1.0)
    accept = np.abs(residuals - med_f) < tol_nsigma * sig_f

    used_root = {}
    for j in range(n_trig):
        if not accept[j]:
            continue
        i = best_idx[j]
        r = abs(residuals[j])
        if i not in used_root or r < used_root[i][1]:
            used_root[i] = (j, r)

    n_matched = 0
    for i, (j, _) in used_root.items():
        root_to_trigger[i] = j
        trigger_to_root[j] = i
        n_matched += 1

    log(f"  Final: slope={slope:.9f}, offset={offset:.1f} ps, "
        f"matched={n_matched}/{n_trig} (σ={sig_f:.1f} ps)")
    return root_to_trigger, trigger_to_root, slope, offset


# ──────────────────────────────────────────────────────────────────
# Diagnostic plots
# ──────────────────────────────────────────────────────────────────

def _save_segment_diagnostic(root_cluster, trigger_ps, slope, offset,
                             root_to_trigger, seg_num, out_prefix):
    """Save per-segment diagnostic: dt comparison, cumulative alignment, dt histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Segment {seg_num} diagnostic", fontsize=14)

    T_root = root_cluster - root_cluster[0]
    T_trig = trigger_ps - trigger_ps[0]
    n_use = min(len(T_root), len(T_trig))

    # Panel 1: dt scatter (ROOT dt[i] vs trigger dt[i] — raw index aligned)
    dt_root = np.diff(root_cluster)
    dt_trig = np.diff(trigger_ps)
    n_dt = min(len(dt_root), len(dt_trig))
    axes[0].scatter(np.arange(n_dt), dt_root[:n_dt] / 1e9, s=1, alpha=0.5, label="ROOT dt (ms)")
    axes[0].scatter(np.arange(n_dt), dt_trig[:n_dt] / 1e9, s=1, alpha=0.5, label="Trig dt (ms)")
    axes[0].set_xlabel("Event index")
    axes[0].set_ylabel("Δt (ms)")
    axes[0].set_title("dt by index (raw)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=7, markerscale=5)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: cumulative time difference
    # T_root[i] - slope * T_trig[i] should be ~constant if aligned
    proj = slope * T_trig[:n_use] + offset
    diff_cum = T_root[:n_use] - proj
    axes[1].plot(np.arange(n_use), diff_cum / 1e9, linewidth=0.5, alpha=0.8)
    axes[1].set_xlabel("Event index")
    axes[1].set_ylabel("T_root − projected T_trig (ms)")
    axes[1].set_title("Cumulative time alignment residual")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: histogram of ROOT dt — reveals anomalous small gaps (extra events)
    dt_root_ms = dt_root / 1e9
    axes[2].hist(dt_root_ms, bins=200, alpha=0.7, color="steelblue")
    axes[2].set_xlabel("ROOT Δt (ms)")
    axes[2].set_ylabel("Counts")
    axes[2].set_title("ROOT dt distribution")
    axes[2].set_yscale("log")
    # Mark the median trigger dt for reference
    med_trig_dt = np.median(np.diff(trigger_ps)) / 1e9
    axes[2].axvline(med_trig_dt, color="red", linestyle="--", linewidth=1.5,
                    label=f"Median trig dt={med_trig_dt:.2f} ms")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=150)
    plt.close(fig)
    log(f"  Saved diagnostic: {out_prefix}.png")

def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_diagnostic_plots(validation_segments, out_prefix):
    """Create validation scatter + residual plots."""
    if not validation_segments:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors
    all_residuals = []

    for seg_i, (trig_ps, root_ps) in enumerate(validation_segments):
        if len(trig_ps) < 2:
            continue
        color = colors[seg_i % len(colors)]
        ax1.scatter(trig_ps, root_ps, s=1, alpha=0.5, color=color,
                    label=f"Seg {seg_i + 1} ({len(trig_ps)} evts)")
        coeffs = np.polyfit(trig_ps, root_ps, 1)
        x_range = np.linspace(trig_ps.min(), trig_ps.max(), 100)
        ax1.plot(x_range, np.polyval(coeffs, x_range), color=color, linewidth=1.5,
                 label=f"Fit: slope={coeffs[0]:.6f}")
        residuals = root_ps - np.polyval(coeffs, trig_ps)
        all_residuals.append(residuals)

    ax1.set_xlabel("Trigger time (ps)")
    ax1.set_ylabel("Channel 192 time (ps)")
    ax1.set_title("Ch192 time vs Trigger time")
    ax1.legend(fontsize=7, markerscale=5)

    if all_residuals:
        all_res = np.concatenate(all_residuals)
        all_res = all_res[np.isfinite(all_res)]
        if len(all_res) > 10:
            n_bins = min(200, max(50, len(all_res) // 20))
            counts, bin_edges, _ = ax2.hist(all_res, bins=n_bins, alpha=0.7,
                                             color="steelblue",
                                             label=f"Residuals (N={len(all_res)})")
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            try:
                p0 = [counts.max(), np.mean(all_res), np.std(all_res)]
                popt, _ = curve_fit(_gauss, bin_centers, counts, p0=p0, maxfev=5000)
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
                ax2.plot(x_fit, _gauss(x_fit, *popt), "r-", linewidth=2,
                         label=f"Gauss: μ={popt[1]:.1f}, σ={popt[2]:.1f}")
            except Exception:
                pass
            ax2.legend(fontsize=8)

    ax2.set_xlabel("Residual (ps)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Fit residuals")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_validation.png", dpi=150)
    plt.close(fig)
    log(f"Saved: {out_prefix}_validation.png")


def make_dt_ratio_plots(validation_segments, out_prefix):
    """Plot Δt_ch192 / Δt_trigger for matched consecutive events per segment."""
    if not validation_segments:
        return
    colors = plt.cm.tab10.colors

    # Combined plot: all segments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    all_ratios = []
    offset = 0

    for seg_i, (trig_ps, root_ps) in enumerate(validation_segments):
        if len(trig_ps) < 3:
            continue
        # Sort by trigger time to get consecutive pairs
        order = np.argsort(trig_ps)
        trig_sorted = trig_ps[order]
        root_sorted = root_ps[order]

        dt_trig = np.diff(trig_sorted)
        dt_root = np.diff(root_sorted)

        # Avoid division by zero
        valid = np.abs(dt_trig) > 1e-6
        ratios = np.full(len(dt_trig), np.nan)
        ratios[valid] = dt_root[valid] / dt_trig[valid]

        finite = np.isfinite(ratios)
        if not np.any(finite):
            continue

        color = colors[seg_i % len(colors)]
        x_vals = np.arange(len(ratios)) + offset
        ax1.scatter(x_vals[finite], ratios[finite], s=2, alpha=0.4, color=color,
                    label=f"Seg {seg_i + 1} (N={finite.sum()})")
        all_ratios.append(ratios[finite])
        offset += len(ratios)

    if all_ratios:
        all_r = np.concatenate(all_ratios)
        med = np.median(all_r)
        ax1.axhline(med, color="red", linewidth=1.5, linestyle="--",
                    label=f"Median={med:.6f}")
        ax1.axhline(1.0, color="grey", linewidth=1, linestyle=":", alpha=0.5)
        ax1.set_xlabel("Event index")
        ax1.set_ylabel("Δt_ch192 / Δt_trigger")
        ax1.set_title("dt ratio (matched consecutive events)")
        ax1.set_yscale("log")
        ax1.legend(fontsize=7, markerscale=4)
        ax1.grid(True, alpha=0.3)

        # Histogram of ratios (clip outliers for readability)
        clip_lo, clip_hi = np.percentile(all_r, [1, 99])
        clipped = all_r[(all_r >= clip_lo) & (all_r <= clip_hi)]
        n_bins = min(200, max(50, len(clipped) // 15))
        counts, bin_edges, _ = ax2.hist(clipped, bins=n_bins, alpha=0.7,
                                         color="steelblue",
                                         label=f"dt ratio (N={len(clipped)})")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        try:
            p0 = [counts.max(), np.median(clipped), np.std(clipped)]
            popt, _ = curve_fit(_gauss, bin_centers, counts, p0=p0, maxfev=5000)
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
            ax2.plot(x_fit, _gauss(x_fit, *popt), "r-", linewidth=2,
                     label=f"Gauss: μ={popt[1]:.6f}, σ={popt[2]:.6f}")
        except Exception:
            pass
        ax2.set_xlabel("Δt_ch192 / Δt_trigger")
        ax2.set_ylabel("Counts")
        ax2.set_title("dt ratio distribution")
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_dt_ratio.png", dpi=150)
    plt.close(fig)
    log(f"Saved: {out_prefix}_dt_ratio.png")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Simplified FEBD sync + MCP attach")
    p.add_argument("--root", required=True, help="Input ROOT file")
    p.add_argument("--meta-path", help="Meta CSV file or glob")
    p.add_argument("--meta-dir", default="../trc_out/", help="Directory for meta CSVs")
    p.add_argument("--peaks-dir", required=True, help="Directory with peaks CSVs")
    p.add_argument("--peaks-pattern", help="Peaks file glob in peaks-dir")
    p.add_argument("--channel", type=int, default=192, help="Channel to sync")
    p.add_argument("--out-root", required=True, help="Output ROOT file")
    p.add_argument("--out-matched-csv", required=True, help="Output matched CSV")
    p.add_argument("--branch-channel", default="channelID")
    p.add_argument("--branch-idx", default="channelIdx")
    p.add_argument("--branch-time", default="time")
    p.add_argument("--branch-energy", default="energy")
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--fast", action="store_true", help="(accepted for compat, always fast)")
    p.add_argument("--step-size", type=int, default=300000)
    p.add_argument("--require-trigger", action="store_true")
    p.add_argument("--gap-factor", type=float, default=500.0)
    p.add_argument("--match-tol", type=float, default=5.0,
                   help="Match tolerance in σ (MAD-based, default: 5)")
    p.add_argument("--dedup", action="store_true",
                   help="Dedup double-counted triggers using t1coarse")
    p.add_argument("--dedup-threshold-t1c", type=float, default=8.0)
    args = p.parse_args()
    log("Starting simplified FEBD sync + MCP attach")

    # Auto-infer meta and peaks patterns from ROOT path
    if not args.meta_path or not args.peaks_pattern:
        m = re.search(r'/([^/]+)/(\d+)_[^/]*\.root$', args.root)
        if m:
            run, spill = int(m.group(1)), int(m.group(2))
        else:
            raise ValueError(f"Cannot extract run/spill from '{args.root}'. "
                             "Provide --meta-path and --peaks-pattern.")
        pr, ps = f"{run:07d}", f"{spill:07d}"
        if not args.meta_path:
            args.meta_path = os.path.join(args.meta_dir, f"raw_C2_{pr}_{ps}_*_meta.csv")
            log(f"Auto-inferred --meta-path: {args.meta_path}")
        if not args.peaks_pattern:
            args.peaks_pattern = f"peaks_raw_C1_{pr}_{ps}_*_data_with_tave.csv"
            log(f"Auto-inferred --peaks-pattern: {args.peaks_pattern}")

    # ── Step 1: Extract ch192 times from ROOT ──
    root_times, root_entries, root_t1c, root_nch, tree_name, used_entries = \
        extract_channel_times(
            args.root, args.channel, args.branch_channel, args.branch_idx,
            args.branch_time, args.max_entries, args.step_size,
            args.require_trigger, trigger_channel=192)
    if len(root_times) == 0:
        raise RuntimeError("No valid channel times extracted from ROOT.")

    # ── Step 2: Optional dedup ──
    if args.dedup:
        root_times, root_entries, root_t1c, root_nch = dedup_triggers(
            root_times, root_entries, root_t1c, root_nch, args.dedup_threshold_t1c)

    # ── Step 3: Cluster by time gaps ──
    log("Clustering ROOT channel times by large gaps")
    clusters, clusters_idx = split_by_time_gaps_with_indices(
        root_times, root_entries, args.gap_factor)
    log(f"Built {len(clusters)} cluster(s): sizes={[len(c) for c in clusters]}")

    # ── Step 4: Resolve meta files ──
    log(f"Resolving meta files from: {args.meta_path}")
    meta_files = resolve_meta_files(args.meta_path)
    if not meta_files:
        raise FileNotFoundError(f"No meta files from: {args.meta_path}")
    log(f"Resolved {len(meta_files)} meta file(s)")

    use_n = min(len(clusters), len(meta_files))
    if use_n == 0:
        raise RuntimeError("No usable segment pairing.")
    log(f"Processing {use_n} segment(s)")

    # ── Step 5: Match per segment ──
    all_indices = []
    all_peak_time, all_peak_amp, all_peak_sigma = [], [], []
    all_peak_phi, all_peak_phi_from_edge = [], []
    all_trigger_time, all_trigger_offset = [], []
    all_trigger_phi, all_trigger_phi_from_edge = [], []
    all_t_ave = []
    matched_rows = []
    validation_segments = []
    seg_stats = []  # per-segment stats for summary

    for seg_idx in range(use_n):
        meta_file = meta_files[seg_idx]
        root_cluster = np.asarray(clusters[seg_idx], dtype=float)
        root_cluster_idx = np.asarray(clusters_idx[seg_idx], dtype=int)
        log(f"\n[segment {seg_idx + 1}/{use_n}] meta={Path(meta_file).name}, "
            f"cluster_size={len(root_cluster)}")

        if len(root_cluster) < 2:
            log(f"[segment {seg_idx + 1}] skipped: cluster has <2 points")
            continue

        # Read trigger times
        trigger_values = read_trigger_values(meta_file)
        trigger_ps = np.array(trigger_values, dtype=float) * 1e12
        log(f"[segment {seg_idx + 1}] trigger_count={len(trigger_ps)}")
        if len(trigger_ps) < 2:
            log(f"[segment {seg_idx + 1}] skipped: <2 triggers")
            continue

        # ── Diagnostic: compare time ranges ──
        r_span = root_cluster[-1] - root_cluster[0]
        t_span = trigger_ps[-1] - trigger_ps[0]
        span_ratio = r_span / t_span if t_span > 0 else float('inf')
        size_ratio = len(root_cluster) / len(trigger_ps)
        log(f"[segment {seg_idx + 1}] ROOT  range: [{root_cluster[0]:.0f}, {root_cluster[-1]:.0f}] "
            f"span={r_span:.0f} ps")
        log(f"[segment {seg_idx + 1}] Trig  range: [{trigger_ps[0]:.0f}, {trigger_ps[-1]:.0f}] "
            f"span={t_span:.0f} ps")
        log(f"[segment {seg_idx + 1}] span_ratio={span_ratio:.6f}, "
            f"size_ratio={size_ratio:.3f} ({len(root_cluster)}/{len(trigger_ps)})")

        # ── Diagnostic: dump first 10 ROOT and trigger events ──
        n_show = min(10, len(root_cluster), len(trigger_ps))
        dt_r = np.diff(root_cluster[:n_show + 1])
        dt_t = np.diff(trigger_ps[:n_show + 1])
        log(f"[segment {seg_idx + 1}] First {n_show} events (relative to first, in ms):")
        log(f"  {'i':>3}  {'ROOT_rel':>14}  {'dt_ROOT':>12}  {'Trig_rel':>14}  {'dt_Trig':>12}")
        for k in range(n_show):
            r_rel = (root_cluster[k] - root_cluster[0]) / 1e9
            t_rel = (trigger_ps[k] - trigger_ps[0]) / 1e9
            dr = dt_r[k] / 1e9 if k < len(dt_r) else float('nan')
            dt = dt_t[k] / 1e9 if k < len(dt_t) else float('nan')
            log(f"  {k:>3}  {r_rel:>14.3f}  {dr:>12.3f}  {t_rel:>14.3f}  {dt:>12.3f}")

        # ── Cumulative-time matching ──
        root_to_trigger, trigger_to_root, slope, offset = \
            match_cumulative_time(root_cluster, trigger_ps, tol_nsigma=args.match_tol)

        # ── Per-segment diagnostic plot ──
        out_prefix_seg = str(Path(args.out_root).with_suffix('')) + f"_seg{seg_idx+1}_diag"
        _save_segment_diagnostic(root_cluster, trigger_ps, slope, offset,
                                 root_to_trigger, seg_idx + 1, out_prefix_seg)

        # Resolve peaks file
        mnum = re.search(r"_(\d+(?:_\d+)+)_meta\.csv$", Path(meta_file).name)
        suffix = mnum.group(1) if mnum else f"seg{seg_idx + 1}"
        peaks_file = find_peaks_file(args.peaks_dir, args.peaks_pattern, suffix)
        if peaks_file is None:
            raise FileNotFoundError(
                f'No peaks file for suffix "{suffix}" with pattern "{args.peaks_pattern}"')
        log(f"[segment {seg_idx + 1}] peaks={Path(peaks_file).name}")
        seg_to_peak = load_peak_lookup(peaks_file)

        # Build output for all ROOT events in this cluster
        before_seg = len(all_indices)
        seg_match_trig, seg_match_root = [], []

        for i_root in range(len(root_cluster)):
            j_trigger = root_to_trigger[i_root]
            root_entry = int(root_cluster_idx[i_root])
            root_time_ps = float(root_cluster[i_root])

            if np.isnan(j_trigger):
                # Unmatched ROOT event
                segment_num = -1
                vals = (np.nan,) * 7
                phi_peak = phi_peak_edge = phi_trigger = phi_trigger_edge = np.nan
                mapped_trigger_ps = np.nan
            else:
                j_trig = int(j_trigger)
                segment_num = j_trig + 1  # Peaks CSV uses 1-based numbering
                vals = seg_to_peak.get(
                    segment_num, (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                peak_time_ps, peak_amp, peak_sigma_ps, t_ave_ps, \
                    trigger_time_ps, trigger_offset_ps, prev_edge_ps = vals
                phi_peak = compute_phi(peak_time_ps, t_ave_ps)
                phi_peak_edge = compute_phi(peak_time_ps, prev_edge_ps)
                phi_trigger = compute_phi(trigger_time_ps, t_ave_ps)
                phi_trigger_edge = compute_phi(trigger_time_ps, prev_edge_ps)
                mapped_trigger_ps = float(trigger_ps[j_trig])

            if np.isnan(j_trigger):
                peak_time_ps = peak_amp = peak_sigma_ps = np.nan
                t_ave_ps = trigger_time_ps = trigger_offset_ps = prev_edge_ps = np.nan
                phi_peak = phi_peak_edge = phi_trigger = phi_trigger_edge = np.nan

            if not np.isnan(mapped_trigger_ps):
                seg_match_trig.append(mapped_trigger_ps)
                seg_match_root.append(root_time_ps)

            all_indices.append(root_entry)
            all_peak_time.append(peak_time_ps)
            all_peak_amp.append(peak_amp)
            all_peak_sigma.append(peak_sigma_ps)
            all_peak_phi.append(phi_peak)
            all_peak_phi_from_edge.append(phi_peak_edge)
            all_trigger_time.append(trigger_time_ps)
            all_trigger_offset.append(trigger_offset_ps)
            all_trigger_phi.append(phi_trigger)
            all_trigger_phi_from_edge.append(phi_trigger_edge)
            all_t_ave.append(t_ave_ps)

            matched_rows.append({
                "entry": root_entry,
                "mcp_index": root_entry,
                "mcp_peak_time": peak_time_ps,
                "mcp_peak_amp": peak_amp,
                "mcp_peak_sigma_ps": peak_sigma_ps,
                "mcp_peak_phase": phi_peak,
                "mcp_trigger_time": trigger_time_ps,
                "mcp_trigger_offset_ps": trigger_offset_ps,
                "t0_abs_ps": t_ave_ps,
                "prev_edge_ps": prev_edge_ps,
                "phi_peak_from_trigger": compute_phi(peak_time_ps, trigger_time_ps),
                "peak_minus_t0_ps": _delta_or_nan(peak_time_ps, t_ave_ps),
                "peak_minus_prev_edge_ps": _delta_or_nan(peak_time_ps, prev_edge_ps),
                "trigger_minus_t0_ps": _delta_or_nan(trigger_time_ps, t_ave_ps),
                "root_time_ps": root_time_ps,
                "meta_file": str(Path(meta_file).name),
                "peaks_file": str(Path(peaks_file).name),
                "segment": int(segment_num),
                "trigger_ps_from_meta": mapped_trigger_ps,
                "phi_peak_from_edge": phi_peak_edge,
                "phi_trigger_from_edge": phi_trigger_edge,
                "t_ave_ps": t_ave_ps,
            })

        n_matched = len(all_indices) - before_seg
        n_with_trig = len(seg_match_trig)
        seg_stats.append({
            "seg": seg_idx + 1,
            "cluster_size": len(root_cluster),
            "trigger_count": len(trigger_ps),
            "matched": n_with_trig,
            "unmatched": n_matched - n_with_trig,
        })
        log(f"[segment {seg_idx + 1}] events_added={n_matched}, with_trigger={n_with_trig}")

        # Validation data
        if seg_match_trig:
            t_arr = np.array(seg_match_trig, dtype=float)
            r_arr = np.array(seg_match_root, dtype=float)
            ok = np.isfinite(t_arr) & np.isfinite(r_arr)
            validation_segments.append((t_arr[ok], r_arr[ok]))

    # ── Step 6: Write output ROOT ──
    out_root_path = Path(args.out_root)
    out_root_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Copying input ROOT → output: {out_root_path}")
    shutil.copy2(args.root, out_root_path)

    log(f"Writing MCP tree with {len(all_indices)} entries")
    with uproot.update(str(out_root_path)) as f:
        f["MCP"] = {
            "index": np.array(all_indices, dtype=np.int64),
            "peak_time": np.array(all_peak_time, dtype=np.float64),
            "peak_amp": np.array(all_peak_amp, dtype=np.float64),
            "peak_sigma_ps": np.array(all_peak_sigma, dtype=np.float64),
            "phi_peak": np.array(all_peak_phi, dtype=np.float64),
            "phi_peak_from_edge": np.array(all_peak_phi_from_edge, dtype=np.float64),
            "trigger_time": np.array(all_trigger_time, dtype=np.float64),
            "trigger_offset_ps": np.array(all_trigger_offset, dtype=np.float64),
            "phi_trigger": np.array(all_trigger_phi, dtype=np.float64),
            "phi_trigger_from_edge": np.array(all_trigger_phi_from_edge, dtype=np.float64),
            "t_ave_ps": np.array(all_t_ave, dtype=np.float64),
        }

    # ── Step 7: Write matched CSV ──
    out_csv_path = Path(args.out_matched_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    log("Fetching original TOFHIR arrays for matched CSV")
    root_arrays = fetch_root_arrays_for_entries(args.root, tree_name, all_indices, args)
    out_df = build_legacy_matched_dataframe(matched_rows, root_arrays)
    out_df.to_csv(out_csv_path, index=False)
    log(f"Wrote {len(out_df)} rows to {out_csv_path}")

    # ── Step 8: Diagnostic plots ──
    out_prefix = str(out_root_path).replace(".root", "")
    make_diagnostic_plots(validation_segments, out_prefix)
    make_dt_ratio_plots(validation_segments, out_prefix)

    # Summary
    total_matched = sum(1 for r in matched_rows if r["segment"] > 0)
    total_unmatched = sum(1 for r in matched_rows if r["segment"] <= 0)
    log(f"\n{'='*60}")
    log(f"  Total ROOT events processed: {len(matched_rows)}")
    log(f"  Matched to trigger:          {total_matched}")
    log(f"  Unmatched:                   {total_unmatched}")
    log(f"  Segments:                    {use_n}")
    log(f"  {'─'*56}")
    log(f"  {'Seg':>4} {'Cluster':>8} {'Triggers':>9} {'Matched':>8} {'Unmatched':>10}")
    for s in seg_stats:
        log(f"  {s['seg']:>4} {s['cluster_size']:>8} {s['trigger_count']:>9} "
            f"{s['matched']:>8} {s['unmatched']:>10}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
