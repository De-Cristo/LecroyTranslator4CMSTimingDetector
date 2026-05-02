#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Single-pass FEBD sync + MCP attach workflow.

This script replaces:
1) read_root_explore.py --dump-df ...
2) Febd_synchronizor.py --csv-path ...
3) apply_mapping_add_peaks.py ...

with one command that:
- reads ROOT events directly,
- synchronizes ROOT channel times with scope trigger_time from meta CSV(s),
- matches MCP peaks,
- writes output ROOT with MCP tree,
- writes a compact matched-events CSV.
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
    print(f"[fast-sync] {msg}", flush=True)


def find_data_tree(f):
    all_keys = list(f.keys())
    data_keys = [k for k in all_keys if k.startswith("data")]
    if data_keys:
        best = None
        best_cycle = -1
        for k in data_keys:
            if ";" in k:
                try:
                    cycle = int(k.split(";", 1)[1])
                except Exception:
                    cycle = 0
            else:
                cycle = 0
            if cycle > best_cycle:
                best_cycle = cycle
                best = k
        return best
    tnames = [k for k, v in f.items() if hasattr(v, "num_entries")]
    if not tnames:
        return None
    return tnames[0]


def split_by_time_gaps_with_indices(values, indices, gap_factor=100):
    vals = np.asarray(values)
    idxs = np.asarray(indices)
    order = np.argsort(vals)
    vals_sorted = vals[order]
    idxs_sorted = idxs[order]
    if len(vals_sorted) < 2:
        return [vals_sorted], [idxs_sorted]
    gaps = np.diff(vals_sorted)
    threshold = np.median(gaps) * gap_factor
    split_idxs = np.where(gaps > threshold)[0]
    vals_clusters = np.split(vals_sorted, split_idxs + 1)
    idxs_clusters = np.split(idxs_sorted, split_idxs + 1)
    return vals_clusters, idxs_clusters


def find_missing_by_shift(dt_trigger, dt_root, ratio_threshold=2.0, eps=1e-13):
    missing_trigger_indices = []
    missing_root_indices = []
    aligned_ratios = []
    root_to_trigger = []

    shift = 0
    i_root = 0

    while i_root < len(dt_root):
        j = i_root + shift
        if j >= len(dt_trigger):
            break

        r = dt_root[i_root]
        t = dt_trigger[j]
        ratio = r / (t + eps)

        if ratio > ratio_threshold and j + 1 < len(dt_trigger):
            t2 = t + dt_trigger[j + 1]
            ratio2 = r / (t2 + eps)

            if 0.99 < ratio2 < 1.01:
                missing_trigger_indices.append(j)
                missing_root_indices.append(i_root)
                shift += 1
                j = i_root + shift
                if j >= len(dt_trigger):
                    break
                t = dt_trigger[j]
                ratio = r / (t + eps)

        aligned_ratios.append(ratio)
        root_to_trigger.append(j)
        i_root += 1

    trigger_to_root = np.full(len(dt_trigger), np.nan)
    for i_root_local, j_trigger in enumerate(root_to_trigger):
        trigger_to_root[j_trigger] = i_root_local

    return (
        np.array(missing_trigger_indices),
        np.array(missing_root_indices),
        np.array(root_to_trigger),
        trigger_to_root,
        np.array(aligned_ratios),
    )


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
            if not p:
                continue
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

    t0_col = None
    if "t0_abs_ps" in df.columns:
        t0_col = "t0_abs_ps"
    elif "t0_abs_ns" in df.columns:
        t0_col = "t0_abs_ns"

    trig_col = None
    if "trigger_time_ps" in df.columns:
        trig_col = "trigger_time_ps"
    elif "trigger_time_s" in df.columns:
        trig_col = "trigger_time_s"

    edge_col = "prev_rising_edge_abs_ps" if "prev_rising_edge_abs_ps" in df.columns else None

    seg_to_peak = {}
    for _, row in df.iterrows():
        try:
            seg = int(row["segment"])
        except Exception:
            continue
        peak_time_ps = row.get("peak_time_ps", np.nan)
        peak_amp = row.get("peak_amp", np.nan)
        t0_abs_ps = np.nan
        trigger_time_ps = np.nan
        edge_ps = np.nan

        if t0_col is not None:
            t0_abs_ps = row.get(t0_col, np.nan)
            if t0_col.endswith("_ns") and t0_abs_ps == t0_abs_ps:
                t0_abs_ps = float(t0_abs_ps) * 1000.0
        if trig_col is not None:
            trigger_time_ps = row.get(trig_col, np.nan)
            if trig_col.endswith("_s") and trigger_time_ps == trigger_time_ps:
                trigger_time_ps = float(trigger_time_ps) * 1e12
        if edge_col is not None:
            edge_ps = row.get(edge_col, np.nan)

        seg_to_peak[seg] = (
            float(peak_time_ps) if peak_time_ps == peak_time_ps else np.nan,
            float(peak_amp) if peak_amp == peak_amp else np.nan,
            float(t0_abs_ps) if t0_abs_ps == t0_abs_ps else np.nan,
            float(trigger_time_ps) if trigger_time_ps == trigger_time_ps else np.nan,
            float(edge_ps) if edge_ps == edge_ps else np.nan,
        )
    return seg_to_peak


def compute_phi(a, b):
    if a != a or b != b:
        return np.nan
    try:
        return float((a - b) % 6250.0)
    except Exception:
        return np.nan


def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def make_validation_plot(segments_data, out_path):
    """Create a 2-panel validation plot: scatter + linear fits | residual histogram + Gaussian."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors
    all_residuals = []

    for seg_i, (trig_ps, root_ps) in enumerate(segments_data):
        if len(trig_ps) < 2:
            continue
        color = colors[seg_i % len(colors)]
        ax1.scatter(trig_ps, root_ps, s=1, alpha=0.5, color=color, label=f"Seg {seg_i + 1}")

        # Linear fit
        coeffs = np.polyfit(trig_ps, root_ps, 1)
        fit_line = np.poly1d(coeffs)
        x_range = np.linspace(trig_ps.min(), trig_ps.max(), 100)
        ax1.plot(x_range, fit_line(x_range), color=color, linewidth=1.5,
                 label=f"Fit {seg_i + 1}: slope={coeffs[0]:.6f}")

        residuals = root_ps - fit_line(trig_ps)
        all_residuals.append(residuals)

    ax1.set_xlabel("Trigger time (ps)")
    ax1.set_ylabel("Channel 192 time (ps)")
    ax1.set_title("Ch192 time vs Trigger time")
    ax1.legend(fontsize=7, markerscale=5)

    # Residual histogram
    if all_residuals:
        all_res = np.concatenate(all_residuals)
        all_res = all_res[np.isfinite(all_res)]
        if len(all_res) > 10:
            n_bins = min(200, max(50, len(all_res) // 20))
            counts, bin_edges, _ = ax2.hist(all_res, bins=n_bins, alpha=0.7, color="steelblue",
                                            label=f"Residuals (N={len(all_res)})")
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            try:
                p0 = [counts.max(), np.mean(all_res), np.std(all_res)]
                popt, _ = curve_fit(_gauss, bin_centers, counts, p0=p0, maxfev=5000)
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
                ax2.plot(x_fit, _gauss(x_fit, *popt), "r-", linewidth=2,
                         label=f"Gaussian: μ={popt[1]:.1f}, σ={popt[2]:.1f}")
            except Exception as e:
                log(f"Gaussian fit failed: {e}")
            ax2.legend(fontsize=8)

    ax2.set_xlabel("Residual (ps)")
    ax2.set_ylabel("Counts")
    ax2.set_title("Fit residuals")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Validation plot saved: {out_path}")


def extract_channel_times_from_root(
    root_path,
    channel,
    branch_channel,
    branch_idx,
    branch_time,
    max_entries,
    fast=False,
    step_size=200000,
    require_trigger=True,
    trigger_channel=192,
):
    log(f"Opening ROOT file: {root_path}")
    f = uproot.open(root_path)
    tree_name = find_data_tree(f)
    if tree_name is None:
        raise RuntimeError("No data tree found in ROOT file.")
    tree = f[tree_name]
    keys = list(tree.keys())

    ch_branch = branch_channel if branch_channel else ("channelID" if "channelID" in keys else None)
    idx_branch = branch_idx if branch_idx else ("channelIdx" if "channelIdx" in keys else None)
    time_branch = branch_time if branch_time else ("time" if "time" in keys else None)
    t1c_branch = "t1coarse" if "t1coarse" in keys else None
    if not ch_branch or not idx_branch or not time_branch:
        raise RuntimeError(
            f"Missing required branches. Found channel={ch_branch}, idx={idx_branch}, time={time_branch}"
        )
    if t1c_branch:
        log(f"Found t1coarse branch, will read for dedup")

    n_entries = tree.num_entries
    use_n = n_entries if max_entries is None else min(max_entries, n_entries)
    log(
        f'Data tree="{tree_name}", total_entries={n_entries}, processing_entries={use_n}, '
        f'channel={channel}, branches=({ch_branch},{idx_branch},{time_branch}), fast={fast}'
    )

    times = []
    entries = []
    t1coarse_vals = []
    n_channels_list = []

    if not fast:
        read_branches = [ch_branch, idx_branch, time_branch]
        if t1c_branch:
            read_branches.append(t1c_branch)
        arrays = tree.arrays(read_branches, library="ak")
        for i in range(use_n):
            if i > 0 and i % 200000 == 0:
                log(f"Scanned ROOT entries: {i}/{use_n} (matched so far: {len(times)})")
            try:
                ev_ch = ak.to_list(arrays[ch_branch][i])
                ev_idx = ak.to_list(arrays[idx_branch][i])
                ev_time = ak.to_list(arrays[time_branch][i])
                ev_t1c = ak.to_list(arrays[t1c_branch][i]) if t1c_branch else None
            except Exception:
                continue

            try:
                if channel not in set(ev_ch):
                    continue
                if require_trigger and trigger_channel not in set(ev_ch):
                    continue
            except Exception:
                continue

            if not isinstance(ev_idx, list) or not isinstance(ev_time, list):
                continue
            if channel < 0 or channel >= len(ev_idx):
                continue

            mapped_idx = ev_idx[channel]
            try:
                mapped_idx = int(mapped_idx)
            except Exception:
                continue
            if mapped_idx < 0 or mapped_idx >= len(ev_time):
                continue

            try:
                t = float(ev_time[mapped_idx])
            except Exception:
                continue
            times.append(t)
            entries.append(i)
            if ev_t1c is not None and isinstance(ev_t1c, list) and mapped_idx < len(ev_t1c):
                t1coarse_vals.append(float(ev_t1c[mapped_idx]))
            else:
                t1coarse_vals.append(np.nan)
            n_channels_list.append(len(set(ev_ch)))
    else:
        entry_start = 0
        read_branches = [ch_branch, idx_branch, time_branch]
        if t1c_branch:
            read_branches.append(t1c_branch)
        while entry_start < use_n:
            entry_stop = min(entry_start + step_size, use_n)
            arrays = tree.arrays(
                read_branches,
                entry_start=entry_start,
                entry_stop=entry_stop,
                library="ak",
            )
            if entry_stop > 0 and entry_stop % 200000 == 0:
                log(f"Scanned ROOT entries: {entry_stop}/{use_n} (matched so far: {len(times)})")

            ch_list = arrays[ch_branch]
            idx_list = arrays[idx_branch]
            time_list = arrays[time_branch]
            t1c_list = arrays[t1c_branch] if t1c_branch else None

            mask = ak.any(ch_list == channel, axis=1)
            if require_trigger:
                mask = mask & ak.any(ch_list == trigger_channel, axis=1)

            if entry_stop % (step_size * 2) == 0:
                try:
                    n_has_ch = int(ak.sum(ak.any(ch_list == channel, axis=1)))
                    n_has_trig = int(ak.sum(ak.any(ch_list == trigger_channel, axis=1)))
                    log(f"Chunk [{entry_start},{entry_stop}) has channel={n_has_ch}, trigger={n_has_trig}, both={int(ak.sum(mask))}")
                except Exception:
                    pass

            if not ak.any(mask):
                entry_start = entry_stop
                continue

            ch_sel = ch_list[mask]
            idx_sel = idx_list[mask]
            time_sel = time_list[mask]
            t1c_sel = t1c_list[mask] if t1c_list is not None else None
            n_ch_per_event = ak.num(ch_sel, axis=1)

            # channelIdx is indexed by channel ID, not aligned with channelID list.
            # Extract mapped index by direct indexing into each channelIdx list.
            idx_sel_py = ak.to_list(idx_sel)
            mapped_idx = np.array(
                [row[channel] if isinstance(row, list) and channel < len(row) else -1 for row in idx_sel_py],
                dtype=int,
            )

            time_sel_py = ak.to_list(time_sel)
            t1c_sel_py = ak.to_list(t1c_sel) if t1c_sel is not None else None
            valid_idx = np.array(
                [0 <= mi < len(trow) if isinstance(trow, list) else False for mi, trow in zip(mapped_idx, time_sel_py)],
                dtype=bool,
            )
            if not ak.any(valid_idx):
                entry_start = entry_stop
                continue

            mapped_idx = mapped_idx[valid_idx]
            time_sel_py = [trow for trow, ok in zip(time_sel_py, valid_idx) if ok]
            tvals = np.array(
                [trow[i] if isinstance(trow, list) and i < len(trow) else np.nan for trow, i in zip(time_sel_py, mapped_idx)],
                dtype=float,
            )
            if t1c_sel_py is not None:
                t1c_sel_py_filt = [trow for trow, ok in zip(t1c_sel_py, valid_idx) if ok]
                t1c_vals_chunk = np.array(
                    [trow[i] if isinstance(trow, list) and i < len(trow) else np.nan for trow, i in zip(t1c_sel_py_filt, mapped_idx)],
                    dtype=float,
                )
            else:
                t1c_vals_chunk = np.full(len(tvals), np.nan)

            local_idx = np.arange(entry_start, entry_stop, dtype=int)
            local_idx = local_idx[ak.to_numpy(mask)][valid_idx]
            n_ch_arr = ak.to_numpy(n_ch_per_event)[valid_idx]

            for t, ev, tc, nch in zip(tvals, local_idx, t1c_vals_chunk, n_ch_arr):
                if t == t:
                    times.append(float(t))
                    entries.append(int(ev))
                    t1coarse_vals.append(float(tc))
                    n_channels_list.append(int(nch))

            entry_start = entry_stop

    log(f"Finished ROOT scan: matched channel-{channel} entries={len(times)}")
    return (
        np.array(times, dtype=float),
        np.array(entries, dtype=int),
        np.array(t1coarse_vals, dtype=float),
        np.array(n_channels_list, dtype=int),
        tree_name,
        use_n,
    )


def main():
    p = argparse.ArgumentParser(description="Fast FEBD sync + MCP attach in one step")
    p.add_argument("--root", required=True, help="Input ROOT file")
    p.add_argument("--meta-path", required=False, help="Meta CSV file, glob, or one file in a sibling set")
    p.add_argument("--meta-dir", default="../trc_out/", help="Directory containing meta CSVs (used if meta-path is not provided)")
    p.add_argument("--peaks-dir", required=True, help="Directory containing peaks CSVs")
    p.add_argument("--peaks-pattern", required=False, help="Peaks file pattern (glob) in peaks-dir")
    p.add_argument("--channel", type=int, default=192, help="Channel to synchronize (default: 192)")
    p.add_argument("--out-root", required=True, help="Output ROOT with MCP tree")
    p.add_argument("--out-matched-csv", required=True, help="Output matched events CSV")
    p.add_argument("--branch-channel", default="channelID", help="ROOT channel branch")
    p.add_argument("--branch-idx", default="channelIdx", help="ROOT channelIdx branch")
    p.add_argument("--branch-time", default="time", help="ROOT time branch")
    p.add_argument("--max-entries", type=int, default=None, help="Optional cap on ROOT entries")
    p.add_argument("--fast", action="store_true", help="Enable fast vectorized ROOT scan")
    p.add_argument("--step-size", type=int, default=200000, help="Chunk size for --fast mode")
    p.add_argument("--require-trigger", action="store_true", help="Require trigger channel 192 in event")
    p.add_argument("--gap-factor", type=float, default=500.0, help="Cluster gap factor")
    p.add_argument("--ratio-threshold", type=float, default=1.01, help="Shift detection threshold")
    p.add_argument("--dedup", action="store_true", help="Enable dedup of double-counted triggers using t1coarse")
    p.add_argument("--dedup-threshold-t1c", type=float, default=8.0,
                   help="t1coarse diff threshold for dedup (default: 8 clock cycles)")
    args = p.parse_args()
    log("Starting single-pass FEBD sync + MCP attach workflow")

    if not args.meta_path or not args.peaks_pattern:
        m = re.search(r'/([^/]+)/(\d+)_[^/]*\.root$', args.root)
        if m:
            try:
                run = int(m.group(1))
                spill = int(m.group(2))
            except ValueError:
                raise ValueError("Could not extract valid run and spill integers from --root path.")
        else:
            raise ValueError(
                f"Could not extract run and spill from --root path '{args.root}'. "
                "Please provide --meta-path and --peaks-pattern explicitly."
            )
        
        padded_run = f"{run:07d}"
        padded_spill = f"{spill:07d}"
        
        if not args.meta_path:
            args.meta_path = os.path.join(args.meta_dir, f"raw_C2_{padded_run}_{padded_spill}_*_meta.csv")
            log(f"Auto-inferred --meta-path: {args.meta_path}")
            
        if not args.peaks_pattern:
            args.peaks_pattern = f"peaks_raw_C1_{padded_run}_{padded_spill}_*_data_with_t0.csv"
            log(f"Auto-inferred --peaks-pattern: {args.peaks_pattern}")

    root_times, root_entries, root_t1coarse, root_n_channels, tree_name, used_entries = extract_channel_times_from_root(
        args.root,
        args.channel,
        args.branch_channel,
        args.branch_idx,
        args.branch_time,
        args.max_entries,
        fast=args.fast,
        step_size=args.step_size,
        require_trigger=args.require_trigger,
        trigger_channel=192,
    )
    if len(root_times) == 0:
        raise RuntimeError("No valid channel times extracted from ROOT.")

    # Deduplicate double-counted triggers using t1coarse
    if args.dedup and len(root_times) > 1:
        has_t1c = not np.all(np.isnan(root_t1coarse))
        if has_t1c:
            order = np.argsort(root_times)
            sorted_times = root_times[order]
            sorted_entries = root_entries[order]
            sorted_t1c = root_t1coarse[order]
            sorted_nch = root_n_channels[order]
            dt1c = np.abs(np.diff(sorted_t1c))

            # Step 1: Identify which events have a twin (t1coarse diff < threshold)
            has_twin = np.zeros(len(sorted_times), dtype=bool)
            i = 0
            while i < len(dt1c):
                if dt1c[i] < args.dedup_threshold_t1c:
                    has_twin[i] = True
                    has_twin[i + 1] = True
                    i += 2
                else:
                    i += 1

            # Step 2: Remove one from each twin pair (keep first, drop second)
            keep = np.ones(len(sorted_times), dtype=bool)
            i = 0
            while i < len(dt1c):
                if dt1c[i] < args.dedup_threshold_t1c:
                    keep[i + 1] = False  # drop the second (duplicate)
                    i += 2
                else:
                    i += 1

            # Step 3: Remove orphan fakes — events without a twin that have only ch 192
            # (n_channels == 1 means only channel 192 fired in that event)
            n_orphans = 0
            for idx in range(len(sorted_times)):
                if keep[idx] and not has_twin[idx] and sorted_nch[idx] <= 1:
                    keep[idx] = False
                    n_orphans += 1

            n_before = len(root_times)
            root_times = sorted_times[keep]
            root_entries = sorted_entries[keep]
            root_t1coarse = sorted_t1c[keep]
            root_n_channels = sorted_nch[keep]
            n_removed = n_before - len(root_times)
            n_twins_removed = n_removed - n_orphans
            if n_removed > 0:
                log(f"Dedup: removed {n_twins_removed} twin duplicates + {n_orphans} orphan fakes "
                    f"(t1coarse threshold={args.dedup_threshold_t1c})")
        else:
            log("Warning: --dedup requested but t1coarse branch not found, skipping dedup")

    log("Clustering ROOT channel times by large gaps")
    clusters, clusters_idx = split_by_time_gaps_with_indices(
        root_times, root_entries, args.gap_factor
    )
    if not clusters:
        raise RuntimeError("No clusters found from ROOT channel times.")
    log(f"Built {len(clusters)} cluster(s) from ROOT times")

    log(f"Resolving meta files from: {args.meta_path}")
    meta_files = resolve_meta_files(args.meta_path)
    if not meta_files:
        raise FileNotFoundError(f"No meta files found from --meta-path: {args.meta_path}")
    log(f"Resolved {len(meta_files)} meta file(s)")

    use_n = min(len(clusters), len(meta_files))
    if use_n == 0:
        raise RuntimeError("No usable segment pairing between ROOT clusters and meta files.")
    log(f"Processing {use_n} segment(s) (min of clusters/meta files)")

    all_indices = []
    all_peak_time = []
    all_peak_amp = []
    all_peak_phi = []
    all_peak_phi_from_edge = []
    all_trigger_time = []
    all_trigger_phi = []
    all_trigger_phi_from_edge = []
    matched_rows = []
    validation_segments = []  # list of (trigger_ps_array, root_time_ps_array) per segment

    for seg_idx in range(use_n):
        meta_file = meta_files[seg_idx]
        root_cluster = np.asarray(clusters[seg_idx], dtype=float)
        root_cluster_idx = np.asarray(clusters_idx[seg_idx], dtype=int)
        log(
            f"[segment {seg_idx + 1}/{use_n}] meta={Path(meta_file).name}, "
            f"cluster_size={len(root_cluster)}"
        )
        if len(root_cluster) < 2:
            log(f"[segment {seg_idx + 1}] skipped: cluster has <2 points")
            continue

        mnum = re.search(r"_(\d+(?:_\d+)+)_meta\.csv$", Path(meta_file).name)
        suffix = mnum.group(1) if mnum else f"seg{seg_idx + 1}"

        trigger_values = read_trigger_values(meta_file)
        trigger_ps = np.array(trigger_values, dtype=float) * 1e12
        log(f"[segment {seg_idx + 1}] trigger_count={len(trigger_ps)}")
        if len(trigger_ps) < 2:
            log(f"[segment {seg_idx + 1}] skipped: trigger list has <2 points")
            continue

        dt_trigger = np.diff(trigger_ps)
        dt_root = np.diff(root_cluster)
        (
            _missing_trig_idx,
            _missing_root_idx,
            root_to_trigger,
            trigger_to_root,
            _aligned_ratio,
        ) = find_missing_by_shift(dt_trigger, dt_root, args.ratio_threshold, 1e-13)

        peaks_file = find_peaks_file(args.peaks_dir, args.peaks_pattern, suffix)
        if peaks_file is None:
            raise FileNotFoundError(
                f'No peaks file matched for segment suffix "{suffix}" with pattern "{args.peaks_pattern}"'
            )
        log(f"[segment {seg_idx + 1}] peaks={Path(peaks_file).name}")
        seg_to_peak = load_peak_lookup(peaks_file)

        before_seg = len(all_indices)
        for j_trigger, i_root in enumerate(trigger_to_root):
            if i_root is None:
                continue
            if isinstance(i_root, float) and np.isnan(i_root):
                continue
            i_root_int = int(i_root)
            if i_root_int < 0 or i_root_int >= len(root_cluster):
                continue

            if 0 <= i_root_int < len(root_to_trigger):
                trig_idx = root_to_trigger[i_root_int]
            else:
                trig_idx = j_trigger
            if trig_idx is None or (isinstance(trig_idx, float) and np.isnan(trig_idx)):
                segment_num = j_trigger + 1
            else:
                segment_num = int(trig_idx) + 1

            peak_time_ps, peak_amp, t0_abs_ps, trigger_time_ps, prev_edge_ps = seg_to_peak.get(
                segment_num, (np.nan, np.nan, np.nan, np.nan, np.nan)
            )
            phi_peak = compute_phi(peak_time_ps, t0_abs_ps)
            phi_peak_edge = compute_phi(peak_time_ps, prev_edge_ps)
            phi_trigger = compute_phi(trigger_time_ps, t0_abs_ps)
            phi_trigger_edge = compute_phi(trigger_time_ps, prev_edge_ps)
            root_entry = int(root_cluster_idx[i_root_int])
            root_time_ps = float(root_cluster[i_root_int])
            mapped_trigger_ps = float(trigger_ps[j_trigger]) if j_trigger < len(trigger_ps) else np.nan

            all_indices.append(root_entry)
            all_peak_time.append(peak_time_ps)
            all_peak_amp.append(peak_amp)
            all_peak_phi.append(phi_peak)
            all_peak_phi_from_edge.append(phi_peak_edge)
            all_trigger_time.append(trigger_time_ps)
            all_trigger_phi.append(phi_trigger)
            all_trigger_phi_from_edge.append(phi_trigger_edge)

            matched_rows.append(
                {
                    "entry": root_entry,
                    "root_time_ps": root_time_ps,
                    "meta_file": str(Path(meta_file).name),
                    "peaks_file": str(Path(peaks_file).name),
                    "segment": int(segment_num),
                    "trigger_ps_from_meta": mapped_trigger_ps,
                    "trigger_time_ps_from_peaks": trigger_time_ps,
                    "peak_time_ps": peak_time_ps,
                    "peak_amp": peak_amp,
                    "phi_peak": phi_peak,
                    "phi_peak_from_edge": phi_peak_edge,
                    "phi_trigger": phi_trigger,
                    "phi_trigger_from_edge": phi_trigger_edge,
                }
            )
        log(f"[segment {seg_idx + 1}] matched_events_added={len(all_indices) - before_seg}")

        # Collect per-segment data for validation plot
        seg_trig_list = [r["trigger_ps_from_meta"] for r in matched_rows[before_seg:]]
        seg_root_list = [r["root_time_ps"] for r in matched_rows[before_seg:]]
        seg_trig_arr = np.array(seg_trig_list, dtype=float)
        seg_root_arr = np.array(seg_root_list, dtype=float)
        valid = np.isfinite(seg_trig_arr) & np.isfinite(seg_root_arr)
        validation_segments.append((seg_trig_arr[valid], seg_root_arr[valid]))

    out_root_path = Path(args.out_root)
    out_root_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Copying input ROOT -> output ROOT: {out_root_path}")
    shutil.copy2(args.root, out_root_path)

    log(f"Writing MCP tree with {len(all_indices)} entries")
    with uproot.update(str(out_root_path)) as f:
        f["MCP"] = {
            "index": np.array(all_indices, dtype=np.int64),
            "peak_time": np.array(all_peak_time, dtype=np.float64),
            "peak_amp": np.array(all_peak_amp, dtype=np.float64),
            "phi_peak": np.array(all_peak_phi, dtype=np.float64),
            "phi_peak_from_edge": np.array(all_peak_phi_from_edge, dtype=np.float64),
            "trigger_time": np.array(all_trigger_time, dtype=np.float64),
            "phi_trigger": np.array(all_trigger_phi, dtype=np.float64),
            "phi_trigger_from_edge": np.array(all_trigger_phi_from_edge, dtype=np.float64),
        }

    out_csv_path = Path(args.out_matched_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(matched_rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values(["entry", "segment"]).reset_index(drop=True)
    log(f"Writing matched CSV: {out_csv_path}")
    out_df.to_csv(out_csv_path, index=False)

    # Validation plot
    if validation_segments:
        plot_path = out_csv_path.with_name(out_csv_path.stem + "_validation.png")
        try:
            make_validation_plot(validation_segments, plot_path)
        except Exception as e:
            log(f"Warning: validation plot failed: {e}")

    log(f"Input ROOT: {args.root}")
    log(f'Data tree used: "{tree_name}" (processed entries: {used_entries})')
    log(f"Extracted channel-{args.channel} points: {len(root_times)}")
    log(f"Meta files: {len(meta_files)}, clusters: {len(clusters)}, processed segments: {use_n}")
    log(f"MCP entries written: {len(all_indices)}")
    log(f"Output ROOT: {out_root_path}")
    log(f"Matched CSV: {out_csv_path} (rows: {len(out_df)})")
    log("Workflow completed")


if __name__ == "__main__":
    main()
