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


def find_matches_by_absolute_alignment(trigger_ps, root_ps, ratio_threshold=2.0, tolerance_ps=50000):
    """
    Robust 1-to-1 Rolling Absolute Alignment.
    Locks pairs exactly sequentially, banning multiple-to-one matches (which cause 10^19 ratios).
    Maintains a smoothly responding polynomial fit over the last 100 pairs to track precise drift.
    """
    root_to_trigger = np.full(len(root_ps), np.nan)
    trigger_to_root = np.full(len(trigger_ps), np.nan)
    aligned_ratios = []

    if len(root_ps) == 0 or len(trigger_ps) == 0:
        return np.array([]), np.array([]), root_to_trigger, trigger_to_root, np.array([])
        
    dt_trigger = np.diff(trigger_ps)
    dt_root = np.diff(root_ps)
    
    valid_dt_t = dt_trigger[dt_trigger > 0]
    valid_dt_r = dt_root[dt_root > 0]
    med_dt_trigger = np.median(valid_dt_t) if len(valid_dt_t) > 0 else 1.0
    med_dt_root = np.median(valid_dt_r) if len(valid_dt_r) > 0 else 1.0
    
    K_global = med_dt_root / med_dt_trigger
    
    matched_t = []
    matched_r = []
    
    # 1. Establish initial offset on the first physical overlap
    best_offset = 0
    max_matches = -1
    for shift in range(min(50, len(trigger_ps))):
        test_offset = root_ps[0] - K_global * trigger_ps[shift]
        expect = K_global * trigger_ps[shift:shift+50] + test_offset
        limit = min(50, len(root_ps))
        diffs = np.abs(root_ps[:limit, None] - expect[None, :])
        matches = np.sum(np.min(diffs, axis=1) < med_dt_root * 0.4)
        if matches > max_matches:
            max_matches = matches
            best_offset = test_offset
            
    current_K = K_global
    current_Offset = best_offset
    
    search_j = 0
    force_tol = max(tolerance_ps, med_dt_root * 0.4) 
    
    # Rolling 1-to-1 matcher
    for i, r_val in enumerate(root_ps):
        if search_j >= len(trigger_ps):
            break
            
        # Expected trigger time
        t_exp = (r_val - current_Offset) / current_K
        
        # Look ahead for the closest trigger in a reasonable window
        # We strictly enforce j >= search_j (1-to-1 monotonic mapping)
        j_max = min(len(trigger_ps), search_j + 50)
        
        sub_dist = np.abs(trigger_ps[search_j:j_max] - t_exp)
        best_local_j = np.argmin(sub_dist)
        actual_j = search_j + best_local_j
        
        # Distance physically back in ROOT domain
        mapped_r_val = current_K * trigger_ps[actual_j] + current_Offset
        dist = np.abs(r_val - mapped_r_val)
        
        if dist < force_tol:
            root_to_trigger[i] = actual_j
            trigger_to_root[actual_j] = i
            matched_t.append(trigger_ps[actual_j])
            matched_r.append(r_val)
            
            # Since matches MUST be 1-to-1 sequential, next root event CANNOT map to same trigger
            search_j = actual_j + 1
            
            # Update local drift model (rolling window of 100 matches)
            if len(matched_t) >= 10:
                calc_len = min(100, len(matched_t))
                xs = np.array(matched_t[-calc_len:])
                ys = np.array(matched_r[-calc_len:])
                # Fit quality safeguard: only update if ≥70% of window has valid pairs
                valid_pairs = np.sum(np.isfinite(xs) & np.isfinite(ys))
                if valid_pairs < 0.7 * calc_len:
                    pass  # keep previous model
                elif xs[-1] > xs[0]: # ensure physical distance spread
                    coeffs = np.polyfit(xs, ys, 1)
                    # Limit catastrophic drift spikes from noise
                    if 0.5 * K_global < coeffs[0] < 2.0 * K_global:
                        current_K = coeffs[0]
                        current_Offset = coeffs[1]

    # # Retroactive Pass: Re-match the first 100 events using the fully stabilized final K and Offset
    # limit_back = min(100, len(root_ps))
    # for i in range(limit_back):
    #     r_val = root_ps[i]
        
    #     # We only want to re-evaluate events that either missed entirely, 
    #     # or had a dubious match before the model stabilized.
    #     # Clear out whatever might be there.
    #     old_j = root_to_trigger[i]
    #     if not np.isnan(old_j):
    #         trigger_to_root[int(old_j)] = np.nan
            
    #     root_to_trigger[i] = np.nan
        
    #     t_exp = (r_val - current_Offset) / current_K
        
    #     # Scan over the very beginning of the trigger array
    #     # We don't have search_j bounds here because it's a retroactive sweep
    #     # over a small, known region (first ~200 triggers)
    #     j_max_back = min(len(trigger_ps), 200)
    #     sub_dist = np.abs(trigger_ps[:j_max_back] - t_exp)
    #     best_local_j = np.argmin(sub_dist)
        
    #     mapped_r_val = current_K * trigger_ps[best_local_j] + current_Offset
    #     dist = np.abs(r_val - mapped_r_val)
        
    #     # If the highly stable model finds a match, and nothing else recently stole it:
    #     if dist < force_tol and np.isnan(trigger_to_root[best_local_j]):
    #         root_to_trigger[i] = best_local_j
    #         trigger_to_root[best_local_j] = i

    # Generate aligned ratios purely to validate intervals
    for i in range(len(root_ps) - 1):
        j = root_to_trigger[i]
        j_next = root_to_trigger[i+1]
        
        if not np.isnan(j) and not np.isnan(j_next):
            dt_t_local = trigger_ps[int(j_next)] - trigger_ps[int(j)]
            if dt_t_local > 0:
                aligned_ratios.append((root_ps[i+1]-root_ps[i]) / dt_t_local)
            else:
                aligned_ratios.append(np.nan)
        else:
            aligned_ratios.append(np.nan)

    return (
        np.array([]), 
        np.array([]), 
        root_to_trigger,
        trigger_to_root,
        np.array(aligned_ratios),
    )


def interval_to_event_mapping(root_to_trigger, trigger_to_root, n_root_events, n_trigger_events):
    """Convert interval-index mappings (from np.diff arrays) into event-index mappings.

    Inputs:
      root_to_trigger: length (n_root_events - 1), maps ROOT interval i -> trigger interval j
      trigger_to_root: length (n_trigger_events - 1), maps trigger interval j -> ROOT interval i
    Returns:
      root_event_to_trigger_event: length n_root_events, event index mapping
      trigger_event_to_root_event: length n_trigger_events, event index mapping
    """
    root_event_to_trigger_event = np.full(n_root_events, np.nan)
    trigger_event_to_root_event = np.full(n_trigger_events, np.nan)

    # Anchor first events when both streams exist.
    if n_root_events > 0 and n_trigger_events > 0:
        root_event_to_trigger_event[0] = 0
        trigger_event_to_root_event[0] = 0

    # ROOT interval i corresponds to ROOT event (i + 1).
    # Trigger interval j corresponds to trigger event (j + 1).
    for i_root_int, j_trigger_int in enumerate(root_to_trigger):
        if j_trigger_int is None:
            continue
        if isinstance(j_trigger_int, float) and np.isnan(j_trigger_int):
            continue
        r_ev = int(i_root_int) + 1
        t_ev = int(j_trigger_int) + 1
        if 0 <= r_ev < n_root_events and 0 <= t_ev < n_trigger_events:
            root_event_to_trigger_event[r_ev] = t_ev
            if np.isnan(trigger_event_to_root_event[t_ev]):
                trigger_event_to_root_event[t_ev] = r_ev

    # Use reverse interval mapping to fill any still-unmapped trigger events.
    for j_trigger_int, i_root_int in enumerate(trigger_to_root):
        if i_root_int is None:
            continue
        if isinstance(i_root_int, float) and np.isnan(i_root_int):
            continue
        t_ev = int(j_trigger_int) + 1
        r_ev = int(i_root_int) + 1
        if 0 <= t_ev < n_trigger_events and 0 <= r_ev < n_root_events:
            if np.isnan(trigger_event_to_root_event[t_ev]):
                trigger_event_to_root_event[t_ev] = r_ev
            if np.isnan(root_event_to_trigger_event[r_ev]):
                root_event_to_trigger_event[r_ev] = t_ev

    return root_event_to_trigger_event, trigger_event_to_root_event


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

    if "trigger_time_ps" in df.columns:
        trig_col = "trigger_time_ps"

    edge_col = "prev_rising_edge_abs_ps" if "prev_rising_edge_abs_ps" in df.columns else None

    seg_to_peak = {}
    for _, row in df.iterrows():
        try:
            seg = int(row["segment"])
        except Exception:
            continue
        peak_time_ps = row.get("peak_time_ps", np.nan)
        peak_amp = row.get("peak_amp", np.nan)
        peak_sigma_ps = row.get("peak_sigma_ps", np.nan)
        t0_abs_ps = np.nan
        trigger_time_ps = np.nan
        trigger_offset_ps = row.get("trigger_offset_ps", np.nan)
        edge_ps = np.nan

        if t0_col is not None:
            t0_abs_ps = row.get(t0_col, np.nan)
        if trig_col is not None:
            trigger_time_ps = row.get(trig_col, np.nan)
        if edge_col is not None:
            edge_ps = row.get(edge_col, np.nan)

        seg_to_peak[seg] = (
            float(peak_time_ps) if peak_time_ps == peak_time_ps else np.nan,
            float(peak_amp) if peak_amp == peak_amp else np.nan,
            float(peak_sigma_ps) if peak_sigma_ps == peak_sigma_ps else np.nan,
            float(t0_abs_ps) if t0_abs_ps == t0_abs_ps else np.nan,
            float(trigger_time_ps) if trigger_time_ps == trigger_time_ps else np.nan,
            float(trigger_offset_ps) if trigger_offset_ps == trigger_offset_ps else np.nan,
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
        if len(all_res) == 0:
            return
        
        # Pure median-absolute-deviation pre-filter to crush all insane billion-PS spikes mathematically prior to standard deviation
        med = np.median(all_res)
        # Cap to a generous physical window (e.g. 500ns, any residual bigger than this is not physics)
        clean_all_res = all_res[np.abs(all_res - med) < 500000]
        
        if len(clean_all_res) > 10:
            mu_clean = np.mean(clean_all_res)
            sig_clean = np.std(clean_all_res)
            limit = max(4 * sig_clean, 100)
            clean_all_res = clean_all_res[np.abs(clean_all_res - mu_clean) < limit]
        
        if len(clean_all_res) == 0:
            clean_all_res = all_res # Fail-safe

        n_bins = min(200, max(50, len(clean_all_res) // 20))
        counts, bin_edges, _ = ax2.hist(clean_all_res, bins=n_bins, alpha=0.7, color="steelblue",
                                        label=f"Residuals (N={len(clean_all_res)})")
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        try:
            p0 = [counts.max(), np.mean(clean_all_res), np.std(clean_all_res)]
            popt, _ = curve_fit(_gauss, bin_centers, counts, p0=p0, maxfev=5000)
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
            ax2.plot(x_fit, _gauss(x_fit, *popt), "r-", linewidth=2,
                     label=f"Gaussian: μ={popt[1]:.1f}, σ={abs(popt[2]):.1f}")
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


def make_ratio_plot(ratio_data, out_path, max_events=None):
    """Create a multi-panel scatter plot of the delta_T ratio (root/trigger) for each segment."""
    n_seg = len(ratio_data)
    if n_seg == 0:
        return
        
    fig, axes = plt.subplots(n_seg, 1, figsize=(7, 4 * n_seg))
    if n_seg == 1:
        axes = [axes]
        
    for (seg_idx, ratios), ax in zip(ratio_data, axes):
        if max_events is not None:
            ratios = ratios[:max_events]
            
        # Clean out NaNs from unmatched ratios so plot correctly scales purely on matched pairs
        valid_mask = ~np.isnan(ratios)
        if not np.any(valid_mask):
            continue
            
        clean_ratios = ratios[valid_mask]
        indices = np.arange(len(clean_ratios))
        
        # keep only strictly positive ratios for log scale
        pos_mask = clean_ratios > 0
        if not np.any(pos_mask):
            continue
            
        ax.scatter(indices[pos_mask], clean_ratios[pos_mask], s=2, alpha=0.5, color='purple')
        
        ax.set_yscale('log')
        ax.set_xlabel("Event Index N (within segment)")
        ax.set_ylabel("$\\Delta T_{root} / \\Delta T_{trigger}$")
        title_suffix = f"(Segment {seg_idx + 1}/{n_seg})" if n_seg > 1 else ""
        ax.set_title(f"Time Diff Ratio {title_suffix}".strip())
        ax.grid(True, alpha=0.3)
        # add a red line at ratio = 1 for reference
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Ratio validation plot saved: {out_path}")


def make_sigma_amp_plot(df, out_path):
    """Create a 2D scatter plot of peak_sigma_ps vs peak_amp."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # filter out nans
    valid = np.isfinite(df['mcp_peak_sigma_ps']) & np.isfinite(df['mcp_peak_amp'])
    df_valid = df[valid]

    if len(df_valid) == 0:
        log("Warning: No valid data for sigma vs amp plot.")
        return
    
    # Use hexbin for better performance with many points
    hb = ax.hexbin(df_valid['mcp_peak_amp'], df_valid['mcp_peak_sigma_ps'], gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hb, ax=ax, label='Counts')
    
    ax.set_xlabel("MCP Peak Amplitude (V)")
    ax.set_ylabel("MCP Peak Sigma (ps)")
    ax.set_title("Peak Sigma vs Peak Amplitude")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"Sigma vs Amp validation plot saved: {out_path}")


def extract_channel_times_from_root(
    root_path,
    channel,
    branch_channel,
    branch_t1coarse,
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
    idx_branch = "channelIdx" if "channelIdx" in keys else None
    time_branch = branch_time if branch_time else ("time" if "time" in keys else None)
    t1c_branch = branch_t1coarse if branch_t1coarse else ("t1coarse" if "t1coarse" in keys else None)
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
    p.add_argument("--branch-t1coarse", default="t1coarse", help="ROOT t1coarse branch")
    p.add_argument("--branch-time", default="time", help="ROOT time branch")
    p.add_argument("--branch-energy", default="energy", help="ROOT energy branch")
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
        args.branch_t1coarse,
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
    all_peak_sigma = []
    all_peak_phi = []
    all_peak_phi_from_edge = []
    all_trigger_time = []
    all_trigger_offset = []
    all_trigger_phi = []
    all_trigger_phi_from_edge = []
    all_t0_abs = []
    matched_rows = []
    validation_segments = []  # list of (trigger_ps_array, root_time_ps_array) per segment
    validation_ratio_segments = []  # list of (seg_idx, ratios_array) per segment

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
            aligned_ratio,
        ) = find_matches_by_absolute_alignment(trigger_ps, root_cluster, args.ratio_threshold)

        # check delta_T ratio
        if len(dt_trigger) > 0 and len(dt_root) > 0:
            min_len = min(len(dt_trigger), len(dt_root))
            valid_mask = (dt_trigger[:min_len] != 0) & (~np.isnan(dt_root[:min_len]))
            if np.any(valid_mask):
                ratios = dt_root[:min_len][valid_mask] / dt_trigger[:min_len][valid_mask]
                # Log true performance mapping using only matched data points
                median_ratio = np.nanmedian(aligned_ratio)
                mean_ratio = np.nanmean(aligned_ratio)
                log(f"[segment {seg_idx + 1}] delta_T ratio (root/trigger) -> median: {median_ratio:.6f}, mean: {mean_ratio:.6f}")
                validation_ratio_segments.append((seg_idx, aligned_ratio))

        peaks_file = find_peaks_file(args.peaks_dir, args.peaks_pattern, suffix)
        if peaks_file is None:
            raise FileNotFoundError(
                f'No peaks file matched for segment suffix "{suffix}" with pattern "{args.peaks_pattern}"'
            )
        log(f"[segment {seg_idx + 1}] peaks={Path(peaks_file).name}")
        seg_to_peak = load_peak_lookup(peaks_file)

        n_map_pairs = int(np.sum(np.isfinite(root_to_trigger)))
        log(
            f"[segment {seg_idx + 1}] interval_map: root_int={len(root_to_trigger)}, "
            f"trig_int={len(trigger_to_root)}; event_map_pairs={n_map_pairs}"
        )

        before_seg = len(all_indices)
        seg_match_trigger = []
        seg_match_root = []
        for i_root_int in range(len(root_cluster)):
            j_trigger_event = root_to_trigger[i_root_int]
            root_entry = int(root_cluster_idx[i_root_int])
            root_time_ps = float(root_cluster[i_root_int])

            if np.isnan(j_trigger_event):
                # ROOT event successfully retained but MCP values are missing
                segment_num = -1
                peak_time_ps, peak_amp, peak_sigma_ps, t0_abs_ps, trigger_time_ps, trigger_offset_ps, prev_edge_ps = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                phi_peak = phi_peak_edge = phi_trigger = phi_trigger_edge = np.nan
                mapped_trigger_ps = np.nan
            else:
                j_trig = int(j_trigger_event)
                segment_num = j_trig + 1 # Peaks CSV uses 1-based event numbering
                peak_time_ps, peak_amp, peak_sigma_ps, t0_abs_ps, trigger_time_ps, trigger_offset_ps, prev_edge_ps = seg_to_peak.get(
                    segment_num, (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                )
                phi_peak = compute_phi(peak_time_ps, t0_abs_ps)
                phi_peak_edge = compute_phi(peak_time_ps, prev_edge_ps)
                phi_trigger = compute_phi(trigger_time_ps, t0_abs_ps)
                phi_trigger_edge = compute_phi(trigger_time_ps, prev_edge_ps)
                mapped_trigger_ps = float(trigger_ps[j_trig])

            if not np.isnan(mapped_trigger_ps):
                seg_match_trigger.append(mapped_trigger_ps)
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
            all_t0_abs.append(t0_abs_ps)

            matched_rows.append(
                {
                    "entry": root_entry,
                    "mcp_index": root_entry,
                    "channelID": "",
                    "t1coarse": "",
                    "time": "",
                    "energy": "",
                    "mcp_peak_time": peak_time_ps,
                    "mcp_peak_amp": peak_amp,
                    "mcp_peak_sigma_ps": peak_sigma_ps,
                    "mcp_peak_phase": phi_peak,
                    "mcp_trigger_time": trigger_time_ps,
                    "mcp_trigger_offset_ps": trigger_offset_ps,
                    "root_time_ps": root_time_ps,
                    "meta_file": str(Path(meta_file).name),
                    "peaks_file": str(Path(peaks_file).name),
                    "segment": int(segment_num),
                    "trigger_ps_from_meta": mapped_trigger_ps,
                    "phi_peak_from_edge": phi_peak_edge,
                    "phi_trigger_from_edge": phi_trigger_edge,
                    "t0_abs_ps": t0_abs_ps,
                    "prev_edge_ps": prev_edge_ps,
                    "phi_peak_from_trigger": compute_phi(peak_time_ps, trigger_time_ps),
                    "peak_minus_t0_ps": (peak_time_ps - t0_abs_ps) if (peak_time_ps == peak_time_ps and t0_abs_ps == t0_abs_ps) else np.nan,
                    "peak_minus_prev_edge_ps": (peak_time_ps - prev_edge_ps) if (peak_time_ps == peak_time_ps and prev_edge_ps == prev_edge_ps) else np.nan,
                    "trigger_minus_t0_ps": (trigger_time_ps - t0_abs_ps) if (trigger_time_ps == trigger_time_ps and t0_abs_ps == t0_abs_ps) else np.nan,
                }
            )
        if len(seg_match_trigger) >= 3:
            t_arr = np.asarray(seg_match_trigger, dtype=float)
            r_arr = np.asarray(seg_match_root, dtype=float)
            valid = np.isfinite(t_arr) & np.isfinite(r_arr)
            if np.sum(valid) >= 3:
                try:
                    a, b = np.polyfit(t_arr[valid], r_arr[valid], 1)
                    resid = r_arr[valid] - (a * t_arr[valid] + b)
                    log(
                        f"[segment {seg_idx + 1}] match_fit slope={a:.9f}, "
                        f"offset={b:.3f}, resid_rms={np.std(resid):.3f} ps"
                    )
                except Exception:
                    pass
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
            "peak_sigma_ps": np.array(all_peak_sigma, dtype=np.float64),
            "phi_peak": np.array(all_peak_phi, dtype=np.float64),
            "phi_peak_from_edge": np.array(all_peak_phi_from_edge, dtype=np.float64),
            "trigger_time": np.array(all_trigger_time, dtype=np.float64),
            "trigger_offset_ps": np.array(all_trigger_offset, dtype=np.float64),
            "phi_trigger": np.array(all_trigger_phi, dtype=np.float64),
            "phi_trigger_from_edge": np.array(all_trigger_phi_from_edge, dtype=np.float64),
            "t0_abs_ps": np.array(all_t0_abs, dtype=np.float64),
        }

    out_csv_path = Path(args.out_matched_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Fetch full arrays for matched events to match mcp_validation_dump format ---
    log("Fetching full JSON arrays for matched events...")
    ch_dumps, t1c_dumps, time_dumps, ener_dumps = {}, {}, {}, {}
    with uproot.open(args.root) as f_in:
        tree_in = f_in[tree_name]
        keys_in = list(tree_in.keys())
        local_ch = args.branch_channel if args.branch_channel else ("channelID" if "channelID" in keys_in else None)
        local_t1c = args.branch_t1coarse if getattr(args, "branch_t1coarse", None) else ("t1coarse" if "t1coarse" in keys_in else None)
        local_time = args.branch_time if args.branch_time else ("time" if "time" in keys_in else None)
        local_ener = args.branch_energy if getattr(args, "branch_energy", None) else ("energy" if "energy" in keys_in else None)

        read_cols = []
        if local_ch and local_ch in keys_in: read_cols.append(local_ch)
        if local_t1c and local_t1c in keys_in: read_cols.append(local_t1c)
        if local_time and local_time in keys_in: read_cols.append(local_time)
        if local_ener and local_ener in keys_in: read_cols.append(local_ener)

        entry_start = 0
        n_entries = tree_in.num_entries
        while entry_start < n_entries:
            entry_stop = min(entry_start + args.step_size, n_entries)
            chunk_matches = [e for e in all_indices if entry_start <= e < entry_stop]
            if chunk_matches:
                arrs = tree_in.arrays(read_cols, entry_start=entry_start, entry_stop=entry_stop, library="ak")
                for ev in chunk_matches:
                    local_i = ev - entry_start
                    try: ch_arr = ak.to_list(arrs[local_ch][local_i]) if local_ch in read_cols else []
                    except: ch_arr = []
                    try: t1c_arr = ak.to_list(arrs[local_t1c][local_i]) if local_t1c in read_cols else []
                    except: t1c_arr = []
                    try: time_arr = ak.to_list(arrs[local_time][local_i]) if local_time in read_cols else []
                    except: time_arr = []
                    try: ener_arr = ak.to_list(arrs[local_ener][local_i]) if local_ener in read_cols else []
                    except: ener_arr = []

                    ch_dumps[ev] = json.dumps(ch_arr)
                    t1c_dumps[ev] = json.dumps(t1c_arr)
                    time_dumps[ev] = json.dumps(time_arr)
                    ener_dumps[ev] = json.dumps(ener_arr)
            entry_start = entry_stop

    for row in matched_rows:
        ev = row["entry"]
        row["channelID"] = ch_dumps.get(ev, "[]")
        row["t1coarse"] = t1c_dumps.get(ev, "[]")
        row["time"] = time_dumps.get(ev, "[]")
        row["energy"] = ener_dumps.get(ev, "[]")

    out_df = pd.DataFrame(matched_rows)
    if len(out_df) > 0:
        out_df = out_df.sort_values(["entry", "segment"]).reset_index(drop=True)
        std_cols = ['entry', 'channelID', 't1coarse', 'time', 'energy', 'mcp_index', 'mcp_peak_time', 'mcp_peak_amp', 'mcp_peak_sigma_ps', 'mcp_peak_phase', 'mcp_trigger_time', 'mcp_trigger_offset_ps', 't0_abs_ps']
        extra_cols = ['root_time_ps', 'meta_file', 'peaks_file', 'segment', 'trigger_ps_from_meta', 'phi_peak_from_edge', 'phi_trigger_from_edge']
        cols = std_cols + [c for c in out_df.columns if c not in std_cols and c not in extra_cols] + extra_cols
        out_df = out_df[cols]

    log(f"Writing matched CSV: {out_csv_path}")
    out_df.to_csv(out_csv_path, index=False)

    # Validation plot
    if validation_segments:
        plot_path = out_csv_path.with_name(out_csv_path.stem + "_validation.png")
        try:
            make_validation_plot(validation_segments, plot_path)
        except Exception as e:
            log(f"Warning: validation plot failed: {e}")

    # Ratio plot
    if validation_ratio_segments:
        ratio_plot_path = out_csv_path.with_name(out_csv_path.stem + "_ratio.png")
        try:
            make_ratio_plot(validation_ratio_segments, ratio_plot_path)
            
            # Additional debug plot: First 100 events only
            ratio_plot_100_path = out_csv_path.with_name(out_csv_path.stem + "_ratio_first100.png")
            make_ratio_plot(validation_ratio_segments, ratio_plot_100_path, max_events=100)
        except Exception as e:
            log(f"Warning: ratio validation plot failed: {e}")

    # Sigma vs Amp validation plot
    if len(out_df) > 0 and 'mcp_peak_sigma_ps' in out_df.columns and 'mcp_peak_amp' in out_df.columns:
        sigma_amp_plot_path = out_csv_path.with_name(out_csv_path.stem + "_sigma_vs_amp.png")
        try:
            make_sigma_amp_plot(out_df, sigma_amp_plot_path)
        except Exception as e:
            log(f"Warning: sigma vs amp plot failed: {e}")

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
