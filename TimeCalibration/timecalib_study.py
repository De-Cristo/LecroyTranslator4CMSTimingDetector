#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Plot channel time vs MCP peak_time.

Example:
  python3 timecalib_plots.py input.root --channels 133
  python3 timecalib_plots.py input.root --channels 133 137 --plot-channel 133
"""

import argparse
import sys
import os
import math
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

print("[info] module loaded", file=sys.stderr, flush=True)

try:
    import uproot
    import awkward as ak
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
except Exception as e:
    print("Missing Python dependency:", e)
    print("Install: pip install uproot awkward numpy matplotlib scipy")
    sys.exit(1)

# Mapping for lyso bars / modules -> channel IDs
from channel_mapping import (
    lyso_bar_to_channels_lr,
    UP_MODULE_BASE,
    DOWN_MODULE_BASE,
    TRIGGER_CHANNEL,
)

# Mapping for lyso bars / modules -> channel IDs
from channel_mapping import (
    lyso_bar_to_channels_lr,
    UP_MODULE_BASE,
    DOWN_MODULE_BASE,
    TRIGGER_CHANNEL,
)


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


def kmeans_1d(x, k=3, iters=50):
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    centers = np.linspace(xmin, xmax, k)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        for i, xv in enumerate(x):
            labels[i] = int(np.argmin(np.abs(centers - xv)))
        new_centers = np.array([x[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def parse_args():
    p = argparse.ArgumentParser(description="TimeCalibration plots from ROOT input")
    p.add_argument("file", nargs="+", help="Input ROOT file(s) or glob pattern(s)")
    # Trigger channel is fixed to TRIGGER_CHANNEL from channel_mapping
    p.add_argument("--module", choices=["up", "down"], help="Detector module to use (up/down)")
    p.add_argument("--lyso-bar", type=int, help="Lyso bar index (0-15)")
    p.add_argument("--side", choices=["L", "R", "both"], help="Lyso bar side (L/R/both)")
    p.add_argument("--second-module", choices=["up", "down"], help="Second detector module (up/down)")
    p.add_argument("--second-lyso-bar", type=int, help="Second lyso bar index (0-15)")
    p.add_argument("--second-side", choices=["L", "R", "both"], help="Second lyso bar side (L/R/both)")
    p.add_argument("--branch-idx", default="channelID")
    p.add_argument("--branch-time", default="time")
    p.add_argument("--branch-energy", default="energy")
    p.add_argument("--mcp-tree", default="MCP")
    p.add_argument("--mcp-index", default="index")
    p.add_argument("--mcp-peak-time", default="peak_time")
    p.add_argument("--mcp-peak-amp", default="peak_amp")
    p.add_argument("--mcp-trigger-time", default="trigger_time")
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--time-peak-lines", type=int, default=3,
                   help="Number of linear segments to fit for ch192 vs trigger (default: 3)")
    p.add_argument("--out-ch192-vs-trigger", default="ch192_vs_mcp_trigger.png",
                   help="Output plot filename for channel 192 time vs MCP trigger_time (with linear fit)")
    p.add_argument("--out-raw-time-diff", default="raw_time_diff.png",
                   help="Output plot filename for raw time difference (t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)")
    p.add_argument("--out-raw-time-scatter", default="raw_time_scatter.png",
                   help="Output plot filename for scatter of (t_bar - t_192) vs (t_mcp_peak - t_mcp_trigger)")
    p.add_argument("--out-raw-time-residual", default="raw_time_residual.png",
                   help="Output plot filename for residuals of the linear fit")
    p.add_argument("--out-raw-time-scatter-clean", default="raw_time_scatter_clean.png",
                   help="Output scatter plot filename with outliers removed")
    p.add_argument("--out-raw-time-residual-clean", default="raw_time_residual_clean.png",
                   help="Output residual plot filename with outliers removed")
    p.add_argument("--out-time-vs-mcp-peak", default="time_vs_mcp_peak.png",
                   help="Output plot filename for TOFHIR time vs MCP peak_time scatter plot")
    p.add_argument("--out-mcp-peak-time-vs-amp", default="mcp_peak_time_vs_amp.png",
                   help="Output plot filename for MCP peak_time vs peak_amp scatter plot")
    p.add_argument("--out-walk-mcp", default="walk_mcp_time.png",
                   help="Output scatter plot filename for (mcp_peak - mcp_trigger) vs 1/peak_amp")
    p.add_argument("--out-walk-mcp-res", default="walk_mcp_time_res.png",
                   help="Output residual plot filename for MCP time walk fit")
    p.add_argument("--out-walk-bar", default="walk_bar_time.png",
                   help="Output scatter plot filename for (bar_time - ch192) vs 1/peak_amp")
    p.add_argument("--out-walk-bar-res", default="walk_bar_time_res.png",
                   help="Output residual plot filename for detector time walk fit")
    p.add_argument("--out-t-diff-first", default="t_diff_first.png",
                   help="Output plot filename for time_left - time_right (first bar)")
    p.add_argument("--out-t-diff-second", default="t_diff_second.png",
                   help="Output plot filename for time_left - time_right (second bar)")
    p.add_argument("--out-energy", default="energy_plot_channel.png",
                   help="Output plot filename for energy histogram of --plot-channel")
    p.add_argument("--out-energy-second", default="energy_plot_second.png",
                   help="Output plot filename for energy histogram of second detector")
    p.add_argument("--energy-bins", type=int, default=120,
                   help="Histogram bins for energy plot (default: 120)")
    p.add_argument("--energy-min", type=float, default=None,
                   help="Lower bound energy cut for plot-channel (applies to all plots)")
    p.add_argument("--energy-max", type=float, default=None,
                   help="Upper bound energy cut for plot-channel (applies to all plots)")
    p.add_argument("--second-energy-min", type=float, default=None,
                   help="Lower bound energy cut for second detector (if provided)")
    p.add_argument("--second-energy-max", type=float, default=None,
                   help="Upper bound energy cut for second detector (if provided)")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of worker processes for multi-file processing (default: 1)")
    p.add_argument("--skip-ch192-plot", action="store_true",
                   help="Skip preparing and saving per-file ch192 vs trigger plots")
    p.add_argument("--verbose", action="store_true",
                   help="Print debug counters")
    return p.parse_args()


def process_file(path, cfg):
    """Process one ROOT file and return per-file accumulators."""
    out = {
        "path": path,
        "x_ch192": [],
        "y_trig": [],
        "time_vs_mcp_peak_x": [],  # TOFHIR time
        "time_vs_mcp_peak_y": [],  # MCP peak_time
        "mcp_peak_time_vs_amp_x": [],  # MCP peak_time
        "mcp_peak_time_vs_amp_y": [],  # MCP peak_amp
        "raw_time_diff": [], # Raw time diff: (t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)
        "raw_det_diff": [], # (t_bar - t_192)
        "raw_mcp_diff": [], # (t_mcp_peak - t_mcp_trigger)
        "raw_peak_amp": [], # peak_amp for time walk
        "t_diff_first": [], # time_left - time_right (first bar)
        "t_diff_second": [], # time_left - time_right (second bar)
        "energy_vals": [],
        "energy_vals_second": [],
        "counters": {
            "total": 0,
            "missing_ch": 0,
            "missing_time": 0,
            "missing_plot_channel": 0,
            "missing_mcp": 0,
            "energy_cut": 0,
            "second_energy_cut": 0,
            "kept": 0,
            "missing_mcp_trigger": 0,
        },
    }

    f = uproot.open(path)
    tree_name = find_data_tree(f)
    if tree_name is None:
        out["counters"]["missing_mcp"] += 1
        return out
    tree = f[tree_name]

    if cfg["mcp_tree"] not in f:
        out["counters"]["missing_mcp"] += 1
        return out
    mcp = f[cfg["mcp_tree"]]
    mcp_idx = mcp[cfg["mcp_index"]].array(library="np")
    mcp_pt = mcp[cfg["mcp_peak_time"]].array(library="np")
    mcp_amp = mcp[cfg["mcp_peak_amp"]].array(library="np")
    mcp_tt = mcp[cfg["mcp_trigger_time"]].array(library="np")
    mcp_map = {}
    for i in range(len(mcp_idx)):
        try:
            idx = int(mcp_idx[i])
        except Exception:
            continue
        try:
            pt = float(mcp_pt[i])
        except Exception:
            pt = math.nan
        try:
            amp = float(mcp_amp[i])
        except Exception:
            amp = math.nan
        try:
            tt = float(mcp_tt[i])
        except Exception:
            tt = math.nan
        mcp_map[idx] = (pt, amp, tt)

    arrays = tree.arrays([cfg["branch_channel"], cfg["branch_time"], cfg["branch_energy"]], library="ak")
    if cfg["branch_channel"] not in arrays.fields or cfg["branch_time"] not in arrays.fields:
        out["counters"]["missing_mcp"] += 1
        print(f"[ERROR] Branches missing in tree: {cfg['branch_channel']} or {cfg['branch_time']}. Fields: {arrays.fields}", flush=True)
        return out


    n_entries = tree.num_entries
    max_e = n_entries if cfg["max_entries"] is None else min(cfg["max_entries"], n_entries)

    required = set(cfg["channels"])
    required.add(int(cfg["trigger_channel"]))

    # DEBUG: Print number of entries and MCP map size
    if max_e > 0:
        print(f"[DEBUG] Processing {path}: entries={max_e}, mcp_map_size={len(mcp_map)}", flush=True)
        print(f"[DEBUG] First 5 MCP map keys: {list(mcp_map.keys())[:5]}", flush=True)
        print(f"[DEBUG] Required channels: {required}", flush=True)

    for i in range(max_e):
        out["counters"]["total"] += 1
        try:
            ch_list = ak.to_list(arrays[cfg["branch_channel"]][i])
        except Exception as e:
            if out["counters"]["missing_ch"] < 5:
                print(f"[DEBUG] Event {i} missing_ch. Error: {e}", flush=True)
            out["counters"]["missing_ch"] += 1
            continue

        try:
            time_list = ak.to_list(arrays[cfg["branch_time"]][i])
        except Exception:
            out["counters"]["missing_time"] += 1
            continue
        try:
            energy_list = ak.to_list(arrays[cfg["branch_energy"]][i]) if cfg["branch_energy"] in arrays.fields else []
        except Exception:
            energy_list = []

        if not required.issubset(set(ch_list)):
            if out["counters"]["missing_ch"] < 3:
                present = [c for c in required if c in ch_list]
                missing = [c for c in required if c not in ch_list]
                print(f"[DEBUG] Event {i} failed required check. Present: {present}, Missing: {missing}, ch_list sample: {ch_list[:10]}", flush=True)
            out["counters"]["missing_ch"] += 1
            continue

        ch_time = math.nan
        ch_energy = math.nan
        ch2_time = math.nan
        ch2_energy = math.nan
        if cfg["combine_lr"]:
            try:
                pos_l = ch_list.index(cfg["ch_l"])
                pos_r = ch_list.index(cfg["ch_r"])
                if pos_l < 0 or pos_r < 0 or pos_l >= len(time_list) or pos_r >= len(time_list):
                    raise IndexError("L/R time index out of range")
                tl = float(time_list[pos_l])
                tr = float(time_list[pos_r])
                ch_time = 0.5 * (tl + tr)
                out["t_diff_first"].append(tl - tr)
                if pos_l < len(energy_list) and pos_r < len(energy_list):
                    ch_energy = float(energy_list[pos_l]) + float(energy_list[pos_r])
            except Exception:
                out["counters"]["missing_plot_channel"] += 1
                continue
        else:
            try:
                pos = ch_list.index(cfg["plot_channel"])
            except Exception:
                out["counters"]["missing_plot_channel"] += 1
                continue
            if pos < 0 or pos >= len(time_list):
                continue
            try:
                ch_time = float(time_list[pos])
            except Exception:
                continue
            try:
                if pos < len(energy_list):
                    ch_energy = float(energy_list[pos])
            except Exception:
                ch_energy = math.nan

        if cfg["second_configured"]:
            if cfg["second_combine_lr"]:
                try:
                    pos2_l = ch_list.index(cfg["ch2_l"])
                    pos2_r = ch_list.index(cfg["ch2_r"])
                    if pos2_l < 0 or pos2_r < 0 or pos2_l >= len(time_list) or pos2_r >= len(time_list):
                        raise IndexError("Second L/R time index out of range")
                    t2l = float(time_list[pos2_l])
                    t2r = float(time_list[pos2_r])
                    ch2_time = 0.5 * (t2l + t2r)
                    out["t_diff_second"].append(t2l - t2r)
                    if pos2_l < len(energy_list) and pos2_r < len(energy_list):
                        ch2_energy = float(energy_list[pos2_l]) + float(energy_list[pos2_r])
                except Exception:
                    ch2_time = math.nan
                    ch2_energy = math.nan
            else:
                try:
                    pos2 = ch_list.index(cfg["plot2_channel"])
                    if pos2 < 0 or pos2 >= len(time_list):
                        raise IndexError("Second channel index out of range")
                    ch2_time = float(time_list[pos2])
                    if pos2 < len(energy_list):
                        ch2_energy = float(energy_list[pos2])
                except Exception:
                    ch2_time = math.nan
                    ch2_energy = math.nan

        if cfg["energy_min"] is not None and not (ch_energy == ch_energy and ch_energy >= cfg["energy_min"]):
            out["counters"]["energy_cut"] += 1
            continue
        if cfg["energy_max"] is not None and not (ch_energy == ch_energy and ch_energy <= cfg["energy_max"]):
            out["counters"]["energy_cut"] += 1
            continue

        if ch_energy == ch_energy:
            out["energy_vals"].append(ch_energy)

        second_energy_ok = True
        if cfg["second_configured"]:
            if cfg["second_energy_min"] is not None and not (ch2_energy == ch2_energy and ch2_energy >= cfg["second_energy_min"]):
                out["counters"]["second_energy_cut"] += 1
                second_energy_ok = False
            if cfg["second_energy_max"] is not None and not (ch2_energy == ch2_energy and ch2_energy <= cfg["second_energy_max"]):
                out["counters"]["second_energy_cut"] += 1
                second_energy_ok = False
            if second_energy_ok and (ch2_energy == ch2_energy):
                out["energy_vals_second"].append(ch2_energy)
        
        if i not in mcp_map:
            # DEBUG: Sample missing MCP for first few events
            if out["counters"]["missing_mcp"] < 5:
                print(f"[DEBUG] Event {i} missing in mcp_map. Keys: {list(mcp_map.keys())[:5]}...", flush=True)
            out["counters"]["missing_mcp"] += 1
            continue

        
        peak_time, peak_amp, trig_time = mcp_map[i]
        
        # DEBUG: Check if required channels are present for first few matched events
        if out["counters"]["kept"] < 10:
            present = [c for c in required if c in ch_list]
            missing = [c for c in required if c not in ch_list]
            print(f"[DEBUG] Event {i} MATCHED MCP. Found channels: {present}. Missing: {missing}", flush=True)

        # Collect MCP peak_time vs peak_amp
        if (peak_time == peak_time) and (peak_amp == peak_amp):
            out["mcp_peak_time_vs_amp_x"].append(peak_time)
            out["mcp_peak_time_vs_amp_y"].append(peak_amp)

        if not cfg["skip_ch192_plot"]:
            try:
                pos_192 = ch_list.index(cfg["trigger_channel"])
                ch192_t = float(time_list[pos_192])
                if (ch192_t == ch192_t) and (trig_time == trig_time):
                    out["x_ch192"].append(ch192_t)
                    out["y_trig"].append(trig_time)
            except Exception:
                pass

        # Collect TOFHIR time vs MCP peak_time
        if (ch_time == ch_time) and (peak_time == peak_time):
            out["time_vs_mcp_peak_x"].append(ch_time)
            out["time_vs_mcp_peak_y"].append(peak_time)

        # Raw time difference (t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)
        try:
            pos_192_u = ch_list.index(cfg["trigger_channel"])
            ch192_t = float(time_list[pos_192_u])
            if not (ch_time == ch_time and ch192_t == ch192_t and peak_time == peak_time and trig_time == trig_time):
                raise ValueError("Missing time values for raw time diff")
            
            # Raw time difference (t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)
            det_diff = ch_time - ch192_t
            mcp_diff = peak_time - trig_time
            raw_diff = det_diff - mcp_diff
            out["raw_time_diff"].append(raw_diff)
            out["raw_det_diff"].append(det_diff)
            out["raw_mcp_diff"].append(mcp_diff)
            out["raw_peak_amp"].append(peak_amp)
        except Exception:
            if not (trig_time == trig_time):
                out["counters"]["missing_mcp_trigger"] += 1



        out["counters"]["kept"] += 1

    return out


def main():
    args = parse_args()

    print(f"[info] running: {os.path.abspath(__file__)}", flush=True)
    print(f"[info] cwd: {os.getcwd()}", flush=True)

    # Expand glob patterns for input ROOT files
    # Expand glob patterns for input ROOT files
    files = []
    for item in args.file:
        if any(ch in item for ch in ["*", "?", "["]):
            files.extend(sorted(glob.glob(item)))
        else:
            files.append(item)
    if not files:
        print("No files matched:", args.file)
        sys.exit(2)

    # Derive primary detector channels from module/lyso
    plot_channel = None
    plot_channel_label = None
    plot_energy_label = None
    combine_lr = False
    ch_l = None
    ch_r = None
    if args.module and args.lyso_bar is not None and args.side:
        base = UP_MODULE_BASE if args.module == "up" else DOWN_MODULE_BASE
        if args.lyso_bar not in lyso_bar_to_channels_lr:
            print(f"Invalid lyso bar index: {args.lyso_bar}. Valid: {sorted(lyso_bar_to_channels_lr.keys())}")
            sys.exit(2)
        rel_map = lyso_bar_to_channels_lr[args.lyso_bar]
        if args.side == "both":
            ch_l = int(base + rel_map["L"])
            ch_r = int(base + rel_map["R"])
            combine_lr = True
            plot_channel_label = f"module {args.module} bar {args.lyso_bar} (L+R)/2"
            plot_energy_label = f"module {args.module} bar {args.lyso_bar} (L+R)"
            args.channels = [ch_l, ch_r]
        else:
            rel_ch = rel_map[args.side]
            plot_channel = int(base + rel_ch)
            plot_channel_label = f"module {args.module} bar {args.lyso_bar} {args.side}"
            plot_energy_label = plot_channel_label
            args.channels = [plot_channel]
    else:
        print("Must provide --module/--lyso-bar/--side.")
        sys.exit(2)

    # Derive secondary detector channels if provided
    second_configured = False
    second_combine_lr = False
    ch2_l = None
    ch2_r = None
    plot2_channel = None
    plot2_label = None
    plot2_energy_label = None
    if args.second_module or args.second_lyso_bar is not None or args.second_side:
        if not (args.second_module and args.second_lyso_bar is not None and args.second_side):
            print("If using a second detector, must provide --second-module/--second-lyso-bar/--second-side.")
            sys.exit(2)
        base2 = UP_MODULE_BASE if args.second_module == "up" else DOWN_MODULE_BASE
        if args.second_lyso_bar not in lyso_bar_to_channels_lr:
            print(f"Invalid second lyso bar index: {args.second_lyso_bar}. Valid: {sorted(lyso_bar_to_channels_lr.keys())}")
            sys.exit(2)
        rel_map2 = lyso_bar_to_channels_lr[args.second_lyso_bar]
        if args.second_side == "both":
            ch2_l = int(base2 + rel_map2["L"])
            ch2_r = int(base2 + rel_map2["R"])
            second_combine_lr = True
            plot2_label = f"module {args.second_module} bar {args.second_lyso_bar} (L+R)/2"
            plot2_energy_label = f"module {args.second_module} bar {args.second_lyso_bar} (L+R)"
            args.channels.extend([ch2_l, ch2_r])
        else:
            rel_ch2 = rel_map2[args.second_side]
            plot2_channel = int(base2 + rel_ch2)
            plot2_label = f"module {args.second_module} bar {args.second_lyso_bar} {args.second_side}"
            plot2_energy_label = plot2_label
            args.channels.append(plot2_channel)
        second_configured = True

    # Always require trigger channel
    if TRIGGER_CHANNEL not in args.channels:
        args.channels.append(int(TRIGGER_CHANNEL))

    # Accumulators across all files
    x_ch192 = []
    y_trig = []
    time_vs_mcp_peak_x = []
    time_vs_mcp_peak_y = []
    mcp_peak_time_vs_amp_x = []
    mcp_peak_time_vs_amp_y = []
    raw_time_diff = []
    raw_det_diff = []
    raw_mcp_diff = []
    raw_peak_amp = []
    t_diff_first = []
    t_diff_second = []
    energy_vals = []
    energy_vals_second = []
    counters = {
        "total": 0,
        "missing_ch": 0,
        "missing_time": 0,
        "missing_plot_channel": 0,
        "missing_mcp": 0,
        "energy_cut": 0,
        "second_energy_cut": 0,
        "kept": 0,
        "missing_mcp_trigger": 0,
    }

    cfg = {
        "channels": args.channels,
        "branch_channel": args.branch_idx,
        "branch_time": args.branch_time,
        "branch_energy": args.branch_energy,
        "mcp_tree": args.mcp_tree,
        "mcp_index": args.mcp_index,
        "mcp_peak_time": args.mcp_peak_time,
        "mcp_peak_amp": args.mcp_peak_amp,
        "mcp_trigger_time": args.mcp_trigger_time,
        "max_entries": args.max_entries,
        "energy_min": args.energy_min,
        "energy_max": args.energy_max,
        "second_energy_min": args.second_energy_min,
        "second_energy_max": args.second_energy_max,
        "combine_lr": combine_lr,
        "ch_l": ch_l,
        "ch_r": ch_r,
        "plot_channel": plot_channel,
        "second_configured": second_configured,
        "second_combine_lr": second_combine_lr,
        "ch2_l": ch2_l,
        "ch2_r": ch2_r,
        "plot2_channel": plot2_channel,
        "trigger_channel": TRIGGER_CHANNEL,
        "skip_ch192_plot": args.skip_ch192_plot,
    }

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(partial(process_file, cfg=cfg), files))
    else:
        results = [process_file(p, cfg) for p in files]

    for res in results:
        path = res["path"]
        # accumulate
        x_ch192.extend(res["x_ch192"])
        y_trig.extend(res["y_trig"])
        time_vs_mcp_peak_x.extend(res["time_vs_mcp_peak_x"])
        time_vs_mcp_peak_y.extend(res["time_vs_mcp_peak_y"])
        mcp_peak_time_vs_amp_x.extend(res["mcp_peak_time_vs_amp_x"])
        mcp_peak_time_vs_amp_y.extend(res["mcp_peak_time_vs_amp_y"])
        raw_time_diff.extend(res["raw_time_diff"])
        raw_det_diff.extend(res["raw_det_diff"])
        raw_mcp_diff.extend(res["raw_mcp_diff"])
        raw_peak_amp.extend(res["raw_peak_amp"])
        t_diff_first.extend(res["t_diff_first"])
        t_diff_second.extend(res["t_diff_second"])
        energy_vals.extend(res["energy_vals"])
        energy_vals_second.extend(res["energy_vals_second"])
        for k in counters:
            counters[k] += res["counters"].get(k, 0)


        # per-file ch192 plot
        if args.skip_ch192_plot:
            continue
        if res["x_ch192"]:
            x = np.asarray(res["x_ch192"], dtype=float)
            y = np.asarray(res["y_trig"], dtype=float)
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x, y, s=10, alpha=0.6)
            if x.size >= 2:
                try:
                    labels, centers = kmeans_1d(x, k=args.time_peak_lines)
                    order = np.argsort(centers)
                    colors = ["red", "green", "orange", "purple", "brown"]
                    for rank, cluster_id in enumerate(order):
                        mask = labels == cluster_id
                        if np.sum(mask) < 2:
                            continue
                        x_seg = x[mask]
                        y_seg = y[mask]
                        m, b = np.polyfit(x_seg, y_seg, 1)
                        x_line = np.linspace(x_seg.min(), x_seg.max(), 200)
                        y_line = m * x_line + b
                        color = colors[rank % len(colors)]
                        plt.plot(x_line, y_line, color=color, linewidth=2,
                                 label=f"fit {rank+1}: m={m:.5g}, b={b:.5g}")
                    plt.legend()
                except Exception as e:
                    print("Linear fit failed for ch192 vs trigger_time:", e)
            plt.xlabel("channel 192 time")
            plt.ylabel("mcp trigger_time")
            plt.title(f"Channel 192 time vs MCP trigger_time ({os.path.basename(path)})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = f"{stem}_{args.out_ch192_vs_trigger}"
            plt.savefig(out_path, dpi=150)
            print("Saved:", out_path)
        else:
            print(f"No valid points for channel 192 vs mcp_trigger_time plot in {path}.")

    if args.verbose:
        print("Debug counters:", counters)

    # Channel 192 vs MCP trigger_time (scatter + grouped linear fits)
    if not x_ch192:
        print("No valid points for channel 192 vs mcp_trigger_time plot.")
    else:
        x = np.asarray(x_ch192, dtype=float)
        y = np.asarray(y_trig, dtype=float)
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x, y, s=10, alpha=0.6)
        if x.size >= 2:
            try:
                labels, centers = kmeans_1d(x, k=args.time_peak_lines)
                order = np.argsort(centers)
                colors = ["red", "green", "orange", "purple", "brown"]
                for rank, cluster_id in enumerate(order):
                    mask = labels == cluster_id
                    if np.sum(mask) < 2:
                        continue
                    x_seg = x[mask]
                    y_seg = y[mask]
                    m, b = np.polyfit(x_seg, y_seg, 1)
                    x_line = np.linspace(x_seg.min(), x_seg.max(), 200)
                    y_line = m * x_line + b
                    color = colors[rank % len(colors)]
                    plt.plot(x_line, y_line, color=color, linewidth=2,
                             label=f"fit {rank+1}: m={m:.5g}, b={b:.5g}")
                plt.legend()
            except Exception as e:
                print("Linear fit failed for ch192 vs trigger_time:", e)
        plt.xlabel("channel 192 time")
        plt.ylabel("mcp trigger_time")
        plt.title("Channel 192 time vs MCP trigger_time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_ch192_vs_trigger, dpi=150)
        print("Saved:", args.out_ch192_vs_trigger)

    # TOFHIR time vs MCP peak_time scatter plot
    if not time_vs_mcp_peak_x:
        print("No valid points for TOFHIR time vs MCP peak_time plot.")
    else:
        x = np.asarray(time_vs_mcp_peak_x, dtype=float)
        y = np.asarray(time_vs_mcp_peak_y, dtype=float)
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x, y, s=10, alpha=0.6, color="steelblue")
        plt.xlabel(f"TOFHIR time ({plot_channel_label})")
        plt.ylabel("MCP peak_time")
        plt.title(f"TOFHIR time vs MCP peak_time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_time_vs_mcp_peak, dpi=150)
        print("Saved:", args.out_time_vs_mcp_peak)

    # MCP peak_time vs peak_amp scatter plot
    if not mcp_peak_time_vs_amp_x:
        print("No valid points for MCP peak_time vs peak_amp plot.")
    else:
        x = np.asarray(mcp_peak_time_vs_amp_x, dtype=float)
        y = np.asarray(mcp_peak_time_vs_amp_y, dtype=float)
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x, y, s=10, alpha=0.6, color="mediumseagreen")
        plt.xlabel("MCP peak_time")
        plt.ylabel("MCP peak_amp")
        plt.title("MCP peak_time vs peak_amp")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_mcp_peak_time_vs_amp, dpi=150)
        print("Saved:", args.out_mcp_peak_time_vs_amp)

    # Raw time difference plot: (t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)
    if not raw_time_diff:
        print("No valid points for raw time difference plot.")
    else:
        plt.figure(figsize=(6.5, 4.5))
        counts, bins, _ = plt.hist(raw_time_diff, bins=args.energy_bins, alpha=0.75, color="forestgreen", edgecolor="white")
        # Gaussian fit
        try:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mask = counts > 0
            x_fit = bin_centers[mask]
            y_fit = counts[mask]
            if x_fit.size >= 3:
                mu0 = float(np.mean(raw_time_diff))
                sigma0 = float(np.std(raw_time_diff, ddof=1))
                a0 = float(np.max(y_fit))
                def gauss_mu(x, a, mu, sigma):
                    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                popt, _ = curve_fit(gauss_mu, x_fit, y_fit, p0=[a0, mu0, sigma0], maxfev=10000)
                a_fit, mu_fit, sigma_fit = popt
                x_line = np.linspace(bin_centers.min(), bin_centers.max(), 400)
                y_line = gauss_mu(x_line, a_fit, mu_fit, abs(sigma_fit))
                plt.plot(x_line, y_line, color="crimson", linewidth=2,
                         label=f"Gaussian fit: μ={mu_fit:.3g}, σ={abs(sigma_fit):.3g}")
                plt.legend()
        except Exception as e:
            print("Gaussian fit failed for raw time difference plot:", e)
        plt.xlabel("(t_bar - t_192) - (t_mcp_peak - t_mcp_trigger)")
        plt.ylabel("counts")
        plt.title("Raw Time Difference (No Modulo)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_raw_time_diff, dpi=150)
        print("Saved:", args.out_raw_time_diff)

    # Raw time scatter plot: (t_bar - t_192) vs (t_mcp_peak - t_mcp_trigger)
    # Scaled by 1/1000
    if not (raw_det_diff and raw_mcp_diff):
        print("No valid points for raw time scatter plot.")
    else:
        # Scale data by 1/1000
        x_raw = np.asarray(raw_mcp_diff, dtype=float)
        y_raw = np.asarray(raw_det_diff, dtype=float)
        x = x_raw / 1000.0
        y = y_raw / 1000.0

        # Linear fit
        try:
            m, b = np.polyfit(x, y, 1)
            residuals = y - (m * x + b)
            
            # Scatter plot with fit line
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x, y, s=10, alpha=0.6, color="darkcyan", label="Data")
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = m * x_line + b
            plt.plot(x_line, y_line, color="crimson", linewidth=2, label=f"Fit: y={m:.5f}x + {b:.5f}")
            plt.xlabel("(t_mcp_peak - t_mcp_trigger) / 1000")
            plt.ylabel("(t_bar - t_192) / 1000")
            plt.title("Detector vs MCP relative timing (Scaled by 1/1000)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_raw_time_scatter, dpi=150)
            print("Saved:", args.out_raw_time_scatter)

            # Residuals plot
            plt.figure(figsize=(6.5, 4.5))
            counts, bins, _ = plt.hist(residuals, bins=args.energy_bins, alpha=0.75, color="mediumpurple", edgecolor="white")
            
            # Gaussian fit on residuals
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mask = counts > 0
            x_fit = bin_centers[mask]
            y_fit = counts[mask]
            if x_fit.size >= 3:
                mu0 = float(np.mean(residuals))
                sigma0 = float(np.std(residuals, ddof=1))
                a0 = float(np.max(y_fit))
                def gauss_mu(x, a, mu, sigma):
                    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                try:
                    popt, _ = curve_fit(gauss_mu, x_fit, y_fit, p0=[a0, mu0, sigma0], maxfev=10000)
                    a_fit, mu_fit, sigma_fit = popt
                    x_line_res = np.linspace(bin_centers.min(), bin_centers.max(), 400)
                    y_line_res = gauss_mu(x_line_res, a_fit, mu_fit, abs(sigma_fit))
                    plt.plot(x_line_res, y_line_res, color="crimson", linewidth=2,
                             label=f"Gaussian fit: μ={mu_fit:.3g}, σ={abs(sigma_fit):.3g}")
                    plt.legend()
                except Exception as e:
                    print("Gaussian fit failed for residuals:", e)
            
            plt.xlabel("Residuals (Data - Fit)")
            plt.ylabel("counts")
            plt.title("Residuals of Linear Fit")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_raw_time_residual, dpi=150)
            print("Saved:", args.out_raw_time_residual)

            # --- Outlier Removal (Iterative Fit) ---
            # 1. Initial cleaning: Mean +/- 3*StdDev on the y-axis
            mu_y = np.mean(y)
            sigma_y = np.std(y)
            mask_clean = (y >= mu_y - 3 * sigma_y) & (y <= mu_y + 3 * sigma_y)
            x_clean = x[mask_clean]
            y_clean = y[mask_clean]
            
            # 2. Iterative Linear Fit (2 iterations)
            # Fit -> Remove residuals > 2.5 sigma -> Refit
            if len(x_clean) > 2:
                for i in range(2):
                    if len(x_clean) < 3: break
                    # Fit
                    m_temp, b_temp = np.polyfit(x_clean, y_clean, 1)
                    res_temp = y_clean - (m_temp * x_clean + b_temp)
                    mu_res = np.mean(res_temp)
                    sigma_res = np.std(res_temp)
                    # Filter based on residuals
                    mask_iter = (res_temp >= mu_res - 2.5 * sigma_res) & (res_temp <= mu_res + 2.5 * sigma_res)
                    x_clean = x_clean[mask_iter]
                    y_clean = y_clean[mask_iter]
                
                # Final Fit on refined data
                if len(x_clean) > 2:
                    m_clean, b_clean = np.polyfit(x_clean, y_clean, 1)
                    residuals_clean = y_clean - (m_clean * x_clean + b_clean)
                    
                    # Setup figure for clean scatter
                    plt.figure(figsize=(6.5, 4.5))
                    plt.scatter(x_clean, y_clean, s=10, alpha=0.6, color="teal", label="Data (Clean)")
                    x_line_c = np.linspace(x_clean.min(), x_clean.max(), 200)
                    y_line_c = m_clean * x_line_c + b_clean
                    plt.plot(x_line_c, y_line_c, color="darkorange", linewidth=2, label=f"IterFit: y={m_clean:.5f}x + {b_clean:.5f}")
                    plt.xlabel("(t_mcp_peak - t_mcp_trigger) / 1000")
                    plt.ylabel("(t_bar - t_192) / 1000")
                    plt.title("Detector vs MCP relative timing (Iterative Outlier Removal)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(args.out_raw_time_scatter_clean, dpi=150)
                    print("Saved:", args.out_raw_time_scatter_clean)
                    
                    # Setup figure for clean residuals
                    plt.figure(figsize=(6.5, 4.5))
                    counts_c, bins_c, _ = plt.hist(residuals_clean, bins=args.energy_bins, alpha=0.75, color="royalblue", edgecolor="white")
                    
                    # Gaussian fit on clean residuals
                    bin_centers_c = 0.5 * (bins_c[:-1] + bins_c[1:])
                    mask_c = counts_c > 0
                    x_fit_c = bin_centers_c[mask_c]
                    y_fit_c = counts_c[mask_c]
                    if x_fit_c.size >= 3:
                         mu0_c = float(np.mean(residuals_clean))
                         sigma0_c = float(np.std(residuals_clean, ddof=1))
                         a0_c = float(np.max(y_fit_c))
                         try:
                            popt_c, _ = curve_fit(gauss_mu, x_fit_c, y_fit_c, p0=[a0_c, mu0_c, sigma0_c], maxfev=10000)
                            a_fit_c, mu_fit_c, sigma_fit_c = popt_c
                            x_line_res_c = np.linspace(bin_centers_c.min(), bin_centers_c.max(), 400)
                            y_line_res_c = gauss_mu(x_line_res_c, a_fit_c, mu_fit_c, abs(sigma_fit_c))
                            plt.plot(x_line_res_c, y_line_res_c, color="darkorange", linewidth=2,
                                     label=f"Gaussian fit: μ={mu_fit_c:.3g}, σ={abs(sigma_fit_c):.3g}")
                            plt.legend()
                         except Exception as e:
                            print("Gaussian fit failed for clean residuals:", e)
                    
                    plt.xlabel("Residuals (Data - Fit)")
                    plt.ylabel("counts")
                    plt.title("Residuals of Iterative Fit (Clean)")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(args.out_raw_time_residual_clean, dpi=150)
                    print("Saved:", args.out_raw_time_residual_clean)
                else:
                    print("Not enough points left after iterative outlier removal.")
            else:
                 print("Not enough points left after initial outlier removal.")

        except Exception as e:
            print("Linear fit failed for raw time scatter:", e)
            # Fallback to just scatter if fit fails
            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x, y, s=10, alpha=0.6, color="darkcyan")
            plt.xlabel("(t_mcp_peak - t_mcp_trigger) / 1000")
            plt.ylabel("(t_bar - t_192) / 1000")
            plt.title("Detector vs MCP relative timing (Scaled by 1/1000)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_raw_time_scatter, dpi=150)
            print("Saved (no fit):", args.out_raw_time_scatter)

    # Time Difference plots: T_L - T_R
    def plot_t_diff(vals, out_path, title):
        if not vals:
            return
        plt.figure(figsize=(6.5, 4.5))
        counts, bins, _ = plt.hist(vals, bins=args.energy_bins, alpha=0.75, color="teal", edgecolor="white")
        try:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mask = counts > 0
            x_fit = bin_centers[mask]
            y_fit = counts[mask]
            if x_fit.size >= 3:
                mu0 = float(np.mean(vals))
                sigma0 = float(np.std(vals, ddof=1))
                a0 = float(np.max(y_fit))
                def gauss_mu(x, a, mu, sigma):
                    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                popt, _ = curve_fit(gauss_mu, x_fit, y_fit, p0=[a0, mu0, sigma0], maxfev=10000)
                a_fit, mu_fit, sigma_fit = popt
                x_line = np.linspace(bin_centers.min(), bin_centers.max(), 400)
                y_line = gauss_mu(x_line, a_fit, mu_fit, abs(sigma_fit))
                plt.plot(x_line, y_line, color="red", linewidth=2,
                         label=f"Gaussian fit: μ={mu_fit:.3g}, σ={abs(sigma_fit):.3g}")
                plt.legend()
        except: pass
        plt.xlabel("T_left - T_right (ps)")
        plt.ylabel("counts")
        plt.title(title)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print("Saved:", out_path)

    if combine_lr:
        plot_t_diff(t_diff_first, args.out_t_diff_first, f"T_left - T_right ({plot_channel_label})")
    if second_configured and second_combine_lr:
        plot_t_diff(t_diff_second, args.out_t_diff_second, f"T_left - T_right ({plot2_channel_label})")

    # Energy histogram for plot_channel
    if not energy_vals:
        print(f"No valid energy values for {plot_energy_label}.")
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(energy_vals, bins=args.energy_bins, alpha=0.75, color="darkgoldenrod", edgecolor="white")
        plt.xlabel(f"{plot_energy_label} energy")
        plt.ylabel("counts")
        plt.title(f"Energy histogram ({plot_energy_label})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_energy, dpi=150)
        print("Saved:", args.out_energy)

    # Energy histogram for second detector (if configured)
    if second_configured:
        if not energy_vals_second:
            print(f"No valid energy values for {plot2_energy_label}.")
        else:
            plt.figure(figsize=(6.5, 4.5))
            plt.hist(energy_vals_second, bins=args.energy_bins, alpha=0.75, color="darkgoldenrod", edgecolor="white")
            plt.xlabel(f"{plot2_energy_label} energy")
            plt.ylabel("counts")
            plt.title(f"Energy histogram ({plot2_energy_label})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out_energy_second, dpi=150)
            print("Saved:", args.out_energy_second)

    # All other plots removed for analysis-focused study
    # Time Walk Plots (vs 1/amplitude)
    if not (raw_mcp_diff and raw_peak_amp):
        print("No valid points for time walk plots.")
    else:
        # Filter for valid amplitude (non-zero to avoid division by zero)
        mcp_arr = np.array(raw_mcp_diff)
        amp_arr = np.array(raw_peak_amp)
        det_arr = np.array(raw_det_diff) # Same length as raw_mcp_diff and raw_peak_amp
        
        valid_mask = (amp_arr != 0) & np.isfinite(amp_arr) & np.isfinite(mcp_arr) & np.isfinite(det_arr)
        if np.sum(valid_mask) > 0:
            x_inv = 1.0 / amp_arr[valid_mask]
            y_mcp = mcp_arr[valid_mask]
            y_det = det_arr[valid_mask]

            # 1. (mcp_peak - mcp_trigger) vs 1/peak_amp
            # Remove outliers in y_mcp - stricter iterative 2-sigma cut
            mask_clean_mcp = np.ones(len(y_mcp), dtype=bool)
            for _ in range(3):
                subset = y_mcp[mask_clean_mcp]
                if len(subset) < 5: break
                mu_sub = np.mean(subset)
                sig_sub = np.std(subset)
                mask_clean_mcp = mask_clean_mcp & (y_mcp >= mu_sub - 2.0*sig_sub) & (y_mcp <= mu_sub + 2.0*sig_sub)

            x_mcp_f = x_inv[mask_clean_mcp]
            y_mcp_f = y_mcp[mask_clean_mcp]

            # Linear fit for MCP
            m_mcp, b_mcp = np.polyfit(x_mcp_f, y_mcp_f, 1)
            res_mcp = y_mcp_f - (m_mcp * x_mcp_f + b_mcp)

            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x_mcp_f, y_mcp_f, s=10, alpha=0.6, color="darkviolet", label="Data")
            x_line = np.linspace(x_mcp_f.min(), x_mcp_f.max(), 100)
            plt.plot(x_line, m_mcp * x_line + b_mcp, color="red", label=f"Fit: m={m_mcp:.3f}, b={b_mcp:.3f}")
            plt.xlabel("1 / peak_amp (1/V)")
            plt.ylabel("mcp_peak - mcp_trigger (ps)")
            plt.title("MCP Time Walk: Delta T vs 1/Amplitude")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(args.out_walk_mcp, dpi=150)
            print("Saved:", args.out_walk_mcp)

            # Residual histogram for MCP
            plt.figure(figsize=(6.5, 4.5))
            c_mcp, b_mcp_h, _ = plt.hist(res_mcp, bins=args.energy_bins, alpha=0.75, color="darkviolet", edgecolor="white")
            # Gaussian fit on residuals
            try:
                bc_mcp = 0.5 * (b_mcp_h[:-1] + b_mcp_h[1:])
                mask_mcp = c_mcp > 0
                def gauss(x, a, mu, sigma): return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                popt_mcp, _ = curve_fit(gauss, bc_mcp[mask_mcp], c_mcp[mask_mcp], p0=[np.max(c_mcp), np.mean(res_mcp), np.std(res_mcp)])
                plt.plot(bc_mcp, gauss(bc_mcp, *popt_mcp), color="red", linewidth=2, label=f"μ={popt_mcp[1]:.3g}, σ={abs(popt_mcp[2]):.3g}")
                plt.legend()
            except: pass
            plt.xlabel("MCP Residuals (ps)")
            plt.ylabel("counts")
            plt.title("MCP Time Walk Residuals")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(args.out_walk_mcp_res, dpi=150)
            print("Saved:", args.out_walk_mcp_res)

            # 2. (bar_time - ch192) vs 1/peak_amp
            # Remove outliers in y_det (bar_time - ch192) - stricter iterative 2-sigma cut
            mask_clean = np.ones(len(y_det), dtype=bool)
            for _ in range(3):
                subset = y_det[mask_clean]
                if len(subset) < 5: break
                mu_sub = np.mean(subset)
                sig_sub = np.std(subset)
                mask_clean = mask_clean & (y_det >= mu_sub - 2.0*sig_sub) & (y_det <= mu_sub + 2.0*sig_sub)
            
            x_det_f = x_inv[mask_clean]
            y_det_f = y_det[mask_clean]

            # Linear fit for Detector
            m_det, b_det = np.polyfit(x_det_f, y_det_f, 1)
            res_det = y_det_f - (m_det * x_det_f + b_det)

            plt.figure(figsize=(6.5, 4.5))
            plt.scatter(x_det_f, y_det_f, s=10, alpha=0.6, color="dodgerblue", label="Data")
            x_line_det = np.linspace(x_det_f.min(), x_det_f.max(), 100)
            plt.plot(x_line_det, m_det * x_line_det + b_det, color="red", label=f"Fit: m={m_det:.3f}, b={b_det:.3f}")
            plt.xlabel("1 / peak_amp (1/V)")
            plt.ylabel("bar_time - ch192 (ps)")
            plt.title("Detector Time Walk Check: Delta T vs 1/MCP Amplitude")
            plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
            plt.savefig(args.out_walk_bar, dpi=150)
            print("Saved:", args.out_walk_bar)

            # Residual histogram for Detector
            plt.figure(figsize=(6.5, 4.5))
            c_det, b_det_h, _ = plt.hist(res_det, bins=args.energy_bins, alpha=0.75, color="dodgerblue", edgecolor="white")
            # Gaussian fit on residuals
            try:
                bc_det = 0.5 * (b_det_h[:-1] + b_det_h[1:])
                mask_det = c_det > 0
                def gauss(x, a, mu, sigma): return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                popt_det, _ = curve_fit(gauss, bc_det[mask_det], c_det[mask_det], p0=[np.max(c_det), np.mean(res_det), np.std(res_det)])
                plt.plot(bc_det, gauss(bc_det, *popt_det), color="red", linewidth=2, label=f"μ={popt_det[1]:.3g}, σ={abs(popt_det[2]):.3g}")
                plt.legend()
            except: pass
            plt.xlabel("Detector Residuals (ps)")
            plt.ylabel("counts")
            plt.title("Detector Time Walk Residuals")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(args.out_walk_bar_res, dpi=150)
            print("Saved:", args.out_walk_bar_res)
        else:
            print("No valid amplitude points for time walk plots.")
            
if __name__ == "__main__":
    main()
