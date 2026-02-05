#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Plot channel time vs MCP peak_time and (channel time % 6250) vs MCP phi_peak.

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
    p.add_argument("--branch-channel", default="channelID")
    p.add_argument("--branch-idx", default="channelIdx")
    p.add_argument("--branch-time", default="time")
    p.add_argument("--branch-energy", default="energy")
    p.add_argument("--mcp-tree", default="MCP")
    p.add_argument("--mcp-index", default="index")
    p.add_argument("--mcp-peak-time", default="peak_time")
    p.add_argument("--mcp-peak-phase", default="phi_peak")
    p.add_argument("--mcp-trigger-time", default="trigger_time")
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--dt-wrap", type=float, default=6250.0,
                   help="Clock period for phase calculations (default: 6250)")
    p.add_argument("--time-peak-lines", type=int, default=3,
                   help="Number of linear segments to fit for ch192 vs trigger (default: 3)")
    p.add_argument("--out-ch192-vs-trigger", default="ch192_vs_mcp_trigger.png",
                   help="Output plot filename for channel 192 time vs MCP trigger_time (with linear fit)")
    p.add_argument("--out-delta-phi", default="delta_phi.png",
                   help="Output plot filename for delta phi (single detector vs MCP)")
    p.add_argument("--out-delta-phi-up-down", default="delta_phi_up_down.png",
                   help="Output plot filename for delta phi (up - down)")
    p.add_argument("--out-delta-phi-up-mcp", default="delta_phi_up_mcp.png",
                   help="Output plot filename for delta phi (up - MCP)")
    p.add_argument("--out-delta-phi-down-mcp", default="delta_phi_down_mcp.png",
                   help="Output plot filename for delta phi (down - MCP)")
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
        "delta_phi": [],
        "delta_phi_up_down": [],
        "delta_phi_up_mcp": [],
        "delta_phi_down_mcp": [],
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
    if cfg["mcp_peak_phase"] in mcp.keys():
        mcp_phi = mcp[cfg["mcp_peak_phase"]].array(library="np")
    else:
        mcp_phi = np.full(len(mcp_idx), np.nan)
    if cfg["mcp_trigger_time"] in mcp.keys():
        mcp_tt = mcp[cfg["mcp_trigger_time"]].array(library="np")
    else:
        mcp_tt = np.full(len(mcp_idx), np.nan)

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
            phi = float(mcp_phi[i])
        except Exception:
            phi = math.nan
        try:
            tt = float(mcp_tt[i])
        except Exception:
            tt = math.nan
        mcp_map[idx] = (pt, phi, tt)

    arrays = tree.arrays([cfg["branch_channel"], cfg["branch_time"], cfg["branch_energy"]], library="ak")
    if cfg["branch_channel"] not in arrays.fields or cfg["branch_time"] not in arrays.fields:
        out["counters"]["missing_mcp"] += 1
        return out

    n_entries = tree.num_entries
    max_e = n_entries if cfg["max_entries"] is None else min(cfg["max_entries"], n_entries)

    required = set(cfg["channels"])
    required.add(int(cfg["trigger_channel"]))

    for i in range(max_e):
        out["counters"]["total"] += 1
        try:
            ch_list = ak.to_list(arrays[cfg["branch_channel"]][i])
        except Exception:
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
                ch_time = 0.5 * (float(time_list[pos_l]) + float(time_list[pos_r]))
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
                    ch2_time = 0.5 * (float(time_list[pos2_l]) + float(time_list[pos2_r]))
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
            out["counters"]["missing_mcp"] += 1
            continue

        peak_time, phi_peak, trig_time = mcp_map[i]

        if not cfg["skip_ch192_plot"]:
            try:
                pos_192 = ch_list.index(cfg["trigger_channel"])
                ch192_t = float(time_list[pos_192])
                if (ch192_t == ch192_t) and (trig_time == trig_time):
                    out["x_ch192"].append(ch192_t)
                    out["y_trig"].append(trig_time)
            except Exception:
                pass

        try:
            pos_192_u = ch_list.index(cfg["trigger_channel"])
            ch192_t = float(time_list[pos_192_u])
            if not (ch_time == ch_time and ch192_t == ch192_t and peak_time == peak_time and trig_time == trig_time):
                raise ValueError("Missing time values for delta phi")
            p = float(cfg["dt_wrap"])
            if p <= 0:
                raise ValueError("dt_wrap must be > 0 for phi")
            phi_obj = ch_time % p
            phi_192 = ch192_t % p
            phi_mcp_peak = peak_time % p
            phi_trigger = trig_time % p
            dphi = (phi_obj - phi_192) - (phi_mcp_peak - phi_trigger)
            dphi = ((dphi + 0.5 * p) % p) - 0.5 * p
            out["delta_phi"].append(dphi)
        except Exception:
            if not (trig_time == trig_time):
                out["counters"]["missing_mcp_trigger"] += 1

        if cfg["second_configured"] and second_energy_ok:
            try:
                t2 = ch2_time
                if not (t2 == t2 and ch_time == ch_time and ch192_t == ch192_t and peak_time == peak_time and trig_time == trig_time):
                    raise ValueError("Missing time values for second delta phi")
                p = float(cfg["dt_wrap"])
                if p <= 0:
                    raise ValueError("dt_wrap must be > 0 for phi")
                phi1 = ch_time % p
                phi2 = t2 % p
                phi_192 = ch192_t % p
                phi_mcp_peak = peak_time % p
                phi_trigger = trig_time % p
                dphi_up_down = (phi1 - phi2)
                dphi_up_mcp = (phi1 - phi_192) - (phi_mcp_peak - phi_trigger)
                dphi_down_mcp = (phi2 - phi_192) - (phi_mcp_peak - phi_trigger)
                dphi_up_down = ((dphi_up_down + 0.5 * p) % p) - 0.5 * p
                dphi_up_mcp = ((dphi_up_mcp + 0.5 * p) % p) - 0.5 * p
                dphi_down_mcp = ((dphi_down_mcp + 0.5 * p) % p) - 0.5 * p
                out["delta_phi_up_down"].append(dphi_up_down)
                out["delta_phi_up_mcp"].append(dphi_up_mcp)
                out["delta_phi_down_mcp"].append(dphi_down_mcp)
            except Exception:
                pass

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
    delta_phi = []
    delta_phi_up_down = []
    delta_phi_up_mcp = []
    delta_phi_down_mcp = []
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
        "branch_channel": args.branch_channel,
        "branch_time": args.branch_time,
        "branch_energy": args.branch_energy,
        "mcp_tree": args.mcp_tree,
        "mcp_index": args.mcp_index,
        "mcp_peak_time": args.mcp_peak_time,
        "mcp_peak_phase": args.mcp_peak_phase,
        "mcp_trigger_time": args.mcp_trigger_time,
        "max_entries": args.max_entries,
        "dt_wrap": args.dt_wrap,
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
        delta_phi.extend(res["delta_phi"])
        delta_phi_up_down.extend(res["delta_phi_up_down"])
        delta_phi_up_mcp.extend(res["delta_phi_up_mcp"])
        delta_phi_down_mcp.extend(res["delta_phi_down_mcp"])
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

    # Delta phi plot (primary vs MCP)
    if not delta_phi:
        print("No valid points for delta phi plot.")
    else:
        plt.figure(figsize=(6.5, 4.5))
        counts, bins, _ = plt.hist(delta_phi, bins=args.energy_bins, alpha=0.75, color="slateblue", edgecolor="white")
        # Gaussian fit on delta phi histogram
        try:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mask = counts > 0
            x_fit = bin_centers[mask]
            y_fit = counts[mask]
            if x_fit.size >= 3:
                mu0 = float(np.mean(delta_phi))
                sigma0 = float(np.std(delta_phi, ddof=1))
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
            print("Gaussian fit failed for delta phi plot:", e)
        plt.xlabel("delta phi = (phi_obj-phi_192)-(phi_mcp_peak-phi_trigger)")
        plt.ylabel("counts")
        plt.title("Delta phi")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_delta_phi, dpi=150)
        print("Saved:", args.out_delta_phi)

    # Delta phi plots for two-detector mode + MCP
    if second_configured:
        def plot_delta_phi(vals, out_path, title):
            if not vals:
                print(f"No valid points for {title}.")
                return None
            plt.figure(figsize=(6.5, 4.5))
            counts, bins, _ = plt.hist(vals, bins=args.energy_bins, alpha=0.75, color="slateblue", edgecolor="white")
            # Gaussian fit
            fit_sigma = None
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
                    fit_sigma = float(abs(sigma_fit))
                    x_line = np.linspace(bin_centers.min(), bin_centers.max(), 400)
                    y_line = gauss_mu(x_line, a_fit, mu_fit, abs(sigma_fit))
                    plt.plot(x_line, y_line, color="crimson", linewidth=2,
                             label=f"Gaussian fit: μ={mu_fit:.3g}, σ={abs(sigma_fit):.3g}")
                    plt.legend()
            except Exception as e:
                print(f"Gaussian fit failed for {title}:", e)
            plt.xlabel(title)
            plt.ylabel("counts")
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print("Saved:", out_path)
            return fit_sigma

        sigma_ud_fit = plot_delta_phi(delta_phi_up_down, args.out_delta_phi_up_down, "delta phi (up - down)")
        sigma_um_fit = plot_delta_phi(delta_phi_up_mcp, args.out_delta_phi_up_mcp, "delta phi (up - MCP)")
        sigma_dm_fit = plot_delta_phi(delta_phi_down_mcp, args.out_delta_phi_down_mcp, "delta phi (down - MCP)")

        # Sigma extraction from three delta distributions
        if sigma_ud_fit and sigma_um_fit and sigma_dm_fit:
            sigma_ud = float(sigma_ud_fit)
            sigma_um = float(sigma_um_fit)
            sigma_dm = float(sigma_dm_fit)
            # Solve: sigma_ud^2 = su^2 + sd^2; sigma_um^2 = su^2 + sm^2; sigma_dm^2 = sd^2 + sm^2
            su2 = 0.5 * (sigma_ud**2 + sigma_um**2 - sigma_dm**2)
            sd2 = 0.5 * (sigma_ud**2 + sigma_dm**2 - sigma_um**2)
            sm2 = 0.5 * (sigma_um**2 + sigma_dm**2 - sigma_ud**2)
            su = math.sqrt(su2) if su2 > 0 else math.nan
            sd = math.sqrt(sd2) if sd2 > 0 else math.nan
            sm = math.sqrt(sm2) if sm2 > 0 else math.nan
            print("[sigma] std(delta_up_down) =", sigma_ud)
            print("[sigma] std(delta_up_mcp)  =", sigma_um)
            print("[sigma] std(delta_down_mcp)=", sigma_dm)
            print("[sigma] solved sigma_up   =", su)
            print("[sigma] solved sigma_down =", sd)
            print("[sigma] solved sigma_mcp  =", sm)

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
if __name__ == "__main__":
    main()
