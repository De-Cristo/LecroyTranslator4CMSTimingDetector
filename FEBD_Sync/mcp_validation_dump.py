#!/usr/bin/env python3
"""
Dump channel 192 (or chosen channel) from ROOT data tree and attach MCP peak info.

Example:
  python3 mcp_validation_dump.py in.root --out dump.csv --channel 192
"""

import argparse
import os
import sys
import math
import csv
import json

try:
    import uproot
    import awkward as ak
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing Python dependency:", e)
    print("Install: pip install uproot awkward numpy matplotlib")
    sys.exit(1)


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


def pick_channel_branch(keys, explicit):
    if explicit:
        return explicit
    for k in keys:
        if k.lower() == "channelid":
            return k
    return None


def kmeans_1d(x, k=3, iters=50):
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    centers = np.linspace(xmin, xmax, k)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        # assign
        for i, xv in enumerate(x):
            labels[i] = int(np.argmin(np.abs(centers - xv)))
        # update
        new_centers = np.array([x[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def linear_fit_with_slope_error(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx == 0:
        return np.nan, np.nan, np.nan
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    resid = y - (slope * x + intercept)
    if n > 2:
        s2 = np.sum(resid ** 2) / (n - 2)
        slope_err = np.sqrt(s2 / sxx)
    else:
        slope_err = np.nan
    return slope, intercept, slope_err


def run_fit_from_csv(csv_path, channel_id, plot_path, n_lines, amp_cut):
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        sys.exit(2)

    x_vals = []
    y_vals = []
    with open(csv_path, "r", newline="") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            try:
                ch_list = json.loads(row["channelID"])
                time_list = json.loads(row["time"])
                peak_time = float(row["mcp_peak_time"])
                peak_amp = float(row.get("mcp_peak_amp", "nan"))
            except Exception:
                continue

            if peak_time != peak_time:  # NaN check
                continue
            if amp_cut is not None and peak_amp == peak_amp:
                if abs(peak_amp) < amp_cut:
                    continue

            try:
                pos = ch_list.index(channel_id)
            except Exception:
                continue

            if pos < 0 or pos >= len(time_list):
                continue

            try:
                ch_time = float(time_list[pos])
            except Exception:
                continue

            x_vals.append(peak_time)
            y_vals.append(ch_time)

    if len(x_vals) < 2:
        print("Not enough valid points for fit:", len(x_vals))
        sys.exit(3)

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    labels, centers = kmeans_1d(y, k=n_lines)
    order = np.argsort(centers)
    fit_params = []
    print("Linear fits (channel time vs mcp_peak_time):")
    print("  total points:", len(x_vals))
    for rank, cluster_id in enumerate(order):
        mask = labels == cluster_id
        if np.sum(mask) < 2:
            continue
        slope, intercept, slope_err = linear_fit_with_slope_error(x[mask], y[mask])
        fit_params.append((cluster_id, slope, intercept))
        print(f"  cluster {rank+1}: points={np.sum(mask)} slope={slope} intercept={intercept} slope_err={slope_err}")

    if plot_path:
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x, y, s=10, alpha=0.6, label="data")
        colors = ["red", "green", "orange", "purple", "brown"]
        for rank, cluster_id in enumerate(order):
            mask = labels == cluster_id
            if np.sum(mask) < 2:
                continue
            slope, intercept, slope_err = linear_fit_with_slope_error(x[mask], y[mask])
            x_line = np.linspace(x[mask].min(), x[mask].max(), 200)
            y_line = slope * x_line + intercept
            color = colors[rank % len(colors)]
            if slope_err == slope_err:
                label = f"fit {rank+1}: m={slope:.4g}±{slope_err:.2g}"
            else:
                label = f"fit {rank+1}: m={slope:.4g}"
            plt.plot(x_line, y_line, color=color, linewidth=2, label=label)
        plt.xlabel("mcp_peak_time")
        plt.ylabel(f"channel {channel_id} time")
        plt.title("Linear fits: channel time vs MCP peak")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print("Saved fit plot to:", plot_path)


def main():
    p = argparse.ArgumentParser(description="Dump channel data with MCP peaks")
    p.add_argument("file", nargs="?", help="Input ROOT file (with data tree and MCP tree)")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--channel", type=int, default=192, help="Channel ID to require in event")
    p.add_argument("--branch-channel", help="ChannelID branch name (default: channelID)")
    p.add_argument("--branch-idx", help="ChannelIdx branch name (default: channelIdx)")
    p.add_argument("--branch-time", help="Time branch name (default: time)")
    p.add_argument("--branch-energy", help="Energy branch name (default: energy)")
    p.add_argument("--mcp-tree", default="MCP", help="MCP tree name (default: MCP)")
    p.add_argument("--mcp-index", default="index", help="MCP index branch name (default: index)")
    p.add_argument("--mcp-peak-time", default="peak_time", help="MCP peak time branch name (default: peak_time)")
    p.add_argument("--mcp-peak-amp", default="peak_amp", help="MCP peak amp branch name (default: peak_amp)")
    p.add_argument("--max-entries", type=int, default=None, help="Max entries to process")
    p.add_argument("--fit-from-csv", help="Run linear fit using dump CSV instead of reading ROOT")
    p.add_argument("--fit-plot", help="Save fit plot to this file (e.g. fit.png)")
    p.add_argument("--fit-lines", type=int, default=3, help="Number of lines to fit for --fit-from-csv (default: 3)")
    p.add_argument("--fit-amp-cut", type=float, help="Keep only rows with abs(mcp_peak_amp) >= cut")
    args = p.parse_args()

    if args.fit_from_csv:
        run_fit_from_csv(args.fit_from_csv, args.channel, args.fit_plot, args.fit_lines, args.fit_amp_cut)
        return

    if not args.file or not os.path.exists(args.file):
        print("File not found:", args.file)
        sys.exit(2)

    f = uproot.open(args.file)
    tree_name = find_data_tree(f)
    if tree_name is None:
        print("No data tree found in file.")
        sys.exit(3)
    tree = f[tree_name]
    keys = list(tree.keys())

    ch_branch = pick_channel_branch(keys, args.branch_channel)
    idx_branch = args.branch_idx if args.branch_idx else ("channelIdx" if "channelIdx" in keys else None)
    time_branch = args.branch_time if args.branch_time else ("time" if "time" in keys else None)
    energy_branch = args.branch_energy if args.branch_energy else ("energy" if "energy" in keys else None)

    if not ch_branch or not idx_branch or not time_branch or not energy_branch:
        print("Missing required branches. Found:")
        print("  channel:", ch_branch)
        print("  channelIdx:", idx_branch)
        print("  time:", time_branch)
        print("  energy:", energy_branch)
        sys.exit(4)

    arrays = tree.arrays([ch_branch, idx_branch, time_branch, energy_branch], library="ak")
    array_fields = set(arrays.fields)
    n_entries = tree.num_entries
    max_e = n_entries if args.max_entries is None else min(args.max_entries, n_entries)

    # Load MCP tree and build index -> peak lookup
    if args.mcp_tree not in f:
        print(f'MCP tree "{args.mcp_tree}" not found in file.')
        sys.exit(5)
    mcp = f[args.mcp_tree]
    mcp_idx = mcp[args.mcp_index].array(library="np")
    mcp_pt = mcp[args.mcp_peak_time].array(library="np")
    mcp_pa = mcp[args.mcp_peak_amp].array(library="np")

    mcp_map = {}
    for i in range(len(mcp_idx)):
        mcp_map[int(mcp_idx[i])] = (float(mcp_pt[i]), float(mcp_pa[i]))

    with open(args.out, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "entry",
            "channelID",
            "channelIdx",
            "time",
            "energy",
            "mcp_index",
            "mcp_peak_time",
            "mcp_peak_amp",
        ])

        for i in range(max_e):
            ev_ch = ak.to_list(arrays[ch_branch][i]) if ch_branch in array_fields else []
            ev_idx = ak.to_list(arrays[idx_branch][i]) if idx_branch in array_fields else []
            ev_time = ak.to_list(arrays[time_branch][i]) if time_branch in array_fields else []
            ev_energy = ak.to_list(arrays[energy_branch][i]) if energy_branch in array_fields else []

            if args.channel not in ev_ch:
                continue

            if i in mcp_map:
                peak_time, peak_amp = mcp_map[i]
                mcp_index = i
            else:
                peak_time, peak_amp = (math.nan, math.nan)
                mcp_index = math.nan
            writer.writerow([
                i,
                json.dumps(ev_ch),
                json.dumps(ev_idx),
                json.dumps(ev_time),
                json.dumps(ev_energy),
                mcp_index,
                peak_time,
                peak_amp,
            ])

    print(f"Wrote {max_e} entries to {args.out}")


if __name__ == "__main__":
    main()
