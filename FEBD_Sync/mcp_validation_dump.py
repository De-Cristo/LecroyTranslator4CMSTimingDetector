#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Dump channel 192 (or chosen channel) from ROOT data tree and attach MCP peak info.

Example:
  python3 mcp_validation_dump.py in.root --out dump.csv --channel 192
  python3 mcp_validation_dump.py in.root --out dump.csv --out-mcp dump_mcp.csv --channel 192
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
    import pandas as pd
    from scipy.optimize import curve_fit
except Exception as e:
    print("Missing Python dependency:", e)
    print("Install: pip install uproot awkward numpy matplotlib pandas")
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


def run_fit_from_csv(csv_path, channel_id, plot_path, n_lines, amp_cut, dump_unit='ps', channel_unit='ps'):
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        sys.exit(2)

    def convert_val(v, src, dst):
        try:
            if v != v:
                return np.nan
        except Exception:
            return np.nan
        if src == dst:
            return float(v)
        if src == 'ps' and dst == 'ns':
            return float(v) / 1000.0
        if src == 'ns' and dst == 'ps':
            return float(v) * 1000.0
        return float(v)

    x_vals = []
    y_vals = []
    rows_out = []
    with open(csv_path, "r", newline="") as cf:
        reader = csv.DictReader(cf)
        for row in reader:
            try:
                ch_list = json.loads(row["channelID"])
                time_list = json.loads(row["time"])
                # read peak time from CSV, interpret according to dump_unit
                if "mcp_peak_time_ps" in row and row["mcp_peak_time_ps"] != "":
                    peak_time_raw = float(row["mcp_peak_time_ps"])  # unit: ps
                    peak_time = convert_val(peak_time_raw, 'ps', channel_unit)
                    peak_time_dump = convert_val(peak_time_raw, 'ps', dump_unit)
                else:
                    peak_time_raw = float(row.get("mcp_peak_time", "nan"))
                    peak_time = convert_val(peak_time_raw, dump_unit, channel_unit)
                    peak_time_dump = convert_val(peak_time_raw, dump_unit, dump_unit)
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
                # assume channel times are in channel_unit already
            except Exception:
                continue

            x_vals.append(peak_time)
            y_vals.append(ch_time)
            rows_out.append({
                'mcp_peak_time_converted': peak_time,  # in channel_unit
                'mcp_peak_time_dump': peak_time_dump,  # in dump_unit
                'channel_time': ch_time,
                'peak_amp': peak_amp,
                'orig_row': row,
            })

    if len(x_vals) < 2:
        print("Not enough valid points for fit:", len(x_vals))
        sys.exit(3)

    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    # Auto-normalization removed: units must be set correctly via CLI args.

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

    # Note: no CSV is written in fit-only mode; input CSV is left unchanged

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
        plt.xlabel(f"mcp_peak_time (unit={channel_unit})")
        plt.ylabel(f"channel {channel_id} time (unit={channel_unit})")
        plt.title("Linear fits: channel time vs MCP peak")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print("Saved fit plot to:", plot_path)


def main():
    p = argparse.ArgumentParser(description="Dump channel data with MCP peaks")
    p.add_argument('--mcp-unit', choices=['ps','ns'], default='ps', help='Unit of peak_time stored in the MCP tree')
    p.add_argument('--dump-unit', choices=['ps','ns'], default='ps', help='Unit to write into the CSV for peak_time')
    p.add_argument('--channel-unit', choices=['ps','ns'], default='ps', help='Unit of channel times in the ROOT dump (time branch)')
    p.add_argument("file", nargs="?", help="Input ROOT file (with data tree and MCP tree)")
    p.add_argument("--out", required=True, help="Output CSV path (dump-df compatible)")
    p.add_argument("--out-mcp", help="Optional MCP-augmented CSV (adds MCP columns)")
    p.add_argument("--channel", type=int, default=192, help="Channel ID used for fit/plots")
    p.add_argument("--branch-channel", help="ChannelID branch name (default: channelID)")
    p.add_argument("--branch-idx", help="ChannelIdx branch name (default: channelIdx)")
    p.add_argument("--branch-time", help="Time branch name (default: time)")
    p.add_argument("--branch-energy", help="Energy branch name (default: energy)")
    p.add_argument("--mcp-tree", default="MCP", help="MCP tree name (default: MCP)")
    p.add_argument("--mcp-index", default="index", help="MCP index branch name (default: index)")
    p.add_argument("--mcp-peak-time", default="peak_time", help="MCP peak time branch name (default: peak_time)")
    p.add_argument("--mcp-peak-amp", default="peak_amp", help="MCP peak amp branch name (default: peak_amp)")
    p.add_argument("--mcp-peak-phase", default="phi_peak", help="MCP peak phase branch name (default: phi_peak)")
    p.add_argument("--mcp-trigger-time", default="trigger_time", help="MCP trigger time branch name (default: trigger_time)")
    p.add_argument("--max-entries", type=int, default=None, help="Max entries to process")
    p.add_argument("--require-channels", nargs="+", type=int, help="ChannelID values that must be present in an event to include it")
    p.add_argument("--workers", type=int, default=1, help="Number of worker threads for per-event dumping")
    p.add_argument("--fit-from-csv", action="store_true", help="Run linear fit using MCP CSV (from --out-mcp)")
    p.add_argument("--fit-plot", help="Save fit plot to this file (e.g. fit.png)")
    p.add_argument("--fit-lines", type=int, default=3, help="Number of lines to fit for --fit-from-csv (default: 3)")
    p.add_argument("--fit-amp-cut", type=float, help="Keep only rows with abs(mcp_peak_amp) >= cut")
    args = p.parse_args()

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
    arrays = tree.arrays(library="ak")
    array_fields = set(arrays.fields)
    n_entries = tree.num_entries
    max_e = n_entries if args.max_entries is None else min(args.max_entries, n_entries)
    required_chs = set(args.require_channels) if getattr(args, "require_channels", None) else set()
    if getattr(args, "channel", None) is not None:
        required_chs.add(int(args.channel))
    if not required_chs:
        required_chs = None

    # Load MCP tree and build index -> peak lookup
    if args.mcp_tree not in f:
        print(f'MCP tree "{args.mcp_tree}" not found in file.')
        sys.exit(5)
    mcp = f[args.mcp_tree]
    mcp_idx = mcp[args.mcp_index].array(library="np")
    mcp_pt = mcp[args.mcp_peak_time].array(library="np")
    mcp_pa = mcp[args.mcp_peak_amp].array(library="np")

    # try to read phi_peak if present
    if args.mcp_peak_phase in mcp.keys():
        mcp_phi = mcp[args.mcp_peak_phase].array(library="np")
    else:
        mcp_phi = np.full(len(mcp_idx), np.nan)

    # try to read phi_peak_from_edge if present
    if "phi_peak_from_edge" in mcp.keys():
        mcp_phi_from_edge = mcp["phi_peak_from_edge"].array(library="np")
    else:
        mcp_phi_from_edge = np.full(len(mcp_idx), np.nan)

    # try to read phi_trigger if present (t0-based)
    if "phi_trigger" in mcp.keys():
        mcp_phi_trigger = mcp["phi_trigger"].array(library="np")
    else:
        mcp_phi_trigger = np.full(len(mcp_idx), np.nan)

    # try to read phi_trigger_from_edge if present (edge-based)
    if "phi_trigger_from_edge" in mcp.keys():
        mcp_phi_trigger_edge = mcp["phi_trigger_from_edge"].array(library="np")
    else:
        mcp_phi_trigger_edge = np.full(len(mcp_idx), np.nan)

    # try to read trigger_time if present
    if args.mcp_trigger_time in mcp.keys():
        mcp_trig = mcp[args.mcp_trigger_time].array(library="np")
    else:
        mcp_trig = np.full(len(mcp_idx), np.nan)

    mcp_map = {}
    def convert_val(v, src, dst):
        try:
            if v != v:
                return np.nan
        except Exception:
            return np.nan
        if src == dst:
            return float(v)
        if src == 'ps' and dst == 'ns':
            return float(v) / 1000.0
        if src == 'ns' and dst == 'ps':
            return float(v) * 1000.0
        return float(v)

    for i in range(len(mcp_idx)):
        try:
            raw = float(mcp_pt[i])
        except Exception:
            raw = math.nan
        try:
            peak_amp_val = float(mcp_pa[i])
        except Exception:
            peak_amp_val = math.nan
        try:
            phi_val = float(mcp_phi[i])
        except Exception:
            phi_val = math.nan
        try:
            phi_from_edge_val = float(mcp_phi_from_edge[i])
        except Exception:
            phi_from_edge_val = math.nan
        try:
            phi_trigger_val = float(mcp_phi_trigger[i])
        except Exception:
            phi_trigger_val = math.nan
        try:
            phi_trigger_edge_val = float(mcp_phi_trigger_edge[i])
        except Exception:
            phi_trigger_edge_val = math.nan
        try:
            trig_val = float(mcp_trig[i])
        except Exception:
            trig_val = math.nan
        # convert from tree unit to desired dump unit
        peak_out = convert_val(raw, args.mcp_unit, args.dump_unit)
        trig_out = convert_val(trig_val, args.mcp_unit, args.dump_unit)
        mcp_map[int(mcp_idx[i])] = (peak_out, peak_amp_val, phi_val, phi_from_edge_val, phi_trigger_val, phi_trigger_edge_val, trig_out)

    # Produce a per-event CSV identical to read_root_explore --dump-df output.
    # Optionally also produce an MCP-augmented CSV with extra columns.
    rows_written = 0
    # Match read_root_explore --dump-df branch selection exactly.
    local_ch_branch = args.branch_channel if args.branch_channel else ("channelID" if "channelID" in keys else None)
    local_idx_branch = args.branch_idx if args.branch_idx else ("channelIdx" if "channelIdx" in keys else None)
    local_time_branch = args.branch_time if args.branch_time else ("time" if "time" in keys else None)
    local_energy_branch = args.branch_energy if args.branch_energy else ("energy" if "energy" in keys else None)

    def process_event(i):
        try:
            ev_ch = ak.to_list(arrays[local_ch_branch][i]) if local_ch_branch in array_fields else []
        except Exception:
            ev_ch = []
        try:
            ev_idx = ak.to_list(arrays[local_idx_branch][i]) if local_idx_branch in array_fields else []
        except Exception:
            ev_idx = []
        try:
            ev_time = ak.to_list(arrays[local_time_branch][i]) if local_time_branch in array_fields else []
        except Exception:
            ev_time = []
        try:
            ev_energy = ak.to_list(arrays[local_energy_branch][i]) if local_energy_branch in array_fields else []
        except Exception:
            ev_energy = []

        if required_chs:
            try:
                if not required_chs.issubset(set(ev_ch)):
                    return None
            except Exception:
                return None

        return [i, json.dumps(ev_ch), json.dumps(ev_idx), json.dumps(ev_time), json.dumps(ev_energy)]

    out_mcp = getattr(args, "out_mcp", None)
    mcp_writer = None
    mcp_file = None
    if out_mcp:
        mcp_file = open(out_mcp, "w", newline="")
        mcp_writer = csv.writer(mcp_file)
        mcp_writer.writerow(['entry', 'channelID', 'channelIdx', 'time', 'energy', 'mcp_index', 'mcp_peak_time', 'mcp_peak_amp', 'mcp_peak_phase', 'mcp_trigger_time'])

    from concurrent.futures import ThreadPoolExecutor
    with open(args.out, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(['entry', 'channelID', 'channelIdx', 'time', 'energy'])

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(process_event, range(max_e)):
                if not result:
                    continue
                writer.writerow(result)
                rows_written += 1

                if mcp_writer:
                    i = result[0]
                    if i in mcp_map:
                        peak_time, peak_amp, phi_peak, phi_from_edge, phi_trigger, phi_trigger_edge, trig_time = mcp_map[i]
                        mcp_index = i
                    else:
                        peak_time, peak_amp, phi_peak, phi_from_edge, phi_trigger, phi_trigger_edge, trig_time = (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
                        mcp_index = math.nan

                    pt_out = '' if peak_time != peak_time else peak_time
                    pa_out = '' if peak_amp != peak_amp else peak_amp
                    ph_out = '' if phi_peak != phi_peak else phi_peak
                    tr_out = '' if trig_time != trig_time else trig_time
                    mcp_writer.writerow([
                        result[0],
                        result[1],
                        result[2],
                        result[3],
                        result[4],
                        mcp_index,
                        pt_out,
                        pa_out,
                        ph_out,
                        tr_out,
                    ])

    if mcp_file:
        try:
            mcp_file.close()
        except OSError as e:
            print(f"[warn] Failed to close MCP output file {out_mcp}: {e}")

    print(f"Wrote {rows_written} rows to {args.out}")

    # --- Diagnostic histograms ---
    out_plot_dir = os.path.dirname(args.out) or '.'
    # Collect arrays from mcp_map for plotting
    peak_time_arr = []   # peak_time (absolute scope time)
    peak_minus_trig_arr = []  # peak_time - trigger_time
    phi_diff_arr = []  # phi_peak_from_edge - phi_trigger_from_edge
    phi_diff_t0_arr = []  # phi_peak - phi_trigger (t0-based)
    inv_amp_arr = []   # 1/peak_amp
    for idx_val, vals in mcp_map.items():
        peak_time, peak_amp, phi_peak, phi_from_edge, phi_trigger, phi_trigger_edge, trig_time = vals
        # peak time
        if peak_time == peak_time:  # not NaN
            peak_time_arr.append(float(peak_time))
        # peak - trigger
        if peak_time == peak_time and trig_time == trig_time:
            peak_minus_trig_arr.append(float(peak_time) - float(trig_time))
        
        if abs(peak_amp) > 1e-6:  # avoid division by zero
            inv_amp = 1.0 / abs(float(peak_amp))
            
            # 1. Edge-based phi difference
            if phi_from_edge == phi_from_edge and phi_trigger_edge == phi_trigger_edge:
                # Center phase difference to [-3125, +3125] ps
                phi_diff_raw = float(phi_from_edge - phi_trigger_edge)
                if phi_diff_raw > 3125.0:
                    phi_diff_raw -= 6250.0
                elif phi_diff_raw < -3125.0:
                    phi_diff_raw += 6250.0
                phi_diff_arr.append(phi_diff_raw)
                # Store inv_amp only once since it's the same for both
                if len(inv_amp_arr) < len(phi_diff_arr): 
                    inv_amp_arr.append(inv_amp)

            # 2. t0-based phi difference
            if phi_peak == phi_peak and phi_trigger == phi_trigger:
                # Center phase difference to [-3125, +3125] ps
                phi_diff_raw = float(phi_peak - phi_trigger)
                if phi_diff_raw > 3125.0:
                    phi_diff_raw -= 6250.0
                elif phi_diff_raw < -3125.0:
                    phi_diff_raw += 6250.0
                phi_diff_t0_arr.append(phi_diff_raw)

            # 3. Raw time difference (peak_time - trigger_time) for amplitude walk
            if peak_time == peak_time and trig_time == trig_time:
                peak_minus_trig_vals = float(peak_time) - float(trig_time)
                # Store tuple (dt, inv_amp) to keep them paired for this specific plot
                # We can reuse inv_amp_arr logic but safer to build separate list if needed
                # Actually, let's just append to a dedicated list of tuples since length might differ from phi arrays
                pass # logic handled below by re-scanning or similar. 
                # Wait, we already built peak_minus_trig_arr earlier. We just need corresponding inv_amp.
                # Let's start a dedicated list for this plot:
                pass
    
    # Re-scan for peak-trig amplitude walk to ensure rigorous pairing
    peak_minus_trig_walk_data = [] 
    for idx_val, vals in mcp_map.items():
        peak_time, peak_amp, phi_peak, phi_from_edge, phi_trigger, phi_trigger_edge, trig_time = vals
        if peak_time == peak_time and trig_time == trig_time and peak_amp == peak_amp and abs(peak_amp) > 1e-6:
            dt = float(peak_time) - float(trig_time)
            inv_a = 1.0 / abs(float(peak_amp))
            peak_minus_trig_walk_data.append((dt, inv_a))
    
    unit_label = args.dump_unit

    if len(peak_time_arr) > 0:
        peak_time_arr = np.array(peak_time_arr)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ax.hist(peak_time_arr, bins=80, edgecolor='black', alpha=0.8)
        mu = float(np.mean(peak_time_arr)); sd = float(np.std(peak_time_arr))
        ax.set_title(f"Peak time (absolute scope time)\nMean={mu:.3f} {unit_label}  RMS={sd:.3f} {unit_label}  N={len(peak_time_arr)}")
        ax.set_xlabel(f"peak_time ({unit_label})"); ax.set_ylabel("Counts"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(out_plot_dir, "hist_peak_time.png")
        plt.savefig(hist_path, dpi=150); plt.close(fig)
        print(f"[ok] Saved peak time histogram: {hist_path}")

    if len(peak_minus_trig_arr) > 0:
        peak_minus_trig_arr = np.array(peak_minus_trig_arr)
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ax.hist(peak_minus_trig_arr, bins=80, edgecolor='black', alpha=0.8)
        mu = float(np.mean(peak_minus_trig_arr)); sd = float(np.std(peak_minus_trig_arr))
        ax.set_title(f"Peak time − trigger time\nMean={mu:.3f} {unit_label}  RMS={sd:.3f} {unit_label}  N={len(peak_minus_trig_arr)}")
        ax.set_xlabel(f"peak_time − trigger_time ({unit_label})"); ax.set_ylabel("Counts"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(out_plot_dir, "hist_peak_minus_trigger.png")
        plt.savefig(hist_path, dpi=150); plt.close(fig)
        print(f"[ok] Saved peak − trigger histogram: {hist_path}")

    # Amplitude walk plot: (phi_peak_from_edge - phi_trigger) vs 1/peak_amp
    if len(phi_diff_arr) > 0 and len(inv_amp_arr) > 0:
        phi_vals = np.array(phi_diff_arr)
        inv_amp_vals = np.array(inv_amp_arr)
        
        # 1. Remove outliers (iterative sigma clipping)
        mask = np.ones(len(phi_vals), dtype=bool)
        for _ in range(3):
            data = phi_vals[mask]
            if len(data) < 2: break
            mu, std = np.median(data), np.std(data)
            if std == 0: break
            mask = mask & (np.abs(phi_vals - mu) < 3 * std)
        
        phi_clean = phi_vals[mask]
        inv_amp_clean = inv_amp_vals[mask]
        print(f"[info] Amplitude walk: kept {len(phi_clean)}/{len(phi_vals)} events after cleaning")

        # 2. Linear fit
        slope, intercept = np.nan, np.nan
        residuals = []
        if len(phi_clean) > 2:
            try:
                slope, intercept = np.polyfit(inv_amp_clean, phi_clean, 1)
                fit_fn = np.poly1d([slope, intercept])
                residuals = phi_clean - fit_fn(inv_amp_clean)
            except Exception as e:
                print(f"[warn] Linear fit failed: {e}")

        # Plot scatter + fit (Panel 1) and Residuals (Panel 2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Scatter
        ax1.scatter(inv_amp_vals, phi_vals, s=5, alpha=0.2, label='All data', color='gray')
        ax1.scatter(inv_amp_clean, phi_clean, s=5, alpha=0.6, label='Cleaned', color='tab:blue')
        if not np.isnan(slope):
            x_line = np.linspace(inv_amp_clean.min(), inv_amp_clean.max(), 100)
            ax1.plot(x_line, fit_fn(x_line), 'r-', label=f'Fit: {slope:.2f}*x + {intercept:.2f}', linewidth=2)
        ax1.set_xlabel("1 / |peak_amp| (1/V)")
        ax1.set_ylabel("φ_peak_from_edge − φ_trigger_from_edge (ps)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Amplitude Walk\nSlope={slope:.4f} ps*V")

        # Panel 2: Residuals
        if len(residuals) > 0:
            # Histogram
            n, bins, patches = ax2.hist(residuals, bins=50, density=True, alpha=0.6, color='g', label='Residuals')
            
            # Gaussian fit to residuals
            try:
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                # Initial guess: a=peak height, x0=mean, sigma=std
                p0 = [np.max(n), np.mean(residuals), np.std(residuals)]
                popt, pcov = curve_fit(gaussian, bin_centers, n, p0=p0)
                
                x_fit = np.linspace(bins[0], bins[-1], 200)
                ax2.plot(x_fit, gaussian(x_fit, *popt), 'r-', linewidth=2, 
                         label=f'Gaus Fit:\n$\mu$={popt[1]:.2f} ps\n$\sigma$={np.abs(popt[2]):.2f} ps')
                ax2.legend()
            except Exception as e:
                print(f"[warn] Gaussian fit to residuals failed: {e}")
            
        ax2.set_xlabel("Residuals (ps)")
        ax2.set_title("Residuals of Linear Fit")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = os.path.join(out_plot_dir, "scatter_phi_diff_vs_inv_amp.png")
        plt.savefig(scatter_path, dpi=150); plt.close(fig)
        print(f"[ok] Saved amplitude walk analysis plot (edge-based): {scatter_path}")

    # Amplitude walk plot (t0-based): (phi_peak - phi_trigger) vs 1/peak_amp
    # Note: inv_amp_arr corresponds one-to-one with phi_diff_t0_arr only if filter logic was identical
    # To be safe, let's re-zip or assume nearly identical coverage. 
    # Actually, simpler to just re-build inv_amp for t0 plot if counts differ, 
    # but for now let's use the one we built if lengths match.
    if len(phi_diff_t0_arr) > 0 and len(phi_diff_t0_arr) == len(inv_amp_arr):
        phi_vals = np.array(phi_diff_t0_arr)
        inv_amp_vals = np.array(inv_amp_arr)
        
        # 1. Remove outliers (iterative sigma clipping)
        mask = np.ones(len(phi_vals), dtype=bool)
        for _ in range(3):
            data = phi_vals[mask]
            if len(data) < 2: break
            mu, std = np.median(data), np.std(data)
            if std == 0: break
            mask = mask & (np.abs(phi_vals - mu) < 3 * std)
        
        phi_clean = phi_vals[mask]
        inv_amp_clean = inv_amp_vals[mask]
        
        # 2. Linear fit
        slope, intercept = np.nan, np.nan
        residuals = []
        if len(phi_clean) > 2:
            try:
                slope, intercept = np.polyfit(inv_amp_clean, phi_clean, 1)
                fit_fn = np.poly1d([slope, intercept])
                residuals = phi_clean - fit_fn(inv_amp_clean)
            except Exception:
                pass

        # Plot scatter + fit (Panel 1) and Residuals (Panel 2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Scatter
        ax1.scatter(inv_amp_vals, phi_vals, s=5, alpha=0.2, label='All data', color='gray')
        ax1.scatter(inv_amp_clean, phi_clean, s=5, alpha=0.6, label='Cleaned', color='tab:orange')
        if not np.isnan(slope):
            x_line = np.linspace(inv_amp_clean.min(), inv_amp_clean.max(), 100)
            ax1.plot(x_line, fit_fn(x_line), 'b-', label=f'Fit: {slope:.2f}*x + {intercept:.2f}', linewidth=2)
        ax1.set_xlabel("1 / |peak_amp| (1/V)")
        ax1.set_ylabel("φ_peak − φ_trigger (t0-based) (ps)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Amplitude Walk (t0-based)\nSlope={slope:.4f} ps*V")

        # Panel 2: Residuals
        if len(residuals) > 0:
            n, bins, patches = ax2.hist(residuals, bins=50, density=True, alpha=0.6, color='purple', label='Residuals')
            try:
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                p0 = [np.max(n), np.mean(residuals), np.std(residuals)]
                popt, pcov = curve_fit(gaussian, bin_centers, n, p0=p0)
                x_fit = np.linspace(bins[0], bins[-1], 200)
                ax2.plot(x_fit, gaussian(x_fit, *popt), 'k-', linewidth=2, 
                         label=f'Gaus Fit:\n$\mu$={popt[1]:.2f} ps\n$\sigma$={np.abs(popt[2]):.2f} ps')
                ax2.legend()
            except Exception:
                pass
            
        ax2.set_xlabel("Residuals (ps)")
        ax2.set_title("Residuals of Linear Fit")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path_t0 = os.path.join(out_plot_dir, "scatter_phi_diff_vs_inv_amp_t0.png")
        plt.savefig(scatter_path_t0, dpi=150); plt.close(fig)
        print(f"[ok] Saved amplitude walk analysis plot (t0-based): {scatter_path_t0}")

    # Amplitude walk plot (raw time): (peak_time - trigger_time) vs 1/peak_amp
    if len(peak_minus_trig_walk_data) > 0:
        dt_vals = np.array([x[0] for x in peak_minus_trig_walk_data])
        inv_vals = np.array([x[1] for x in peak_minus_trig_walk_data])

        # 1. Remove outliers (iterative sigma clipping)
        mask = np.ones(len(dt_vals), dtype=bool)
        for _ in range(3):
            data = dt_vals[mask]
            if len(data) < 2: break
            mu, std = np.median(data), np.std(data)
            if std == 0: break
            mask = mask & (np.abs(dt_vals - mu) < 3 * std)
        
        dt_clean = dt_vals[mask]
        inv_clean = inv_vals[mask]
        
        # 2. Linear fit
        slope, intercept = np.nan, np.nan
        residuals = []
        if len(dt_clean) > 2:
            try:
                slope, intercept = np.polyfit(inv_clean, dt_clean, 1)
                fit_fn = np.poly1d([slope, intercept])
                residuals = dt_clean - fit_fn(inv_clean)
            except Exception:
                pass

        # Plot scatter + fit (Panel 1) and Residuals (Panel 2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Scatter
        ax1.scatter(inv_vals, dt_vals, s=5, alpha=0.2, label='All data', color='gray')
        ax1.scatter(inv_clean, dt_clean, s=5, alpha=0.6, label='Cleaned', color='tab:green')
        if not np.isnan(slope):
            x_line = np.linspace(inv_clean.min(), inv_clean.max(), 100)
            ax1.plot(x_line, fit_fn(x_line), 'r-', label=f'Fit: {slope:.2f}*x + {intercept:.2f}', linewidth=2)
        ax1.set_xlabel("1 / |peak_amp| (1/V)")
        ax1.set_ylabel("peak_time − trigger_time (ps)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Amplitude Walk (Raw Time)\nSlope={slope:.4f} ps*V")

        # Panel 2: Residuals
        if len(residuals) > 0:
            n, bins, patches = ax2.hist(residuals, bins=50, density=True, alpha=0.6, color='teal', label='Residuals')
            try:
                def gaussian(x, a, x0, sigma):
                    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                p0 = [np.max(n), np.mean(residuals), np.std(residuals)]
                popt, pcov = curve_fit(gaussian, bin_centers, n, p0=p0)
                x_fit = np.linspace(bins[0], bins[-1], 200)
                ax2.plot(x_fit, gaussian(x_fit, *popt), 'k-', linewidth=2, 
                         label=f'Gaus Fit:\n$\mu$={popt[1]:.2f} ps\n$\sigma$={np.abs(popt[2]):.2f} ps')
                ax2.legend()
            except Exception:
                pass
            
        ax2.set_xlabel("Residuals (ps)")
        ax2.set_title("Residuals of Linear Fit")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path_raw = os.path.join(out_plot_dir, "scatter_peak_minus_trig_vs_inv_amp.png")
        plt.savefig(scatter_path_raw, dpi=150); plt.close(fig)
        print(f"[ok] Saved amplitude walk analysis plot (raw time): {scatter_path_raw}")

    # Optional: run fit using the freshly dumped CSV
    if args.fit_from_csv:
        fit_csv = args.out_mcp if args.out_mcp else args.out
        if not args.out_mcp:
            print("Fit requested but --out-mcp was not provided. Fit requires MCP columns.")
            sys.exit(6)
        run_fit_from_csv(
            fit_csv,
            args.channel,
            args.fit_plot,
            args.fit_lines,
            args.fit_amp_cut,
            dump_unit=args.dump_unit,
            channel_unit=args.channel_unit,
        )


if __name__ == "__main__":
    main()
