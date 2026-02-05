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
    p.add_argument("file", help="Input ROOT file")
    p.add_argument("--channels", nargs="+", type=int, required=True,
                   help="Channel IDs required to exist in event (space-separated)")
    p.add_argument("--plot-channel", type=int,
                   help="Channel ID to plot (defaults to first value in --channels)")
    p.add_argument("--require-channel", type=int, default=192,
                   help="Additional required channel (default: 192)")
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
    p.add_argument("--out-time-peak", default="time_vs_mcp_peak_time.png",
                   help="Output plot filename for time vs MCP peak_time")
    p.add_argument("--out-timephi", default="time_mod6250_vs_phi_peak.png",
                   help="Output plot filename for time%%6250 vs MCP phi_peak")
    p.add_argument("--out-dt-hist", default="dt_channel_minus_mcp_peak_time.png",
                   help="Output histogram filename for (channel time - MCP peak_time)")
    p.add_argument("--dt-bins", type=int, default=120,
                   help="Histogram bins for dt plot (default: 120)")
    p.add_argument("--time-peak-lines", type=int, default=3,
                   help="Number of linear segments to fit in time vs MCP peak_time (default: 3)")
    p.add_argument("--dt-wrap", type=float, default=6250.0,
                   help="Wrap period for dt histogram (default: 6250)")
    p.add_argument("--out-dt-wrap-hist", default="dt_wrapped_hist.png",
                   help="Output histogram filename for wrapped dt")
    p.add_argument("--out-dt-resid-hist", default="dt_residual_hist.png",
                   help="Output histogram filename for dt residuals vs fitted lines")
    p.add_argument("--out-dt-abs-hist", default="dt_abs_hist.png",
                   help="Output histogram filename for per-cluster |dt|")
    p.add_argument("--out-dt-chdiff-vs-mcp", default=None,
                   help="Output plot filename for (chA-chB) vs (mcp_peak-mcp_trigger). If omitted, auto-name from channels.")
    p.add_argument("--out-ch192-vs-trigger", default="ch192_vs_mcp_trigger.png",
                   help="Output plot filename for channel 192 time vs MCP trigger_time (with linear fit)")
    p.add_argument("--out-dt-ch133-peak-unified", default="dt_ch133_minus_peak_unified.png",
                   help="Output plot filename for dt = ch133 - (peak + (ch192 - trigger))")
    p.add_argument("--out-dt-ch133-peak-fit", default="dt_ch133_minus_peak_fit.png",
                   help="Output plot filename for dt using fitted ch192 vs trigger (global linear fit)")
    p.add_argument("--out-dt-phi", default="dt_ch133_peak_phi.png",
                   help="Output plot filename for dt phase (wrapped by 6250)")
    p.add_argument("--out-energy", default="energy_plot_channel.png",
                   help="Output plot filename for energy histogram of --plot-channel")
    p.add_argument("--energy-bins", type=int, default=120,
                   help="Histogram bins for energy plot (default: 120)")
    p.add_argument("--energy-min", type=float, default=None,
                   help="Lower bound energy cut for plot-channel (applies to all plots)")
    p.add_argument("--energy-max", type=float, default=None,
                   help="Upper bound energy cut for plot-channel (applies to all plots)")
    p.add_argument("--verbose", action="store_true",
                   help="Print debug counters")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[info] running: {os.path.abspath(__file__)}", flush=True)
    print(f"[info] cwd: {os.getcwd()}", flush=True)

    if not os.path.exists(args.file):
        print("File not found:", args.file)
        sys.exit(2)

    plot_channel = args.plot_channel if args.plot_channel is not None else args.channels[0]
    ch_a = int(plot_channel)
    ch_b = int(args.require_channel) if args.require_channel is not None else (int(args.channels[1]) if len(args.channels) > 1 else None)
    if args.out_dt_chdiff_vs_mcp is None:
        if ch_b is not None:
            args.out_dt_chdiff_vs_mcp = f"dt_ch{ch_a}_ch{ch_b}_vs_mcp_peak_trigger.png"
        else:
            args.out_dt_chdiff_vs_mcp = "dt_chA_chB_vs_mcp_peak_trigger.png"

    f = uproot.open(args.file)
    tree_name = find_data_tree(f)
    if tree_name is None:
        print("No data tree found in file.")
        sys.exit(3)
    tree = f[tree_name]

    # Load MCP tree and build index -> (peak_time, phi_peak, trigger_time)
    if args.mcp_tree not in f:
        print(f'MCP tree "{args.mcp_tree}" not found in file.')
        sys.exit(4)
    mcp = f[args.mcp_tree]
    mcp_idx = mcp[args.mcp_index].array(library="np")
    mcp_pt = mcp[args.mcp_peak_time].array(library="np")
    if args.mcp_peak_phase in mcp.keys():
        mcp_phi = mcp[args.mcp_peak_phase].array(library="np")
    else:
        mcp_phi = np.full(len(mcp_idx), np.nan)
    if args.mcp_trigger_time in mcp.keys():
        mcp_tt = mcp[args.mcp_trigger_time].array(library="np")
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

    arrays = tree.arrays([args.branch_channel, args.branch_time, args.branch_energy], library="ak")
    # Access arrays safely
    if args.branch_channel not in arrays.fields or args.branch_time not in arrays.fields:
        print("Missing required branches in data tree.")
        sys.exit(5)

    n_entries = tree.num_entries
    max_e = n_entries if args.max_entries is None else min(args.max_entries, n_entries)

    required = set(args.channels)
    if args.require_channel is not None:
        required.add(int(args.require_channel))

    x_time_vs_peak = []
    y_peak = []
    x_time_mod = []
    y_phi = []
    x_dt_ch = []
    y_dt_mcp = []
    x_ch192 = []
    y_trig = []
    dt_unified = []
    dt_fit = []
    dt_phi = []
    energy_vals = []
    counters = {
        "total": 0,
        "missing_ch": 0,
        "missing_time": 0,
        "missing_plot_channel": 0,
        "missing_mcp": 0,
        "energy_cut": 0,
        "kept": 0,
        "missing_chdiff_channels": 0,
        "missing_mcp_trigger": 0,
        "kept_chdiff": 0,
    }

    for i in range(max_e):
        counters["total"] += 1
        try:
            ch_list = ak.to_list(arrays[args.branch_channel][i])
        except Exception:
            counters["missing_ch"] += 1
            continue
        try:
            time_list = ak.to_list(arrays[args.branch_time][i])
        except Exception:
            counters["missing_time"] += 1
            continue
        try:
            energy_list = ak.to_list(arrays[args.branch_energy][i]) if args.branch_energy in arrays.fields else []
        except Exception:
            energy_list = []

        if not required.issubset(set(ch_list)):
            counters["missing_ch"] += 1
            continue

        try:
            pos = ch_list.index(plot_channel)
        except Exception:
            counters["missing_plot_channel"] += 1
            continue

        if pos < 0 or pos >= len(time_list):
            continue

        try:
            ch_time = float(time_list[pos])
        except Exception:
            continue
        ch_energy = math.nan
        try:
            if pos < len(energy_list):
                ch_energy = float(energy_list[pos])
        except Exception:
            ch_energy = math.nan

        # Apply energy cut for plot-channel to all plots
        if args.energy_min is not None and not (ch_energy == ch_energy and ch_energy >= args.energy_min):
            counters["energy_cut"] += 1
            continue
        if args.energy_max is not None and not (ch_energy == ch_energy and ch_energy <= args.energy_max):
            counters["energy_cut"] += 1
            continue

        if ch_energy == ch_energy:
            energy_vals.append(ch_energy)

        if i not in mcp_map:
            counters["missing_mcp"] += 1
            continue

        peak_time, phi_peak, trig_time = mcp_map[i]
        if peak_time == peak_time:
            x_time_vs_peak.append(ch_time)
            y_peak.append(peak_time)
        if phi_peak == phi_peak:
            x_time_mod.append(ch_time % 6250.0)
            y_phi.append(phi_peak)

        # Extra plot: (chA - chB) vs (mcp_peak_time - mcp_trigger_time)
        # Require both channels and both MCP times
        try:
            if ch_b is None:
                raise ValueError("Second channel not defined for dt plot")
            pos_a = ch_list.index(ch_a)
            pos_b = ch_list.index(ch_b)
        except Exception:
            counters["missing_chdiff_channels"] += 1
            pos_a = pos_b = None
        if pos_a is not None and pos_b is not None:
            try:
                ch_a_t = float(time_list[pos_a])
                ch_b_t = float(time_list[pos_b])
            except Exception:
                ch_a_t = ch_b_t = math.nan
            if not (trig_time == trig_time):
                counters["missing_mcp_trigger"] += 1
            if (ch_a_t == ch_a_t) and (ch_b_t == ch_b_t) and (peak_time == peak_time) and (trig_time == trig_time):
                x_dt_ch.append(ch_a_t - ch_b_t)
                y_dt_mcp.append(peak_time - trig_time)
                counters["kept_chdiff"] += 1

        # Channel 192 vs MCP trigger_time scatter (with linear fit)
        try:
            pos_192 = ch_list.index(192)
            ch192_t = float(time_list[pos_192])
            if (ch192_t == ch192_t) and (trig_time == trig_time):
                x_ch192.append(ch192_t)
                y_trig.append(trig_time)
        except Exception:
            pass

        # Unified time-frame dt: ch133 - (peak_time + (ch192 - trigger_time))
        try:
            pos_133 = ch_list.index(133)
            pos_192_u = ch_list.index(192)
            ch133_t = float(time_list[pos_133])
            ch192_t = float(time_list[pos_192_u])
            if (ch133_t == ch133_t) and (ch192_t == ch192_t) and (peak_time == peak_time) and (trig_time == trig_time):
                peak_in_ch_frame = peak_time + (ch192_t - trig_time)
                dt_val = ch133_t - peak_in_ch_frame
                dt_unified.append(dt_val)
                # phase-wrapped dt (phi period = 6250)
                dt_phi.append(((dt_val + 0.5 * 6250.0) % 6250.0) - 0.5 * 6250.0)
        except Exception:
            pass
        counters["kept"] += 1

    if args.verbose:
        print("Debug counters:", counters)

    if not x_time_vs_peak:
        print("No valid points for time vs MCP peak_time plot.")
    else:
        # Also build dt array for histogram + gaussian fit
        dt_vals = [t - p for t, p in zip(x_time_vs_peak, y_peak)]

        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x_time_vs_peak, y_peak, s=10, alpha=0.6)
        # Fit linear segments by clustering in channel time
        fit_coeffs = {}
        x0 = float(np.mean(x_time_vs_peak))
        y0 = float(np.mean(y_peak))
        try:
            labels, centers = kmeans_1d(x_time_vs_peak, k=args.time_peak_lines)
            order = np.argsort(centers)
            colors = ["red", "green", "orange", "purple", "brown"]
            for rank, cluster_id in enumerate(order):
                mask = labels == cluster_id
                if np.sum(mask) < 2:
                    continue
                x_seg = np.asarray(x_time_vs_peak)[mask]
                y_seg = np.asarray(y_peak)[mask]
                # Center to improve numerical precision at large values
                x_c = x_seg - x0
                y_c = y_seg - y0
                coeff = np.polyfit(x_c, y_c, 1)  # y_c = m*x_c + b_c
                fit_coeffs[cluster_id] = coeff
                x_line = np.linspace(x_seg.min(), x_seg.max(), 200)
                y_line = (coeff[0] * (x_line - x0) + coeff[1]) + y0
                color = colors[rank % len(colors)]
                plt.plot(x_line, y_line, color=color, linewidth=2,
                         label=f"fit {rank+1}: m={coeff[0]:.4g}")
            plt.legend()
        except Exception as e:
            print("Segmented linear fit failed:", e)
        plt.xlabel(f"channel {plot_channel} time")
        plt.ylabel("MCP peak_time")
        plt.title("Time vs MCP peak_time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_time_peak, dpi=150)
        print("Saved:", args.out_time_peak)

        # Two histograms: wrapped dt and residuals vs fitted lines
        dt = np.asarray(dt_vals, dtype=float)
        if dt.size > 2:
            def gauss0(x, a, sigma):
                return a * np.exp(-0.5 * ((x - 0) / sigma) ** 2)

            def gauss_mu(x, a, mu, sigma):
                return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

            def fit_and_plot_hist(data, out_path, xlabel, title, hist_range=None, allow_mu=False):
                mu0 = float(np.mean(data))
                sigma0 = float(np.std(data, ddof=1))
                plt.figure(figsize=(6.5, 4.5))
                counts, bins, _ = plt.hist(data, bins=args.dt_bins, range=hist_range, alpha=0.75, color="steelblue", edgecolor="white")
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                if sigma0 > 0:
                    mask = counts > 0
                    x_fit = bin_centers[mask]
                    y_fit = counts[mask]
                    if x_fit.size >= 3:
                        a0 = float(np.max(y_fit))
                        try:
                            if allow_mu:
                                popt, pcov = curve_fit(gauss_mu, x_fit, y_fit, p0=[a0, mu0, sigma0], maxfev=10000)
                                a_fit, mu_fit, sigma_fit = popt
                            else:
                                popt, pcov = curve_fit(gauss0, x_fit, y_fit, p0=[a0, sigma0], maxfev=10000)
                                a_fit, sigma_fit = popt
                                mu_fit = 0.0
                            x_line = np.linspace(bin_centers.min(), bin_centers.max(), 400)
                            y_line = gauss_mu(x_line, a_fit, mu_fit, abs(sigma_fit))
                            plt.plot(x_line, y_line, color="crimson", linewidth=2,
                                     label=f"Gaussian fit: μ={mu_fit:.3g}, σ={abs(sigma_fit):.3g}")
                            plt.legend()
                        except Exception as e:
                            print("Gaussian fit failed:", e)
                plt.xlabel(xlabel)
                plt.ylabel("counts")
                plt.title(title)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                print("Saved:", out_path)

            # Wrapped dt histogram
            p = float(args.dt_wrap)
            if p > 0:
                dt_wrapped = ((dt + 0.5 * p) % p) - 0.5 * p
                fit_and_plot_hist(
                    dt_wrapped,
                    args.out_dt_wrap_hist,
                    f"wrapped (channel {plot_channel} time - MCP peak_time)",
                    "Wrapped Δt histogram with Gaussian fit",
                    hist_range=(-3200, 3200)
                )
            else:
                print("dt-wrap must be > 0 to build wrapped histogram.")

            # Residuals vs fitted lines (if fit available)
            if "labels" in locals() and fit_coeffs:
                resid = []
                for idx, (xv, yv) in enumerate(zip(x_time_vs_peak, y_peak)):
                    cl = labels[idx]
                    if cl not in fit_coeffs:
                        continue
                    m, b = fit_coeffs[cl]
                    # residual in peak_time relative to fitted line (centered for precision)
                    resid.append((yv - y0) - (m * (xv - x0) + b))
                resid = np.asarray(resid, dtype=float)
                if resid.size > 2:
                    resid = resid - np.mean(resid)
                    fit_and_plot_hist(
                        resid,
                        args.out_dt_resid_hist,
                        f"residual (MCP peak_time - fit)",
                        "Residuals vs fitted lines with Gaussian fit",
                        hist_range=(-50, 50)
                    )
                else:
                    print("Not enough residual points for histogram/fit.")

            else:
                print("No fitted lines available for residual histogram.")
        else:
            print("Not enough points for dt histograms/fit.")

    if not x_time_mod:
        print("No valid points for time%6250 vs MCP phi_peak plot.")
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x_time_mod, y_phi, s=10, alpha=0.6)
        plt.xlabel(f"channel {plot_channel} time % 6250")
        plt.ylabel("MCP phi_peak")
        plt.title("Time%6250 vs MCP phi_peak")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_timephi, dpi=150)
        print("Saved:", args.out_timephi)

    # Extra plot: (chA - chB) vs (mcp_peak_time - mcp_trigger_time)
    if not x_dt_ch:
        print(f"No valid points for ch{ch_a}-ch{ch_b} vs mcp_peak-mcp_trigger plot.")
        if args.verbose:
            print("chdiff debug:", {
                "missing_chdiff_channels": counters["missing_chdiff_channels"],
                "missing_mcp_trigger": counters["missing_mcp_trigger"],
                "kept_chdiff": counters["kept_chdiff"],
            })
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(x_dt_ch, y_dt_mcp, s=10, alpha=0.6)
        plt.xlabel(f"channel {ch_a} time - channel {ch_b} time")
        plt.ylabel("mcp peak_time - mcp trigger_time")
        plt.title(f"Δt(ch{ch_a}-ch{ch_b}) vs Δt(mcp_peak-mcp_trigger)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dt_chdiff_vs_mcp, dpi=150)
        print("Saved:", args.out_dt_chdiff_vs_mcp)

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

    # Unified dt plot: ch133 - (peak + (ch192 - trigger))
    if not dt_unified:
        print("No valid points for unified dt plot (ch133 - peak_in_channel_frame).")
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(dt_unified, bins=args.dt_bins, alpha=0.75, color="steelblue", edgecolor="white")
        plt.xlabel("ch133 - (mcp_peak + (ch192 - mcp_trigger))")
        plt.ylabel("counts")
        plt.title("Unified dt histogram")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dt_ch133_peak_unified, dpi=150)
        print("Saved:", args.out_dt_ch133_peak_unified)

    # Phase-wrapped dt plot (phi period = 6250)
    if not dt_phi:
        print("No valid points for dt phi plot (wrapped by 6250).")
    else:
        plt.figure(figsize=(6.5, 4.5))
        counts, bins, _ = plt.hist(dt_phi, bins=args.dt_bins, alpha=0.75, color="slateblue", edgecolor="white")
        # Gaussian fit on phi histogram
        try:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            mask = counts > 0
            x_fit = bin_centers[mask]
            y_fit = counts[mask]
            if x_fit.size >= 3:
                mu0 = float(np.mean(dt_phi))
                sigma0 = float(np.std(dt_phi, ddof=1))
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
            print("Gaussian fit failed for dt phi plot:", e)
        plt.xlabel("dt phase (wrapped by 6250)")
        plt.ylabel("counts")
        plt.title("Unified dt phase (phi=6250)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dt_phi, dpi=150)
        print("Saved:", args.out_dt_phi)

    # Energy histogram for plot_channel
    if not energy_vals:
        print(f"No valid energy values for channel {plot_channel}.")
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(energy_vals, bins=args.energy_bins, alpha=0.75, color="darkgoldenrod", edgecolor="white")
        plt.xlabel(f"channel {plot_channel} energy")
        plt.ylabel("counts")
        plt.title(f"Energy histogram (channel {plot_channel})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_energy, dpi=150)
        print("Saved:", args.out_energy)

    # Fitted method: use global linear fit ch192 = m*trigger + b, then
    # peak_in_channel_frame = m*peak_time + b, dt = ch133 - peak_in_channel_frame
    if x_ch192 and y_trig:
        x = np.asarray(y_trig, dtype=float)  # trigger_time
        y = np.asarray(x_ch192, dtype=float)  # ch192 time
        if x.size >= 2:
            try:
                m_fit, b_fit = np.polyfit(x, y, 1)
                # compute dt for events where we have ch133, peak_time, trigger_time
                for i in range(max_e):
                    # reuse arrays to avoid double reading; re-derive per event
                    try:
                        ch_list = ak.to_list(arrays[args.branch_channel][i])
                        time_list = ak.to_list(arrays[args.branch_time][i])
                    except Exception:
                        continue
                    try:
                        pos_133 = ch_list.index(133)
                        ch133_t = float(time_list[pos_133])
                    except Exception:
                        continue
                    if i not in mcp_map:
                        continue
                    peak_time, _, trig_time = mcp_map[i]
                    if not (ch133_t == ch133_t and peak_time == peak_time):
                        continue
                    peak_in_ch_frame_fit = m_fit * peak_time + b_fit
                    dt_fit.append(ch133_t - peak_in_ch_frame_fit)
            except Exception as e:
                print("Global linear fit failed for ch192 vs trigger_time:", e)

    if not dt_fit:
        print("No valid points for fitted dt plot (global ch192 vs trigger).")
    else:
        plt.figure(figsize=(6.5, 4.5))
        plt.hist(dt_fit, bins=args.dt_bins, alpha=0.75, color="seagreen", edgecolor="white")
        plt.xlabel("ch133 - (m_fit * mcp_peak + b_fit)")
        plt.ylabel("counts")
        plt.title("Fitted dt histogram (global ch192 vs trigger)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dt_ch133_peak_fit, dpi=150)
        print("Saved:", args.out_dt_ch133_peak_fit)



if __name__ == "__main__":
    main()
