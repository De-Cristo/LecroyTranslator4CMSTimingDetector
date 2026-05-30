#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
MCP time-walk analysis from ROOT MCP tree.

Input: ROOT file with MCP tree (same files as bar_analysis.py).
Output: time-walk scatter plot and fit (peak_time - trigger_time vs 1/|peak_amp|).
"""

import argparse
import os
import sys

try:
    import numpy as np
    import uproot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing Python dependency:", e)
    print("Install: pip install uproot numpy matplotlib")
    sys.exit(1)


def log(msg):
    print(f"[mcp] {msg}", flush=True)


def load_mcp_arrays(root_path, tree_name, branch_peak_time, branch_trigger_time, branch_peak_amp):
    f = uproot.open(root_path)
    if tree_name not in f:
        raise KeyError(f'MCP tree "{tree_name}" not found in {root_path}')
    t = f[tree_name]
    peak_time = t[branch_peak_time].array(library="np")
    trig_time = t[branch_trigger_time].array(library="np")
    peak_amp = t[branch_peak_amp].array(library="np")
    return peak_time, trig_time, peak_amp


def sigma_clip(vals, iters=3, nsig=3.0):
    mask = np.ones(len(vals), dtype=bool)
    for _ in range(iters):
        data = vals[mask]
        if len(data) < 2:
            break
        mu = np.median(data)
        std = np.std(data)
        if std == 0:
            break
        mask = mask & (np.abs(vals - mu) < nsig * std)
    return mask


def main():
    p = argparse.ArgumentParser(description="MCP time-walk analysis")
    p.add_argument("file", help="Input ROOT file")
    p.add_argument("--mcp-tree", default="MCP", help="MCP tree name (default: MCP)")
    p.add_argument("--mcp-peak-time", default="peak_time", help="MCP peak time branch")
    p.add_argument("--mcp-trigger-time", default="trigger_time", help="MCP trigger time branch")
    p.add_argument("--mcp-peak-amp", default="peak_amp", help="MCP peak amp branch")
    p.add_argument("--out", default="mcp_timewalk.png", help="Output plot path")
    p.add_argument("--nbins", type=int, default=50, help="Residual histogram bins")
    p.add_argument("--sigma-iters", type=int, default=3, help="Sigma-clip iterations")
    p.add_argument("--sigma", type=float, default=3.0, help="Sigma-clip threshold")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print("File not found:", args.file)
        sys.exit(2)

    log(f"Opening ROOT: {args.file}")
    peak_time, trig_time, peak_amp = load_mcp_arrays(
        args.file, args.mcp_tree, args.mcp_peak_time, args.mcp_trigger_time, args.mcp_peak_amp
    )

    peak_time = np.asarray(peak_time, dtype=float)
    trig_time = np.asarray(trig_time, dtype=float)
    peak_amp = np.asarray(peak_amp, dtype=float)

    valid = (
        (peak_time == peak_time)
        & (trig_time == trig_time)
        & (peak_amp == peak_amp)
        & (np.abs(peak_amp) > 1e-6)
    )
    if not np.any(valid):
        print("No valid MCP entries found for time-walk plot.")
        sys.exit(3)

    dt = peak_time[valid] - trig_time[valid]
    inv_amp = 1.0 / np.abs(peak_amp[valid])

    log(f"Valid MCP points: {len(dt)}")
    mask = sigma_clip(dt, iters=args.sigma_iters, nsig=args.sigma)
    dt_clean = dt[mask]
    inv_clean = inv_amp[mask]
    log(f"After sigma-clip: {len(dt_clean)}/{len(dt)}")

    slope = np.nan
    intercept = np.nan
    residuals = []
    if len(dt_clean) > 2:
        slope, intercept = np.polyfit(inv_clean, dt_clean, 1)
        fit_fn = np.poly1d([slope, intercept])
        residuals = dt_clean - fit_fn(inv_clean)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(inv_amp, dt, s=5, alpha=0.2, label="All data", color="gray")
    ax1.scatter(inv_clean, dt_clean, s=5, alpha=0.6, label="Cleaned", color="tab:green")
    if slope == slope:
        x_line = np.linspace(inv_clean.min(), inv_clean.max(), 100)
        ax1.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
                 label=f"Fit: {slope:.2f}*x + {intercept:.2f}")
    ax1.set_xlabel("1 / |peak_amp| (1/V)")
    ax1.set_ylabel("peak_time − trigger_time (ps)")
    ax1.set_title(f"Time Walk\nSlope={slope:.4f} ps*V")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if len(residuals) > 0:
        ax2.hist(residuals, bins=args.nbins, density=True, alpha=0.6, color="teal")
    ax2.set_xlabel("Residuals (ps)")
    ax2.set_title("Residuals of Linear Fit")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    log(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
