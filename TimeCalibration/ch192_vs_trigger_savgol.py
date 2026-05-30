#!/usr/bin/env python3
"""
Side-branch calibration test: Savitzky-Golay filter variant.

This script is a parallel implementation of ch192_vs_trigger.py that replaces the
binned-median residual correction with a Savitzky-Golay (SavGol) filter for
baseline drift cancellation.

Calibration approach per segment:
  1. Linear fit: trigger_time = m * ch192_aligned + b  (same as original)
  2. Compute residuals from linear fit                  (same as original)
  3. Sort residuals by trigger_time, apply SavGol filter to extract baseline
  4. Calibrate: ch192_cal = ch192_aligned + savgol_baseline / m

Example:
  python3 ch192_vs_trigger_savgol.py /path/to/4405_*_e.root --mcp-internal-dt-cut
"""

import argparse
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import uproot
import awkward as ak

# ---- Reuse from the main workflow & original script ----
from bar_helpers import find_data_tree, gauss, log
from bar_processing import _mcp_internal_dt_selector
from bar_plotting import plot_t_diff

from ch192_vs_trigger import (
    build_mcp_map,
    extract_ch_times,
    detect_segments,
    _process_one_file,
    _gauss_fit_hist,
)


# ──────────────────────────────────────────────────────────────────
# Argument parsing  (extends original with SavGol-specific options)
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Side-branch test: ch192 vs MCP trigger_time (SavGol filter)"
    )
    p.add_argument("file", nargs="+",
                   help="Input ROOT file(s), supports glob patterns")
    # channel selection
    p.add_argument("--channel", type=int, default=192,
                   help="Channel to compare against MCP trigger_time (default: 192)")
    # branch names
    p.add_argument("--branch-channel", default="channelID")
    p.add_argument("--branch-time", default="time")
    # MCP tree / branches
    p.add_argument("--mcp-tree", default="MCP")
    p.add_argument("--mcp-index", default="index")
    p.add_argument("--mcp-peak-time", default="peak_time")
    p.add_argument("--mcp-trigger-time", default="trigger_time")
    p.add_argument("--mcp-peak-amp", default="peak_amp")
    # MCP quality cuts
    p.add_argument("--mcp-peak-amp-min", type=float, default=None)
    p.add_argument("--mcp-peak-amp-max", type=float, default=None)
    p.add_argument("--mcp-internal-dt-cut", action="store_true",
                   help="Apply robust internal dt cut on MCP entries")
    p.add_argument("--mcp-internal-dt-nmad", type=float, default=3.0,
                   help="Number of MAD widths for internal dt cut (default: 3)")
    # processing
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--step-size", type=int, default=200000,
                   help="Chunk size for vectorised processing (default: 200000)")
    p.add_argument("--nworkers", type=int, default=1,
                   help="Number of parallel workers for file processing (default: 1)")
    # output
    p.add_argument("--out-prefix", default="ch192_vs_trig_sg",
                   help="Prefix for all output plot filenames (default: ch192_vs_trig_sg)")
    p.add_argument("--nbins", type=int, default=120)
    p.add_argument("--fit-degree", type=int, default=3,
                   help="(Unused, kept for CLI compat) Polynomial degree (default: 3)")
    p.add_argument("--val-channel", type=int, default=137,
                   help="Validation channel to compare (default: 137)")
    p.add_argument("--verbose", action="store_true")
    # ── SavGol-specific ──
    p.add_argument("--savgol-window", type=int, default=51,
                   help="SavGol filter window length (must be odd, default: 51)")
    p.add_argument("--savgol-polyorder", type=int, default=3,
                   help="SavGol filter polynomial order (default: 3)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Per-file analysis with SavGol calibration
# ──────────────────────────────────────────────────────────────────

def _per_file_analysis_savgol(ch_time, mcp_trig, mcp_peak, val_time, val_bar_ok,
                              val_energy, prefix, ch, val_ch, nbins,
                              savgol_window=51, savgol_polyorder=3):
    """
    Per-file segment analysis with SavGol filter residual calibration.

    Calibration approach per segment:
      1. Linear fit: trigger = m * ch192_aligned + b  (iterative ±3 MAD)
      2. Residual = trigger - (m * ch192_aligned + b)
      3. Sort residuals by trigger_time, apply SavGol filter
      4. ch192_cal = ch192_aligned + savgol_baseline / m
    """
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fit_colors = ["red", "darkgreen", "darkblue", "purple", "brown"]
    n = len(ch_time)

    # ── Detect segments ──
    seg_masks = detect_segments(mcp_trig, ch_time)
    n_seg = len(seg_masks)

    # ── Per-segment alignment (identical to original) ──
    ch_time_aligned = ch_time.copy()
    seg_offsets = []
    for s, mask in enumerate(seg_masks):
        x_s = ch_time[mask]
        y_s = mcp_trig[mask]
        if len(x_s) > 0 and len(y_s) > 0:
            offset = float(x_s[0] - y_s[0])
            seg_offsets.append(offset)
            ch_time_aligned[mask] = ch_time[mask] - offset
            log(f"Segment {s+1}: aligned ch{ch} by subtracting offset = {offset:.2f} ps")
        else:
            seg_offsets.append(0.0)

    # ── Plot 1: Combined scatter (all segments) ──
    plt.figure(figsize=(8, 6))
    seg_slopes = []
    for s, mask in enumerate(seg_masks):
        x_s, y_s = ch_time_aligned[mask], mcp_trig[mask]
        c = colors[s % len(colors)]
        plt.scatter(y_s, x_s, s=4, alpha=0.4, color=c, label=f"Seg {s+1} ({len(x_s)} evts)")
        if len(x_s) >= 2:
            x0, y0 = float(np.mean(x_s)), float(np.mean(y_s))
            m, b = np.polyfit(y_s - y0, x_s - x0, 1)
            seg_slopes.append(m)
            t_line = np.linspace(y_s.min(), y_s.max(), 200)
            ch_line = m * (t_line - y0) + b + x0
            fc = fit_colors[s % len(fit_colors)]
            plt.plot(t_line, ch_line, color=fc, linewidth=2,
                     label=f"Fit {s+1}: slope={m:.6f}")
        else:
            seg_slopes.append(None)
    plt.xlabel("MCP trigger_time (ps)")
    plt.ylabel(f"Channel {ch} time − aligned (ps)")
    plt.title(f"[SavGol] MCP trigger_time vs Channel {ch} (aligned)  ({n_seg} segments)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter.png")

    # ── Plot 2: Per-segment scatter subplots ──
    fig, axes = plt.subplots(1, n_seg, figsize=(7 * n_seg, 5), squeeze=False)
    for s, mask in enumerate(seg_masks):
        ax = axes[0][s]
        x_s, y_s = ch_time_aligned[mask], mcp_trig[mask]
        c = colors[s % len(colors)]
        ax.scatter(y_s, x_s, s=4, alpha=0.4, color=c)
        if len(x_s) >= 2:
            x0, y0 = float(np.mean(x_s)), float(np.mean(y_s))
            m, b = np.polyfit(y_s - y0, x_s - x0, 1)
            t_line = np.linspace(y_s.min(), y_s.max(), 200)
            ch_line = m * (t_line - y0) + b + x0
            ax.plot(t_line, ch_line, color="red", linewidth=2,
                    label=f"slope={m:.6f}")
            ax.legend()
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"Channel {ch} time − aligned (ps)")
        ax.set_title(f"Segment {s+1}  ({len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments.png")

    # ── Per-segment residual analysis + SavGol calibration ──
    ch_time_cal = ch_time_aligned.copy()
    all_resid = []
    all_resid_cal = []
    seg_sigmas = []
    seg_sigmas_cal = []

    for s, mask in enumerate(seg_masks):
        x_s, y_s = ch_time_aligned[mask], mcp_trig[mask]
        if len(x_s) < 3:
            seg_sigmas.append(None)
            seg_sigmas_cal.append(None)
            continue

        # ── Linear fit (iterative ±3 MAD): ch192 = m * trigger + b ──
        x0, y0 = float(np.mean(x_s)), float(np.mean(y_s))
        fit_idx = np.ones(len(x_s), dtype=bool)
        m, b = np.polyfit(y_s - y0, x_s - x0, 1)
        for _it in range(3):
            resid_all = (x_s - x0) - (m * (y_s - y0) + b)
            r_med = np.median(resid_all[fit_idx])
            r_mad = np.median(np.abs(resid_all[fit_idx] - r_med))
            if r_mad > 0:
                fit_idx = np.abs(resid_all - r_med) < 3 * 1.4826 * r_mad
            else:
                break
            if fit_idx.sum() < 2:
                break
            m, b = np.polyfit(y_s[fit_idx] - y0, x_s[fit_idx] - x0, 1)

        # Compute residuals on ALL events using the cleaned fit
        resid = (x_s - x0) - (m * (y_s - y0) + b)
        resid -= np.mean(resid[fit_idx])
        all_resid.append(resid)
        log(f"Segment {s+1}: linear slope={m:.6f}, fit on {fit_idx.sum()}/{len(x_s)} events")

        # ── Residual histogram (before calibration) ──
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad > 0:
            clean = resid[np.abs(resid - med) < 5 * 1.4826 * mad]
        else:
            clean = resid
        if len(clean) < 5:
            clean = resid

        sig = plot_t_diff(
            clean.tolist(), f"{prefix}_residuals_seg{s+1}.png",
            f"[SavGol] Residuals (pre-cal): Seg {s+1}  (slope={m:.6f}, "
            f"N={len(clean)}/{len(resid)})",
            nbins=min(nbins, max(30, len(clean) // 15)), xlabel="Residual (ps)"
        )
        seg_sigmas.append(sig)
        if sig is not None:
            log(f"Segment {s+1} pre-cal residual sigma = {sig:.2f} ps")

        # ── SavGol filter correction (outlier-cleaned) ──
        # Use fit_idx to exclude outliers before applying SavGol,
        # then interpolate the baseline back to ALL events.
        from scipy.interpolate import interp1d

        clean_mask = fit_idx.copy()
        n_clean = int(clean_mask.sum())
        savgol_baseline = np.zeros(len(resid))

        if n_clean >= savgol_polyorder + 2:
            # Sort CLEAN residuals by trigger_time
            y_clean = y_s[clean_mask]
            resid_clean = resid[clean_mask]
            sort_order_clean = np.argsort(y_clean)
            y_clean_sorted = y_clean[sort_order_clean]
            resid_clean_sorted = resid_clean[sort_order_clean]

            # Ensure window length is odd and does not exceed clean data length
            win = savgol_window
            if win > len(resid_clean_sorted):
                win = len(resid_clean_sorted)
            if win % 2 == 0:
                win -= 1
            if win < savgol_polyorder + 2:
                win = savgol_polyorder + 2
                if win % 2 == 0:
                    win += 1

            savgol_clean_sorted = savgol_filter(
                resid_clean_sorted, win, savgol_polyorder
            )

            # Interpolate the clean baseline back to ALL events
            interp_func = interp1d(
                y_clean_sorted, savgol_clean_sorted,
                kind="linear", fill_value="extrapolate"
            )
            savgol_baseline = interp_func(y_s)

            log(f"Segment {s+1}: SavGol filter applied on {n_clean}/{len(x_s)} "
                f"clean events (window={win}, polyorder={savgol_polyorder})")
        else:
            win = savgol_window
            log(f"Segment {s+1}: too few clean events ({n_clean}) for SavGol, "
                f"skipping correction")

        # For plotting: sorted by ALL trigger_time
        sort_order_all = np.argsort(y_s)
        y_sorted = y_s[sort_order_all]
        savgol_baseline_sorted_all = savgol_baseline[sort_order_all]

        # Plot: residual vs trigger_time with SavGol overlay
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot outliers in a different color
        ax.scatter(y_s[~clean_mask], resid[~clean_mask], s=2, alpha=0.3,
                   color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
        ax.scatter(y_s[clean_mask], resid[clean_mask], s=2, alpha=0.3,
                   color="steelblue", label=f"Clean ({clean_mask.sum()})")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        # Plot the SavGol baseline
        ax.plot(y_sorted, savgol_baseline_sorted_all,
                color="red", linewidth=2,
                label=f"SavGol (win={win}, order={savgol_polyorder})")
        ax.legend(fontsize=8)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[SavGol] Seg {s+1}: residual vs trigger_time  "
                     f"(N={len(resid)}, clean={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_vs_trig_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_vs_trig_seg{s+1}.png")

        # Plot: clean-only events with y-axis clipped to see filter detail
        resid_fit = resid[clean_mask]
        y_fit = y_s[clean_mask]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_fit, resid_fit, s=2, alpha=0.4, color="steelblue")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        # Plot the SavGol baseline (sorted by clean trigger_time)
        sort_fit = np.argsort(y_fit)
        ax.plot(y_fit[sort_fit], savgol_baseline[clean_mask][sort_fit],
                color="red", linewidth=2,
                label=f"SavGol (win={win}, order={savgol_polyorder})")
        ax.legend(fontsize=8)
        # Clip y-axis to clean data range with some padding
        if len(resid_fit) > 0:
            r_lo, r_hi = float(resid_fit.min()), float(resid_fit.max())
            pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
            ax.set_ylim(r_lo - pad, r_hi + pad)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[SavGol] Seg {s+1}: clean-only events  (N={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_fitonly_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_fitonly_seg{s+1}.png")

        # ── Calibrate ch192 (SavGol correction) ──
        if n_clean >= savgol_polyorder + 2:
            correction = savgol_baseline
            ch_time_cal[mask] = x_s - correction

            # Calibrated residuals (should be flat)
            resid_cal = resid - savgol_baseline
            all_resid_cal.append(resid_cal)

            # ±3 MAD clip for histogram
            cal_med = np.median(resid_cal)
            cal_mad = np.median(np.abs(resid_cal - cal_med))
            cal_cut = 3 * 1.4826 * cal_mad if cal_mad > 0 else np.std(resid_cal) * 3
            cal_clip = np.abs(resid_cal - cal_med) < cal_cut
            clean_cal = resid_cal[cal_clip]

            sig_cal = plot_t_diff(
                clean_cal.tolist(), f"{prefix}_residuals_cal_seg{s+1}.png",
                f"[SavGol] Residuals (post-cal): Seg {s+1} "
                f"(±3MAD, N={len(clean_cal)}/{len(resid_cal)})",
                nbins=min(nbins, max(30, len(clean_cal) // 15)),
                xlabel="Calibrated residual (ps)"
            )
            seg_sigmas_cal.append(sig_cal)
            if sig_cal is not None:
                log(f"Segment {s+1} post-cal (SavGol) residual sigma = {sig_cal:.2f} ps")

            # Plot: calibrated residual vs trigger_time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_s, resid_cal, s=2, alpha=0.3, color="steelblue")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Calibrated residual (ps)")
            ax.set_title(f"[SavGol] Seg {s+1}: calibrated residual vs trigger_time  "
                         f"(N={len(resid_cal)})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_vs_trig_cal_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_vs_trig_cal_seg{s+1}.png")

            # Plot: calibrated residual (clean-only, y-axis clipped)
            resid_cal_fit = resid_cal[clean_mask]
            y_cal_fit = y_s[clean_mask]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_cal_fit, resid_cal_fit, s=2, alpha=0.4, color="steelblue")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            if len(resid_cal_fit) > 0:
                r_lo, r_hi = float(resid_cal_fit.min()), float(resid_cal_fit.max())
                pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
                ax.set_ylim(r_lo - pad, r_hi + pad)
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Calibrated residual (ps)")
            ax.set_title(f"[SavGol] Seg {s+1}: cal residual clean-only  "
                         f"(N={int(clean_mask.sum())})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_cal_fitonly_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_cal_fitonly_seg{s+1}.png")

        else:
            seg_sigmas_cal.append(None)

    # ── Scatter segments after SavGol cal ──
    fig, axes = plt.subplots(1, n_seg, figsize=(7 * n_seg, 5), squeeze=False)
    for s, mask in enumerate(seg_masks):
        ax = axes[0][s]
        x_s, y_s = ch_time_cal[mask], mcp_trig[mask]
        c = colors[s % len(colors)]
        ax.scatter(y_s, x_s, s=4, alpha=0.4, color=c)
        if len(x_s) >= 2:
            x0t, y0t = float(np.mean(x_s)), float(np.mean(y_s))
            mt, bt = np.polyfit(y_s - y0t, x_s - x0t, 1)
            t_line = np.linspace(y_s.min(), y_s.max(), 200)
            ax.plot(t_line, mt * (t_line - y0t) + bt + x0t, color="red", linewidth=2,
                    label=f"slope={mt:.6f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"Channel {ch} SavGol cal (ps)")
        ax.set_title(f"Segment {s+1} (SavGol cal, {len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments_cal.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments_cal.png")

    # ── Per-file combined pre-cal residuals ──
    seg_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    if all_resid:
        combined_resid = np.concatenate(all_resid)
        med = np.median(combined_resid)
        mad = np.median(np.abs(combined_resid - med))
        if mad > 0:
            cut = 5 * 1.4826 * mad
            lo, hi = med - cut, med + cut
        else:
            lo, hi = combined_resid.min(), combined_resid.max()
        n_bins_comb = min(nbins, max(30, len(combined_resid) // 15))
        bin_edges = np.linspace(lo, hi, n_bins_comb + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for si, resid_s in enumerate(all_resid):
            clean_s = resid_s[(resid_s >= lo) & (resid_s <= hi)]
            ax.hist(clean_s, bins=bin_edges, alpha=0.5,
                    color=seg_colors[si % len(seg_colors)],
                    label=f"Seg {si+1} ({len(clean_s)} evts)")
        ax.set_xlabel("Residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title(f"[SavGol] Pre-cal residuals: {n_seg} segments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined.png")

    # ── Per-file combined cal residuals ──
    if all_resid_cal:
        comb_cal = np.concatenate(all_resid_cal)
        med_cc = np.median(comb_cal)
        mad_cc = np.median(np.abs(comb_cal - med_cc))
        if mad_cc > 0:
            cut_cc = 5 * 1.4826 * mad_cc
            lo_cc, hi_cc = med_cc - cut_cc, med_cc + cut_cc
        else:
            lo_cc, hi_cc = comb_cal.min(), comb_cal.max()
        n_bins_cc = min(nbins, max(30, len(comb_cal) // 15))
        bin_edges_cc = np.linspace(lo_cc, hi_cc, n_bins_cc + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for si, resid_ci in enumerate(all_resid_cal):
            clean_ci = resid_ci[(resid_ci >= lo_cc) & (resid_ci <= hi_cc)]
            ax.hist(clean_ci, bins=bin_edges_cc, alpha=0.5,
                    color=seg_colors[si % len(seg_colors)],
                    label=f"Seg {si+1} ({len(clean_ci)} evts)")
        ax.set_xlabel("Calibrated residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title(f"[SavGol] Post-cal residuals: {n_seg} segments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    # Recover calibrated ch192 to original (non-aligned) frame
    ch_time_cal_orig = ch_time_cal.copy()
    for s, mask_val in enumerate(seg_masks):
        ch_time_cal_orig[mask_val] += seg_offsets[s]

    # ── Per-file validation: [ch_val − ch192] − [mcp_peak − trigger] ──
    val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
              & np.isfinite(ch_time) & np.isfinite(mcp_trig)
              & val_bar_ok)
    if val_ok.sum() >= 10:
        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_after = (val_time[val_ok] - ch_time_cal_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])

        def _mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        cb = _mad_clip(delta_before)
        ca = _mad_clip(delta_after)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − [mcp_peak−trigger] (ps)"
        nb = min(nbins, max(30, len(cb) // 15))

        res_b = _gauss_fit_hist(ax1, cb, nb, "tab:blue", val_xlabel,
                                "Before cal", len(delta_before))
        res_a = _gauss_fit_hist(ax2, ca, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_SG_cal] − [mcp_peak−trigger] (ps)",
                                "After SavGol cal", len(delta_after))

        fig.suptitle(f"[SavGol] Validation: [ch{val_ch} − ch{ch}] − [mcp_peak − trigger]",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(cb)
        sig_a = res_a[1] if res_a else np.std(ca)
        log(f"  Per-file validation: before σ={sig_b:.2f}, after σ={sig_a:.2f} ps")

    return {
        "ch_time_aligned": ch_time_aligned,
        "ch_time_cal_orig": ch_time_cal_orig,
        "seg_masks": seg_masks,
        "seg_offsets": seg_offsets,
        "seg_slopes": seg_slopes,
        "seg_sigmas": seg_sigmas,
        "seg_sigmas_cal": seg_sigmas_cal,
        "all_resid": all_resid,
        "all_resid_cal": all_resid_cal,
        "n_seg": n_seg,
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve file globs
    paths = []
    for pattern in args.file:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(pattern)
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No valid input files found.")
        sys.exit(1)
    log(f"[SavGol] Processing {len(paths)} file(s)")

    cfg = {
        "channel": args.channel,
        "branch_channel": args.branch_channel,
        "branch_time": args.branch_time,
        "mcp_tree": args.mcp_tree,
        "mcp_index": args.mcp_index,
        "mcp_peak_time": args.mcp_peak_time,
        "mcp_trigger_time": args.mcp_trigger_time,
        "mcp_peak_amp": args.mcp_peak_amp,
        "mcp_peak_amp_min": args.mcp_peak_amp_min,
        "mcp_peak_amp_max": args.mcp_peak_amp_max,
        "mcp_internal_dt_cut": args.mcp_internal_dt_cut,
        "mcp_internal_dt_nmad": args.mcp_internal_dt_nmad,
        "val_channel": args.val_channel,
        "max_entries": args.max_entries,
        "step_size": args.step_size,
    }

    # ── Phase 1: Collect data per file ──
    data_keys = ["ch_time", "mcp_trig_time", "mcp_peak_time", "event_idx",
                 "val_time", "val_bar_ok", "val_energy"]
    per_file_data = []

    nw = min(args.nworkers, len(paths))
    if nw > 1:
        log(f"Using {nw} parallel workers")
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futures = {pool.submit(_process_one_file, p, cfg): p for p in paths}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    data = fut.result()
                except Exception as e:
                    log(f"  Error processing {p}: {e}")
                    continue
                if data is None:
                    log(f"  Skipping {p} (no valid MCP entries)")
                    continue
                log(f"  Done: {p}")
                per_file_data.append((p, data))
    else:
        for path in paths:
            log(f"Opening: {path}")
            f = uproot.open(path)
            mcp_map = build_mcp_map(f, cfg)
            if not mcp_map:
                log(f"  Skipping (no valid MCP entries)")
                continue
            data = extract_ch_times(f, cfg, mcp_map)
            per_file_data.append((path, data))

    if not per_file_data:
        print("No valid files processed.")
        sys.exit(0)

    prefix = args.out_prefix
    nbins = args.nbins
    ch = args.channel
    val_ch = args.val_channel
    prefix_dir = os.path.dirname(prefix) or "."
    prefix_base = os.path.basename(prefix)
    sg_win = args.savgol_window
    sg_ord = args.savgol_polyorder

    # ── Phase 2: Per-file analysis (create directory per ROOT file) ──
    combined = {k: [] for k in data_keys + ["ch_time_cal_orig", "ch_time_aligned"]}
    all_resid_global = []
    all_resid_cal_global = []
    all_file_summary = []

    for file_path, data in per_file_data:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        file_dir = os.path.join(prefix_dir, basename)
        os.makedirs(file_dir, exist_ok=True)
        file_prefix = os.path.join(file_dir, prefix_base)

        fch = data["ch_time"]
        fmcp = data["mcp_trig_time"]
        fmpk = data["mcp_peak_time"]
        fval = data["val_time"]
        fbar = data["val_bar_ok"].astype(bool)
        feng = data["val_energy"]

        n_f = len(fch)
        log(f"\n{'='*60}")
        log(f"  [SavGol] File: {basename}  ({n_f} events)")
        log(f"  Output dir: {file_dir}/")
        log(f"{'='*60}")
        if n_f < 2:
            log(f"  Skipping (not enough events)")
            continue

        result = _per_file_analysis_savgol(
            fch, fmcp, fmpk, fval, fbar, feng,
            file_prefix, ch, val_ch, nbins,
            savgol_window=sg_win, savgol_polyorder=sg_ord
        )

        # Accumulate for combined analysis
        for k in data_keys:
            combined[k].append(data[k])
        combined["ch_time_cal_orig"].append(result["ch_time_cal_orig"])
        combined["ch_time_aligned"].append(result["ch_time_aligned"])
        all_resid_global.extend(result["all_resid"])
        all_resid_cal_global.extend(result["all_resid_cal"])
        all_file_summary.append((basename, result["n_seg"], n_f,
                                 result["seg_slopes"], result["seg_sigmas"],
                                 result["seg_sigmas_cal"]))

    # ── Phase 3: Concatenate all data ──
    for k in combined:
        combined[k] = np.concatenate(combined[k]) if combined[k] else np.array([])

    n = len(combined["ch_time"])
    log(f"\n[SavGol] Total matched events across all files: {n}")
    if n < 2:
        print("Not enough matched events to produce combined plots.")
        sys.exit(0)

    ch_time = combined["ch_time"]
    ch_time_aligned = combined["ch_time_aligned"]
    ch_time_cal_orig = combined["ch_time_cal_orig"]
    mcp_trig = combined["mcp_trig_time"]
    mcp_peak = combined["mcp_peak_time"]
    val_time = combined["val_time"]
    val_bar_ok = combined["val_bar_ok"].astype(bool)
    val_energy = combined["val_energy"]

    # ────────────────────────────────────────────────────
    # Combined residuals (all files, all segments)
    # ────────────────────────────────────────────────────
    seg_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    if all_resid_global:
        combined_resid = np.concatenate(all_resid_global)
        med = np.median(combined_resid)
        mad = np.median(np.abs(combined_resid - med))
        if mad > 0:
            cut = 5 * 1.4826 * mad
            lo, hi = med - cut, med + cut
        else:
            lo, hi = combined_resid.min(), combined_resid.max()
        n_bins_comb = min(nbins, max(30, len(combined_resid) // 15))
        bin_edges = np.linspace(lo, hi, n_bins_comb + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for s, resid_s in enumerate(all_resid_global):
            clean_s = resid_s[(resid_s >= lo) & (resid_s <= hi)]
            ax.hist(clean_s, bins=bin_edges, alpha=0.5,
                    color=seg_colors[s % len(seg_colors)],
                    label=f"Seg {s+1} ({len(clean_s)} evts)")
        ax.set_xlabel("Residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title(f"[SavGol] Pre-cal residuals: all {len(all_resid_global)} segments combined")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined.png")

    if all_resid_cal_global:
        comb_cal = np.concatenate(all_resid_cal_global)
        med_cc = np.median(comb_cal)
        mad_cc = np.median(np.abs(comb_cal - med_cc))
        if mad_cc > 0:
            cut_cc = 5 * 1.4826 * mad_cc
            lo_cc, hi_cc = med_cc - cut_cc, med_cc + cut_cc
        else:
            lo_cc, hi_cc = comb_cal.min(), comb_cal.max()
        n_bins_cc = min(nbins, max(30, len(comb_cal) // 15))
        bin_edges_cc = np.linspace(lo_cc, hi_cc, n_bins_cc + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for si, resid_ci in enumerate(all_resid_cal_global):
            clean_ci = resid_ci[(resid_ci >= lo_cc) & (resid_ci <= hi_cc)]
            ax.hist(clean_ci, bins=bin_edges_cc, alpha=0.5,
                    color=seg_colors[si % len(seg_colors)],
                    label=f"Seg {si+1} ({len(clean_ci)} evts)")
        ax.set_xlabel("Calibrated residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title(f"[SavGol] Post-cal residuals: all segments combined")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    # ────────────────────────────────────────────────────
    # Wrapped dt (combined)
    # ────────────────────────────────────────────────────
    dt = ch_time_aligned - mcp_trig
    dt_wrapped = ((dt + 3125.0) % 6250.0) - 3125.0
    sig_wrap = plot_t_diff(
        dt_wrapped.tolist(), f"{prefix}_dt_wrapped.png",
        f"[SavGol] Δt = ch{ch} − trigger_time (wrapped ±3125 ps)",
        nbins=nbins, xlabel=f"Wrapped Δt (ps)"
    )
    if sig_wrap is not None:
        log(f"Wrapped dt sigma = {sig_wrap:.2f} ps")

    # ────────────────────────────────────────────────────
    # Validation: [ch_val − ch192] − [mcp_peak − trigger]
    # ────────────────────────────────────────────────────
    log(f"\n=== [SavGol] Validation: ch{val_ch} vs ch{ch} ===")

    val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
              & np.isfinite(ch_time) & np.isfinite(mcp_trig)
              & val_bar_ok)
    log(f"Events with all 4 finite + bar channels present: {val_ok.sum()}/{len(val_ok)}")

    if val_ok.sum() >= 10:
        # Energy distribution for validation events
        val_e = val_energy[val_ok]
        val_e_finite = val_e[np.isfinite(val_e)]
        if len(val_e_finite) >= 5:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(val_e_finite, bins=min(nbins, max(30, len(val_e_finite) // 15)),
                    color="tab:orange", alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_xlabel(f"ch{val_ch} energy")
            ax.set_ylabel("Counts")
            ax.set_title(f"ch{val_ch} energy distribution (validation events, "
                         f"N={len(val_e_finite)})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_val_energy.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_val_energy.png")

        # Before calibration (original ch192, no alignment)
        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        # After SavGol calibration (ch192_cal recovered to original frame)
        delta_after = (val_time[val_ok] - ch_time_cal_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])

        def mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        clean_before = mad_clip(delta_before)
        clean_after = mad_clip(delta_after)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − [mcp_peak−trigger] (ps)"
        nb = min(nbins, max(30, len(clean_before) // 15))

        res_b = _gauss_fit_hist(ax1, clean_before, nb, "tab:blue", val_xlabel,
                                "Before cal", len(delta_before))
        res_a = _gauss_fit_hist(ax2, clean_after, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_SG] − [mcp_peak−trigger] (ps)",
                                "After SavGol cal", len(delta_after))

        fig.suptitle(f"[SavGol] Validation: [ch{val_ch} − ch{ch}] − [mcp_peak − trigger]",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(clean_before)
        sig_a = res_a[1] if res_a else np.std(clean_after)
        log(f"Before cal: σ={sig_b:.2f} ps")
        log(f"After SavGol cal: σ={sig_a:.2f} ps")

    # ────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  [SavGol] ch{ch} vs MCP trigger_time  —  Summary")
    print("=" * 60)
    print(f"  SavGol window        : {sg_win}")
    print(f"  SavGol polyorder     : {sg_ord}")
    print(f"  Total matched events : {n}")
    print(f"  Files processed      : {len(all_file_summary)}")
    for fname, n_seg, n_evt, slopes, sigmas, sigmas_cal in all_file_summary:
        print(f"\n  ── {fname} ({n_evt} events, {n_seg} segments) ──")
        for s in range(n_seg):
            slope_s = slopes[s] if s < len(slopes) else None
            sig_s = sigmas[s] if s < len(sigmas) else None
            sig_c = sigmas_cal[s] if s < len(sigmas_cal) else None
            print(f"    Segment {s+1}:")
            if slope_s is not None:
                print(f"      Linear fit slope       : {slope_s:.6f}")
            if sig_s is not None:
                print(f"      Pre-cal resid σ        : {sig_s:.2f} ps")
            if sig_c is not None:
                print(f"      Post-SavGol-cal resid σ: {sig_c:.2f} ps")
    print("-" * 60)
    if sig_wrap is not None:
        print(f"  Wrapped Δt σ           : {sig_wrap:.2f} ps")
    print("=" * 60)


if __name__ == "__main__":
    main()
