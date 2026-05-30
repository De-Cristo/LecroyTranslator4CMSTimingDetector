#!/usr/bin/env python3
"""
Side-branch calibration test: Wiener filter variant.

This script is a parallel implementation of ch192_vs_trigger.py that replaces the
binned-median residual correction with a Wiener filter for simultaneous baseline
drift tracking AND white noise suppression.

Unlike the SavGol/binned-median methods which only *shift* the baseline, the Wiener
filter is a true noise-cancelling filter that reduces event-by-event jitter by
optimally weighting each event's residual against its temporal neighbors.

Calibration approach per segment:
  1. Linear fit: ch192_aligned = m * trigger_time + b  (same as original)
  2. Compute residuals from linear fit                  (same as original)
  3. Sort residuals by trigger_time, apply Wiener filter
  4. ch192_cal = ch192_aligned - wiener_filtered_baseline

Example:
  python3 ch192_vs_trigger_wiener.py /path/to/4405_*_e.root --mcp-internal-dt-cut
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
from scipy.signal import wiener, savgol_filter
from scipy.interpolate import interp1d

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
# Argument parsing
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Side-branch test: ch192 vs MCP trigger_time (Wiener filter)"
    )
    p.add_argument("file", nargs="+",
                   help="Input ROOT file(s), supports glob patterns")
    p.add_argument("--channel", type=int, default=192)
    p.add_argument("--branch-channel", default="channelID")
    p.add_argument("--branch-time", default="time")
    p.add_argument("--mcp-tree", default="MCP")
    p.add_argument("--mcp-index", default="index")
    p.add_argument("--mcp-peak-time", default="peak_time")
    p.add_argument("--mcp-trigger-time", default="trigger_time")
    p.add_argument("--mcp-peak-amp", default="peak_amp")
    p.add_argument("--mcp-peak-amp-min", type=float, default=None)
    p.add_argument("--mcp-peak-amp-max", type=float, default=None)
    p.add_argument("--mcp-internal-dt-cut", action="store_true")
    p.add_argument("--mcp-internal-dt-nmad", type=float, default=3.0)
    p.add_argument("--max-entries", type=int, default=None)
    p.add_argument("--step-size", type=int, default=200000)
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--out-prefix", default="ch192_vs_trig_wf",
                   help="Prefix for output plots (default: ch192_vs_trig_wf)")
    p.add_argument("--nbins", type=int, default=120)
    p.add_argument("--val-channel", type=int, default=137)
    p.add_argument("--verbose", action="store_true")
    # ── Wiener-specific ──
    p.add_argument("--wiener-window", type=int, default=51,
                   help="Wiener filter window size (default: 51)")
    # ── SavGol for drift removal (step 2) ──
    p.add_argument("--savgol-window", type=int, default=51,
                   help="SavGol window for drift removal after Wiener (default: 51)")
    p.add_argument("--savgol-polyorder", type=int, default=3,
                   help="SavGol polynomial order for drift removal (default: 3)")
    return p.parse_args()


def _save_validation_trend_plot(prefix, trig_time, series, val_ch, ch, title_prefix):
    labels = [
        ("Before cal", "tab:blue"),
        ("Wiener only", "tab:orange"),
        ("Wiener + SavGol", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), squeeze=False)
    axes = axes[0]

    for ax, (values, (label, color)) in zip(axes, zip(series, labels)):
        ax.scatter(trig_time, values, s=2, alpha=0.3, color=color)
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
        if len(values) > 0:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            if mad > 0:
                pad = 5 * 1.4826 * mad
                ax.set_ylim(med - pad, med + pad)
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"ch{val_ch} - t_MCP (ps)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{title_prefix}: ch{val_ch} - t_MCP, where "
        f"t_MCP = ch{ch}_variant + (mcp_peak - trigger)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_validation_vs_trig.png", dpi=150)
    plt.close(fig)
    log(f"Saved: {prefix}_validation_vs_trig.png")


def _save_validation_tmcp_hist(prefix, series, val_ch, ch, nbins, title_prefix):
    labels = [
        ("Before cal", "tab:blue"),
        ("Wiener only", "tab:orange"),
        ("Wiener + SavGol", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), squeeze=False)
    axes = axes[0]

    for ax, (values, (label, color)) in zip(axes, zip(series, labels)):
        clean = values[np.isfinite(values)]
        if len(clean) > 0:
            med = np.median(clean)
            mad = np.median(np.abs(clean - med))
            if mad > 0:
                cut = 3 * 1.4826 * mad
                clean = clean[np.abs(clean - med) < cut]
        n_bins = min(nbins, max(30, len(clean) // 15)) if len(clean) > 0 else nbins
        _gauss_fit_hist(
            ax,
            clean,
            n_bins,
            color,
            f"ch{val_ch} - t_MCP (ps)",
            label,
            len(values),
        )

    fig.suptitle(
        f"{title_prefix}: ch{val_ch} - t_MCP, where "
        f"t_MCP = ch{ch}_variant + (mcp_peak - trigger)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_validation_tmcp.png", dpi=150)
    plt.close(fig)
    log(f"Saved: {prefix}_validation_tmcp.png")


def _save_inverse_energy_scatter(prefix, energy, series, val_ch, ch, title_prefix):
    labels = [
        ("Before cal", "tab:blue"),
        ("Wiener only", "tab:orange"),
        ("Wiener + SavGol", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), squeeze=False)
    axes = axes[0]

    for ax, (values, (label, color)) in zip(axes, zip(series, labels)):
        mask = np.isfinite(energy) & np.isfinite(values) & (energy > 0)
        x = 1.0 / energy[mask]
        y = values[mask]
        inv_e_mask = x < 0.01
        x = x[inv_e_mask]
        y = y[inv_e_mask]
        fit_mask = np.ones(len(y), dtype=bool)
        if len(x) >= 2:
            y_med = np.median(y)
            y_mad = np.median(np.abs(y - y_med))
            if y_mad > 0:
                fit_cut = 5 * 1.4826 * y_mad
                fit_mask = np.abs(y - y_med) < fit_cut

        x_fit = x[fit_mask]
        y_fit = y[fit_mask]

        ax.scatter(x_fit, y_fit, s=3, alpha=0.3, color=color)
        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")

        if len(x_fit) >= 2:
            x0 = float(np.mean(x_fit))
            y0 = float(np.mean(y_fit))
            m, b = np.polyfit(x_fit - x0, y_fit - y0, 1)
            x_line = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 200)
            y_line = m * (x_line - x0) + b + y0
            ax.plot(
                x_line,
                y_line,
                color="red",
                linewidth=2,
                label=f"slope={m:.3e}, N={len(x_fit)}",
            )
            ax.legend(fontsize=8)

        ax.set_xlabel(f"1 / E(ch{val_ch})")
        ax.set_ylabel(f"t(ch{val_ch}) - t(ch{ch}_variant) (ps)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{title_prefix}: t(ch{val_ch}) - t(ch{ch}) versus inverse energy",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{prefix}_validation_invE.png", dpi=150)
    plt.close(fig)
    log(f"Saved: {prefix}_validation_invE.png")


# ──────────────────────────────────────────────────────────────────
# Per-file analysis with Wiener filter calibration
# ──────────────────────────────────────────────────────────────────

def _per_file_analysis_wiener(ch_time, mcp_trig, mcp_peak, val_time, val_bar_ok,
                              val_energy, prefix, ch, val_ch, nbins,
                              wiener_window=51, savgol_window=51,
                              savgol_polyorder=3):
    """
    Per-file segment analysis with Wiener filter residual calibration.

    The Wiener filter both tracks the baseline drift AND suppresses the
    event-by-event jitter. Each event's residual is replaced by the
    Wiener-filtered value, which is an optimally smoothed estimate.
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
    plt.title(f"[Wiener] MCP trigger_time vs Channel {ch} (aligned)  ({n_seg} segments)")
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
            ax.plot(t_line, ch_line, color="red", linewidth=2, label=f"slope={m:.6f}")
            ax.legend()
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"Channel {ch} time − aligned (ps)")
        ax.set_title(f"Segment {s+1}  ({len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments.png")

    # ── Per-segment residual analysis + Wiener calibration ──
    ch_time_cal = ch_time_aligned.copy()      # final: Wiener + SavGol
    ch_time_cal_wf = ch_time_aligned.copy()   # intermediate: Wiener only
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

        # Compute residuals on ALL events
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
            f"[Wiener] Residuals (pre-cal): Seg {s+1}  (slope={m:.6f}, "
            f"N={len(clean)}/{len(resid)})",
            nbins=min(nbins, max(30, len(clean) // 15)), xlabel="Residual (ps)"
        )
        seg_sigmas.append(sig)
        if sig is not None:
            log(f"Segment {s+1} pre-cal residual sigma = {sig:.2f} ps")

        # ── Wiener filter: apply to RESIDUALS, use as REPLACEMENT (noise suppression) ──
        # Apply Wiener to detrended residuals (~100ps scale, not raw ch192 ~10^11ps).
        # Use filtered residuals as REPLACEMENT: ch192_cal = predicted + wiener(resid)
        # This pulls each event's residual toward its neighbors (noise suppression)
        # while the Wiener curve itself tracks the drift (baseline correction).
        clean_mask = fit_idx.copy()
        n_clean = int(clean_mask.sum())
        wiener_resid = resid.copy()  # fallback: unfiltered

        if n_clean >= wiener_window:
            # Sort CLEAN residuals by trigger_time
            y_clean = y_s[clean_mask]
            resid_clean = resid[clean_mask]
            sort_order_clean = np.argsort(y_clean)
            y_clean_sorted = y_clean[sort_order_clean]
            resid_clean_sorted = resid_clean[sort_order_clean]

            # Apply Wiener filter to detrended residuals
            win = min(wiener_window, len(resid_clean_sorted))
            if win % 2 == 0:
                win -= 1
            wiener_clean_sorted = wiener(resid_clean_sorted, mysize=win)

            # Interpolate filtered residuals back to ALL events
            interp_func = interp1d(
                y_clean_sorted, wiener_clean_sorted,
                kind="linear", fill_value="extrapolate"
            )
            wiener_resid = interp_func(y_s)

            log(f"Segment {s+1}: Wiener filter applied on {n_clean}/{len(x_s)} "
                f"clean events (window={win})")
        else:
            win = wiener_window
            log(f"Segment {s+1}: too few clean events ({n_clean}) for Wiener, "
                f"skipping correction")

        # For plotting: sorted by ALL trigger_time
        sort_order_all = np.argsort(y_s)
        y_sorted = y_s[sort_order_all]
        wiener_resid_sorted_all = wiener_resid[sort_order_all]

        # Plot: pre-cal residual vs trigger_time (all events + Wiener overlay)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_s[~clean_mask], resid[~clean_mask], s=2, alpha=0.3,
                   color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
        ax.scatter(y_s[clean_mask], resid[clean_mask], s=2, alpha=0.3,
                   color="steelblue", label=f"Clean ({clean_mask.sum()})")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.plot(y_sorted, wiener_resid_sorted_all,
                color="red", linewidth=2,
                label=f"Wiener filtered (win={win})")
        ax.legend(fontsize=8)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[Wiener] Seg {s+1}: residual vs trigger_time  "
                     f"(N={len(resid)}, clean={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_vs_trig_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_vs_trig_seg{s+1}.png")

        # Plot: clean-only residuals with Wiener overlay (y-axis clipped)
        resid_fit = resid[clean_mask]
        y_fit = y_s[clean_mask]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_fit, resid_fit, s=2, alpha=0.4, color="steelblue",
                   label="Original residual")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        sort_fit = np.argsort(y_fit)
        ax.plot(y_fit[sort_fit], wiener_resid[clean_mask][sort_fit],
                color="red", linewidth=2,
                label=f"Wiener (win={win})")
        ax.legend(fontsize=8)
        if len(resid_fit) > 0:
            r_lo, r_hi = float(resid_fit.min()), float(resid_fit.max())
            pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
            ax.set_ylim(r_lo - pad, r_hi + pad)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[Wiener] Seg {s+1}: clean-only events  (N={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_fitonly_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_fitonly_seg{s+1}.png")

        # ── Calibrate ch192 (two-step: Wiener noise reduction + SavGol drift removal) ──
        # Step 1: Wiener replaces each residual with noise-reduced estimate
        # Step 2: SavGol estimates and subtracts the remaining drift curve
        if n_clean >= wiener_window:
            # predicted ch192 from linear fit
            predicted = x0 + m * (y_s - y0) + b

            # ── Step 1 result: Wiener-only calibration ──
            ch_time_cal_wf[mask] = predicted + wiener_resid

            # Plot: Wiener-only calibrated residual vs trigger_time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_s[~clean_mask], wiener_resid[~clean_mask], s=2, alpha=0.3,
                       color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
            ax.scatter(y_s[clean_mask], wiener_resid[clean_mask], s=2, alpha=0.3,
                       color="steelblue", label=f"Clean ({clean_mask.sum()})")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax.legend(fontsize=8)
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Wiener-filtered residual (ps)")
            ax.set_title(f"[Wiener] Seg {s+1}: after Wiener (before drift removal)  "
                         f"(N={len(wiener_resid)})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_vs_trig_wf_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_vs_trig_wf_seg{s+1}.png")

            # Plot: Wiener-only clean-only (y-axis clipped)
            wr_fit = wiener_resid[clean_mask]
            y_wr_fit = y_s[clean_mask]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_wr_fit, wr_fit, s=2, alpha=0.4, color="steelblue")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            if len(wr_fit) > 0:
                r_lo, r_hi = float(wr_fit.min()), float(wr_fit.max())
                pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
                ax.set_ylim(r_lo - pad, r_hi + pad)
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Wiener-filtered residual (ps)")
            ax.set_title(f"[Wiener] Seg {s+1}: Wiener-only clean-only  "
                         f"(N={int(clean_mask.sum())})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_wf_fitonly_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_wf_fitonly_seg{s+1}.png")

            # ── Step 2: SavGol drift removal from Wiener-filtered residuals ──
            wiener_resid_clean = wiener_resid[clean_mask]
            y_clean_cal = y_s[clean_mask]
            sort_cal = np.argsort(y_clean_cal)
            y_cal_sorted = y_clean_cal[sort_cal]
            wr_cal_sorted = wiener_resid_clean[sort_cal]

            sg_win = min(savgol_window, len(wr_cal_sorted))
            if sg_win % 2 == 0:
                sg_win -= 1
            sg_poly = min(savgol_polyorder, sg_win - 1)
            drift_estimate_sorted = savgol_filter(wr_cal_sorted, sg_win, sg_poly)

            # Interpolate drift curve back to ALL events
            drift_func = interp1d(
                y_cal_sorted, drift_estimate_sorted,
                kind="linear", fill_value="extrapolate"
            )
            drift_all = drift_func(y_s)

            # Plot: SavGol drift curve overlaid on Wiener scatter (clean-only)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_wr_fit, wr_fit, s=2, alpha=0.4, color="steelblue",
                       label="Wiener-filtered residual")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            sort_drift = np.argsort(y_wr_fit)
            ax.plot(y_cal_sorted, drift_estimate_sorted,
                    color="red", linewidth=2,
                    label=f"SavGol drift (win={sg_win}, poly={sg_poly})")
            ax.legend(fontsize=8)
            if len(wr_fit) > 0:
                r_lo, r_hi = float(wr_fit.min()), float(wr_fit.max())
                pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
                ax.set_ylim(r_lo - pad, r_hi + pad)
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Residual (ps)")
            ax.set_title(f"[Wiener+SG] Seg {s+1}: SavGol drift on Wiener scatter  "
                         f"(N={int(clean_mask.sum())})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_wf_sg_overlay_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_wf_sg_overlay_seg{s+1}.png")

            # Final calibrated ch192: linear prediction + (wiener_resid - drift)
            ch_time_cal[mask] = predicted + wiener_resid - drift_all

            # Final calibrated residual: noise-reduced AND drift-removed
            resid_cal = wiener_resid - drift_all
            all_resid_cal.append(resid_cal)

            log(f"Segment {s+1}: SavGol drift removal (win={sg_win}, poly={sg_poly}) "
                f"applied to Wiener-filtered residuals")

            # ±3 MAD clip for histogram
            cal_med = np.median(resid_cal)
            cal_mad = np.median(np.abs(resid_cal - cal_med))
            cal_cut = 3 * 1.4826 * cal_mad if cal_mad > 0 else np.std(resid_cal) * 3
            cal_clip = np.abs(resid_cal - cal_med) < cal_cut
            clean_cal = resid_cal[cal_clip]

            sig_cal = plot_t_diff(
                clean_cal.tolist(), f"{prefix}_residuals_cal_seg{s+1}.png",
                f"[Wiener+SG] Residuals (post-cal): Seg {s+1} "
                f"(±3MAD, N={len(clean_cal)}/{len(resid_cal)})",
                nbins=min(nbins, max(30, len(clean_cal) // 15)),
                xlabel="Calibrated residual (ps)"
            )
            seg_sigmas_cal.append(sig_cal)
            if sig_cal is not None:
                log(f"Segment {s+1} post-cal (Wiener+SG) residual sigma = {sig_cal:.2f} ps")

            # Plot: calibrated residual vs trigger_time
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(y_s, resid_cal, s=2, alpha=0.3, color="steelblue")
            ax.axhline(0, color="grey", linewidth=1, linestyle="--")
            ax.set_xlabel("Trigger time (ps)")
            ax.set_ylabel("Calibrated residual (ps)")
            ax.set_title(f"[Wiener+SG] Seg {s+1}: calibrated residual vs trigger_time  "
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
            ax.set_title(f"[Wiener+SG] Seg {s+1}: cal residual clean-only  "
                         f"(N={int(clean_mask.sum())})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_resid_cal_fitonly_seg{s+1}.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_resid_cal_fitonly_seg{s+1}.png")

        else:
            seg_sigmas_cal.append(None)

    # ── Scatter segments after Wiener cal ──
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
        ax.set_ylabel(f"Channel {ch} Wiener cal (ps)")
        ax.set_title(f"Segment {s+1} (Wiener cal, {len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments_cal.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments_cal.png")

    # ── Per-file combined residuals ──
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
        ax.set_title(f"[Wiener] Pre-cal residuals: {n_seg} segments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined.png")

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
        ax.set_title(f"[Wiener] Post-cal residuals: {n_seg} segments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    # Recover calibrated ch192 to original (non-aligned) frame
    ch_time_cal_orig = ch_time_cal.copy()
    ch_time_cal_wf_orig = ch_time_cal_wf.copy()
    for s, mask_val in enumerate(seg_masks):
        ch_time_cal_orig[mask_val] += seg_offsets[s]
        ch_time_cal_wf_orig[mask_val] += seg_offsets[s]

    # ── Per-file validation ──
    val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
              & np.isfinite(ch_time) & np.isfinite(mcp_trig)
              & val_bar_ok)
    if val_ok.sum() >= 10:
        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_wf = (val_time[val_ok] - ch_time_cal_wf_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_after = (val_time[val_ok] - ch_time_cal_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        trig_val = mcp_trig[val_ok]
        val_energy_ok = val_energy[val_ok]
        dt_before = val_time[val_ok] - ch_time[val_ok]
        dt_wf = val_time[val_ok] - ch_time_cal_wf_orig[val_ok]
        dt_after = val_time[val_ok] - ch_time_cal_orig[val_ok]

        def _mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        _save_validation_trend_plot(
            prefix,
            trig_val,
            [delta_before, delta_wf, delta_after],
            val_ch,
            ch,
            "[Wiener] Validation trend",
        )
        _save_validation_tmcp_hist(
            prefix,
            [delta_before, delta_wf, delta_after],
            val_ch,
            ch,
            nbins,
            "[Wiener] Validation projection",
        )
        _save_inverse_energy_scatter(
            prefix,
            val_energy_ok,
            [dt_before, dt_wf, dt_after],
            val_ch,
            ch,
            "[Wiener] Validation inverse-energy",
        )

        cb = _mad_clip(delta_before)
        cw = _mad_clip(delta_wf)
        ca = _mad_clip(delta_after)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − [mcp_peak−trigger] (ps)"
        nb = min(nbins, max(30, len(cb) // 15))

        res_b = _gauss_fit_hist(ax1, cb, nb, "tab:blue", val_xlabel,
                                "Before cal", len(delta_before))
        res_w = _gauss_fit_hist(ax2, cw, nb, "tab:orange",
                                f"[ch{val_ch}−ch{ch}_WF] − [mcp_peak−trigger] (ps)",
                                "Wiener only", len(delta_wf))
        res_a = _gauss_fit_hist(ax3, ca, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_WF+SG] − [mcp_peak−trigger] (ps)",
                                "Wiener + SavGol", len(delta_after))

        fig.suptitle(f"[Wiener] Validation: [ch{val_ch} − ch{ch}] − [mcp_peak − trigger]",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(cb)
        sig_w = res_w[1] if res_w else np.std(cw)
        sig_a = res_a[1] if res_a else np.std(ca)
        log(f"  Per-file validation: before σ={sig_b:.2f}, WF σ={sig_w:.2f}, WF+SG σ={sig_a:.2f} ps")

    return {
        "ch_time_aligned": ch_time_aligned,
        "ch_time_cal_orig": ch_time_cal_orig,
        "ch_time_cal_wf_orig": ch_time_cal_wf_orig,
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
    log(f"[Wiener] Processing {len(paths)} file(s)")

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
    wf_win = args.wiener_window

    # ── Phase 2: Per-file analysis ──
    combined = {k: [] for k in data_keys + ["ch_time_cal_orig", "ch_time_cal_wf_orig", "ch_time_aligned"]}
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
        log(f"  [Wiener] File: {basename}  ({n_f} events)")
        log(f"  Output dir: {file_dir}/")
        log(f"{'='*60}")
        if n_f < 2:
            log(f"  Skipping (not enough events)")
            continue

        result = _per_file_analysis_wiener(
            fch, fmcp, fmpk, fval, fbar, feng,
            file_prefix, ch, val_ch, nbins,
            wiener_window=wf_win,
            savgol_window=args.savgol_window,
            savgol_polyorder=args.savgol_polyorder
        )

        for k in data_keys:
            combined[k].append(data[k])
        combined["ch_time_cal_orig"].append(result["ch_time_cal_orig"])
        combined["ch_time_cal_wf_orig"].append(result["ch_time_cal_wf_orig"])
        combined["ch_time_aligned"].append(result["ch_time_aligned"])
        all_resid_global.extend(result["all_resid"])
        all_resid_cal_global.extend(result["all_resid_cal"])
        all_file_summary.append((basename, result["n_seg"], n_f,
                                 result["seg_slopes"], result["seg_sigmas"],
                                 result["seg_sigmas_cal"]))

    # ── Phase 3: Concatenate ──
    for k in combined:
        combined[k] = np.concatenate(combined[k]) if combined[k] else np.array([])

    n = len(combined["ch_time"])
    log(f"\n[Wiener] Total matched events across all files: {n}")
    if n < 2:
        print("Not enough matched events to produce combined plots.")
        sys.exit(0)

    ch_time = combined["ch_time"]
    ch_time_aligned = combined["ch_time_aligned"]
    ch_time_cal_orig = combined["ch_time_cal_orig"]
    ch_time_cal_wf_orig = combined["ch_time_cal_wf_orig"]
    mcp_trig = combined["mcp_trig_time"]
    mcp_peak = combined["mcp_peak_time"]
    val_time = combined["val_time"]
    val_bar_ok = combined["val_bar_ok"].astype(bool)
    val_energy = combined["val_energy"]

    # ── Combined residuals ──
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
        ax.set_title(f"[Wiener] Pre-cal residuals: all {len(all_resid_global)} segments combined")
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
        ax.set_title(f"[Wiener] Post-cal residuals: all segments combined")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    # ── Wrapped dt ──
    dt = ch_time_aligned - mcp_trig
    dt_wrapped = ((dt + 3125.0) % 6250.0) - 3125.0
    sig_wrap = plot_t_diff(
        dt_wrapped.tolist(), f"{prefix}_dt_wrapped.png",
        f"[Wiener] Δt = ch{ch} − trigger_time (wrapped ±3125 ps)",
        nbins=nbins, xlabel=f"Wrapped Δt (ps)"
    )
    if sig_wrap is not None:
        log(f"Wrapped dt sigma = {sig_wrap:.2f} ps")

    # ── Combined validation ──
    log(f"\n=== [Wiener] Validation: ch{val_ch} vs ch{ch} ===")

    val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
              & np.isfinite(ch_time) & np.isfinite(mcp_trig)
              & val_bar_ok)
    log(f"Events with all 4 finite + bar channels present: {val_ok.sum()}/{len(val_ok)}")

    if val_ok.sum() >= 10:
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

        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_wf = (val_time[val_ok] - ch_time_cal_wf_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_after = (val_time[val_ok] - ch_time_cal_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        trig_val = mcp_trig[val_ok]
        val_energy_ok = val_energy[val_ok]
        dt_before = val_time[val_ok] - ch_time[val_ok]
        dt_wf = val_time[val_ok] - ch_time_cal_wf_orig[val_ok]
        dt_after = val_time[val_ok] - ch_time_cal_orig[val_ok]

        def mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        _save_validation_trend_plot(
            prefix,
            trig_val,
            [delta_before, delta_wf, delta_after],
            val_ch,
            ch,
            "[Wiener] Validation trend",
        )
        _save_validation_tmcp_hist(
            prefix,
            [delta_before, delta_wf, delta_after],
            val_ch,
            ch,
            nbins,
            "[Wiener] Validation projection",
        )
        _save_inverse_energy_scatter(
            prefix,
            val_energy_ok,
            [dt_before, dt_wf, dt_after],
            val_ch,
            ch,
            "[Wiener] Validation inverse-energy",
        )

        clean_before = mad_clip(delta_before)
        clean_wf = mad_clip(delta_wf)
        clean_after = mad_clip(delta_after)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − [mcp_peak−trigger] (ps)"
        nb = min(nbins, max(30, len(clean_before) // 15))

        res_b = _gauss_fit_hist(ax1, clean_before, nb, "tab:blue", val_xlabel,
                                "Before cal", len(delta_before))
        res_w = _gauss_fit_hist(ax2, clean_wf, nb, "tab:orange",
                                f"[ch{val_ch}−ch{ch}_WF] − [mcp_peak−trigger] (ps)",
                                "Wiener only", len(delta_wf))
        res_a = _gauss_fit_hist(ax3, clean_after, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_WF+SG] − [mcp_peak−trigger] (ps)",
                                "Wiener + SavGol", len(delta_after))

        fig.suptitle(f"[Wiener] Validation: [ch{val_ch} − ch{ch}] − [mcp_peak − trigger]",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(clean_before)
        sig_w = res_w[1] if res_w else np.std(clean_wf)
        sig_a = res_a[1] if res_a else np.std(clean_after)
        log(f"Before cal: σ={sig_b:.2f} ps")
        log(f"Wiener only: σ={sig_w:.2f} ps")
        log(f"Wiener + SavGol: σ={sig_a:.2f} ps")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  [Wiener] ch{ch} vs MCP trigger_time  —  Summary")
    print("=" * 60)
    print(f"  Wiener window        : {wf_win}")
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
                print(f"      Linear fit slope        : {slope_s:.6f}")
            if sig_s is not None:
                print(f"      Pre-cal resid σ         : {sig_s:.2f} ps")
            if sig_c is not None:
                print(f"      Post-Wiener-cal resid σ : {sig_c:.2f} ps")
    print("-" * 60)
    if sig_wrap is not None:
        print(f"  Wrapped Δt σ           : {sig_wrap:.2f} ps")
    print("=" * 60)


if __name__ == "__main__":
    main()
