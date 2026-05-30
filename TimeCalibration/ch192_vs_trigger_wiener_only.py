#!/usr/bin/env python3
"""
Side-branch calibration test: LOWESS residual correction variant.

This script follows the same matched-event and segment workflow as
ch192_vs_trigger.py, but replaces the residual correction stage with a
LOWESS curve on residual vs MCP trigger_time plus a final unit-slope correction.

Calibration approach per segment:
  1. Linear fit: ch192_aligned = m * trigger_time + b
  2. Compute residuals from linear fit
  3. Fit LOWESS to residual vs trigger_time
  4. ch192_lowess = ch192_aligned - lowess_drift
  5. Force final ch192 vs trigger slope to exactly 1

Example:
  python3 ch192_vs_trigger_lowess.py /path/to/4405_*_e.root --mcp-internal-dt-cut
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
from scipy.interpolate import interp1d
from scipy.signal import wiener

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
        description="Side-branch test: ch192 vs MCP trigger_time (LOWESS)"
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
    p.add_argument("--out-prefix", default="ch192_vs_trig_wiener_only",
                   help="Prefix for output plots (default: ch192_vs_trig_wiener_only)")
    p.add_argument("--nbins", type=int, default=120)
    p.add_argument("--val-channel", type=int, default=137)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--lowess-frac", type=float, default=0.15,
                   help="LOWESS neighborhood fraction (default: 0.15)")
    p.add_argument("--lowess-it", type=int, default=3,
                   help="LOWESS robust iterations (default: 3)")
    p.add_argument("--lowess-delta", type=float, default=0.0,
                   help="LOWESS delta optimization parameter (default: 0.0)")
    p.add_argument("--wiener-size", type=int, default=29,
                   help="Wiener filter window size (default: 29)")
    return p.parse_args()


def _tricube(u):
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    out[mask] = (1.0 - np.abs(u[mask]) ** 3) ** 3
    return out


def _bisquare(u):
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) < 1.0
    out[mask] = (1.0 - u[mask] ** 2) ** 2
    return out


def _weighted_linear_predict(x, y, w, x_eval):
    sw = float(np.sum(w))
    if sw <= 0.0:
        return float(np.nan)
    xw = float(np.sum(w * x) / sw)
    yw = float(np.sum(w * y) / sw)
    dx = x - xw
    denom = float(np.sum(w * dx * dx))
    if denom <= 0.0:
        return yw
    slope = float(np.sum(w * dx * (y - yw)) / denom)
    intercept = yw - slope * xw
    return slope * x_eval + intercept


def _lowess_smooth(x, y, frac=0.15, it=3, delta=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return y.copy()

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    fitted = np.zeros(n, dtype=float)
    robust = np.ones(n, dtype=float)
    n_neighbors = max(2, int(math.ceil(frac * n)))
    if n_neighbors > n:
        n_neighbors = n

    for iteration in range(max(1, it + 1)):
        for i in range(n):
            left = max(0, min(i - n_neighbors // 2, n - n_neighbors))
            right = left + n_neighbors
            x_win = x_sorted[left:right]
            y_win = y_sorted[left:right]
            dmax = max(
                abs(x_sorted[i] - x_win[0]),
                abs(x_win[-1] - x_sorted[i]),
            )
            if dmax <= 0.0:
                fitted[i] = y_sorted[i]
                continue
            w = _tricube(np.abs(x_win - x_sorted[i]) / dmax) * robust[left:right]
            pred = _weighted_linear_predict(x_win, y_win, w, x_sorted[i])
            fitted[i] = pred if math.isfinite(pred) else y_sorted[i]

        if iteration == it:
            break
        resid = y_sorted - fitted
        mad = float(np.median(np.abs(resid)))
        if mad <= 0.0:
            break
        robust = _bisquare(resid / (6.0 * mad))

    if delta > 0.0:
        pass

    out = np.empty(n, dtype=float)
    out[order] = fitted
    return out


def _series_label_pairs(labels):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    return [(label, colors[i % len(colors)]) for i, label in enumerate(labels)]


def _validation_fit_nbins():
    return 20


def _force_unit_slope(ch_time_corr, trigger_time, fit_mask):
    ch_time_corr = np.asarray(ch_time_corr, dtype=float)
    trigger_time = np.asarray(trigger_time, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    adjusted = ch_time_corr.copy()

    if fit_mask.sum() < 2:
        return adjusted

    x_fit = trigger_time[fit_mask]
    y_fit = adjusted[fit_mask]
    x_ref = float(np.mean(x_fit))
    m, _ = np.polyfit(x_fit - x_ref, y_fit, 1)
    adjusted -= (m - 1.0) * (trigger_time - x_ref)
    return adjusted


def _build_corrected_stages(ch_time_seg, lowess_resid, trigger_time, fit_mask):
    lowess_only = ch_time_seg - lowess_resid
    final = _force_unit_slope(lowess_only, trigger_time, fit_mask)
    return lowess_only, final


def _save_inverse_energy_scatter(prefix, energy, series, val_ch, ch, title_prefix, labels):
    label_pairs = _series_label_pairs(labels)
    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5), squeeze=False)
    axes = axes[0]

    for ax, (values, (label, color)) in zip(axes, zip(series, label_pairs)):
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




def _apply_walk_correction(delta_t, energy, prefix="", title=""):
    mask = np.isfinite(delta_t) & np.isfinite(energy) & (energy > 0)
    if mask.sum() < 10:
        return delta_t.copy()
    
    x = 1.0 / energy[mask]
    y = delta_t[mask]
    
    # Simple cut to remove extreme outliers in 1/E
    inv_e_mask = (x < 0.01) & (x > 0.0005)
    x_clean = x[inv_e_mask]
    y_clean = y[inv_e_mask]
    
    if len(x_clean) < 10:
        return delta_t.copy()

    # MAD cut for y to avoid fitting outliers
    y_med = np.median(y_clean)
    y_mad = np.median(np.abs(y_clean - y_med))
    if y_mad > 0:
        fit_mask = np.abs(y_clean - y_med) < 5 * 1.4826 * y_mad
        x_fit = x_clean[fit_mask]
        y_fit = y_clean[fit_mask]
    else:
        x_fit = x_clean
        y_fit = y_clean

    if len(x_fit) < 3:
        return delta_t.copy()

    # Fit quadratic
    p = np.polyfit(x_fit, y_fit, 2)
    
    # Plot to verify
    if prefix:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_fit, y_fit, s=3, alpha=0.3, color="steelblue", label=f"Data (N={len(x_fit)})")
        x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
        y_line = np.polyval(p, x_line)
        ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Fit: {p[0]:.2e}x² + {p[1]:.2e}x + {p[2]:.2e}")
        ax.set_xlabel("1 / Energy")
        ax.set_ylabel("Delta t (ps)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f"{prefix}_walk_fit.png", dpi=150)
        plt.close(fig)
    
    # Correct all valid events
    corrected_delta = delta_t.copy()
    valid_e = (energy > 0) & np.isfinite(energy) & np.isfinite(delta_t)
    corrected_delta[valid_e] = delta_t[valid_e] - np.polyval(p, 1.0 / energy[valid_e])
    
    # Center the corrected delta so its median is around 0
    clean_corr = corrected_delta[valid_e]
    if len(clean_corr) > 0:
        med_corr = np.median(clean_corr)
        corrected_delta[valid_e] -= med_corr
    
    return corrected_delta


# ──────────────────────────────────────────────────────────────────
# Per-file analysis with LOWESS calibration
# ──────────────────────────────────────────────────────────────────

def _per_file_analysis_lowess(ch_time, mcp_trig, mcp_peak, val_time, val_bar_ok,
                              val_energy, prefix, ch, val_ch, nbins,
                              lowess_frac=0.15, lowess_it=3, lowess_delta=0.0,
                              wiener_size=29):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fit_colors = ["red", "darkgreen", "darkblue", "purple", "brown"]

    seg_masks = detect_segments(mcp_trig, ch_time)
    n_seg = len(seg_masks)
    ch_time_aligned = ch_time.copy()
    seg_offsets = []

    for s, mask in enumerate(seg_masks):
        x_s = ch_time[mask]
        y_s = mcp_trig[mask]
        if len(x_s) > 0 and len(y_s) > 0:
            offset = float(x_s[0] - y_s[0])
            seg_offsets.append(offset)
            ch_time_aligned[mask] = x_s - offset
            log(f"Segment {s+1}: aligned ch{ch} by subtracting offset = {offset:.2f} ps")
        else:
            seg_offsets.append(0.0)

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
            plt.plot(t_line, ch_line, color=fit_colors[s % len(fit_colors)], linewidth=2,
                     label=f"Fit {s+1}: slope={m:.6f}")
        else:
            seg_slopes.append(None)
    plt.xlabel("MCP trigger_time (ps)")
    plt.ylabel(f"Channel {ch} time − aligned (ps)")
    plt.title(f"[LOWESS] MCP trigger_time vs Channel {ch} (aligned)  ({n_seg} segments)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter.png")

    fig, axes = plt.subplots(1, n_seg, figsize=(7 * n_seg, 5), squeeze=False)
    for s, mask in enumerate(seg_masks):
        ax = axes[0][s]
        x_s, y_s = ch_time_aligned[mask], mcp_trig[mask]
        ax.scatter(y_s, x_s, s=4, alpha=0.4, color=colors[s % len(colors)])
        if len(x_s) >= 2:
            x0, y0 = float(np.mean(x_s)), float(np.mean(y_s))
            m, b = np.polyfit(y_s - y0, x_s - x0, 1)
            t_line = np.linspace(y_s.min(), y_s.max(), 200)
            ch_line = m * (t_line - y0) + b + x0
            ax.plot(t_line, ch_line, color="red", linewidth=2, label=f"slope={m:.6f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"Channel {ch} time − aligned (ps)")
        ax.set_title(f"Segment {s+1}  ({len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments.png")

    ch_time_cal_lowess = ch_time_aligned.copy()
    ch_time_cal = ch_time_aligned.copy()
    ch_time_expected = ch_time_aligned.copy()
    ch_time_wiener = ch_time_aligned.copy()
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

        x0, y0 = float(np.mean(x_s)), float(np.mean(y_s))
        fit_idx = np.ones(len(x_s), dtype=bool)
        m, b = np.polyfit(y_s - y0, x_s - x0, 1)
        for _ in range(3):
            resid_all = (x_s - x0) - (m * (y_s - y0) + b)
            r_med = np.median(resid_all[fit_idx])
            r_mad = np.median(np.abs(resid_all[fit_idx] - r_med))
            if r_mad <= 0:
                break
            fit_idx = np.abs(resid_all - r_med) < 3 * 1.4826 * r_mad
            if fit_idx.sum() < 2:
                break
            m, b = np.polyfit(y_s[fit_idx] - y0, x_s[fit_idx] - x0, 1)

        resid = (x_s - x0) - (m * (y_s - y0) + b)
        resid -= np.mean(resid[fit_idx])
        all_resid.append(resid)
        log(f"Segment {s+1}: linear slope={m:.6f}, fit on {fit_idx.sum()}/{len(x_s)} events")

        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        clean = resid[np.abs(resid - med) < 5 * 1.4826 * mad] if mad > 0 else resid
        if len(clean) < 5:
            clean = resid

        sig = plot_t_diff(
            clean.tolist(), f"{prefix}_residuals_seg{s+1}.png",
            f"[LOWESS] Residuals (pre-cal): Seg {s+1}  (slope={m:.6f}, "
            f"N={len(clean)}/{len(resid)})",
            nbins=min(nbins, max(30, len(clean) // 15)), xlabel="Residual (ps)"
        )
        seg_sigmas.append(sig)
        if sig is not None:
            log(f"Segment {s+1} pre-cal residual sigma = {sig:.2f} ps")

        clean_mask = fit_idx.copy()
        n_clean = int(clean_mask.sum())
        lowess_resid = np.zeros_like(resid)
        lowess_available = False

        if n_clean >= 3:
            y_clean = y_s[clean_mask]
            resid_clean = resid[clean_mask]
            sort_order_clean = np.argsort(y_clean)
            y_clean_sorted = y_clean[sort_order_clean]
            resid_clean_sorted = resid_clean[sort_order_clean]
            
            actual_wsize = min(wiener_size, len(resid_clean_sorted))
            if actual_wsize < 3:
                resid_wiener_sorted = resid_clean_sorted
            else:
                resid_wiener_sorted = wiener(resid_clean_sorted, mysize=actual_wsize)
                
            # Interpolate Wiener residual for all events in segment
            interp_func_wf = interp1d(
                y_clean_sorted, resid_wiener_sorted,
                kind="linear", fill_value="extrapolate"
            )
            wiener_resid_all = interp_func_wf(y_s)
            
            # Disable LOWESS: just set drift to 0
            lowess_resid = np.zeros_like(y_s)
            lowess_available = True
            log(f"Segment {s+1}: Wiener applied on {n_clean}/{len(x_s)} clean events")
        else:
            log(f"Segment {s+1}: too few clean events ({n_clean}) for LOWESS, skipping correction")

        sort_order_all = np.argsort(y_s)
        y_sorted = y_s[sort_order_all]
        lowess_sorted_all = lowess_resid[sort_order_all]
        predicted = x0 + m * (y_s - y0) + b

        # [NEW] Plot 1: Raw residual with Wiener overlay
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_s[~clean_mask], resid[~clean_mask], s=2, alpha=0.3,
                   color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
        ax.scatter(y_s[clean_mask], resid[clean_mask], s=2, alpha=0.3,
                   color="steelblue", label=f"Clean ({clean_mask.sum()})")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        
        # Calculate actual_wsize for the label
        actual_wsize = min(wiener_size, len(resid[clean_mask])) if 'wiener_size' in locals() else 29
        if 'wiener_resid_all' in locals():
            ax.plot(y_s[sort_order_all], wiener_resid_all[sort_order_all], color="orange", linewidth=2,
                    label=f"Wiener (win={actual_wsize})")
        ax.legend(fontsize=8)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[Wiener] Seg {s+1}: residual vs trigger_time  (N={len(resid)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_vs_trig_wf_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_vs_trig_wf_seg{s+1}.png")

        # [NEW] Plot 2: Wiener residual with LOWESS overlay
        fig, ax = plt.subplots(figsize=(10, 5))
        if 'wiener_resid_all' in locals():
            ax.scatter(y_s, wiener_resid_all, s=2, alpha=0.4, color="orange", label="Wiener-filtered")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")

        ax.legend(fontsize=8)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Wiener Residual (ps)")
        ax.set_title(f"[LOWESS] Seg {s+1}: Wiener-filtered vs trigger_time  (N={len(resid)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_wf_vs_trig_lowess_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_wf_vs_trig_lowess_seg{s+1}.png")


        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_s[~clean_mask], resid[~clean_mask], s=2, alpha=0.3,
                   color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
        ax.scatter(y_s[clean_mask], resid[clean_mask], s=2, alpha=0.3,
                   color="steelblue", label=f"Clean ({clean_mask.sum()})")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")

        ax.legend(fontsize=8)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[LOWESS] Seg {s+1}: residual vs trigger_time  "
                     f"(N={len(resid)}, clean={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_vs_trig_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_vs_trig_seg{s+1}.png")

        resid_fit = resid[clean_mask]
        y_fit = y_s[clean_mask]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_fit, resid_fit, s=2, alpha=0.4, color="steelblue",
                   label="Original residual")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")

        ax.legend(fontsize=8)
        if len(resid_fit) > 0:
            r_lo, r_hi = float(resid_fit.min()), float(resid_fit.max())
            pad = 0.1 * (r_hi - r_lo) if r_hi > r_lo else 10.0
            ax.set_ylim(r_lo - pad, r_hi + pad)
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Residual (ps)")
        ax.set_title(f"[LOWESS] Seg {s+1}: clean-only events  (N={n_clean})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_fitonly_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_fitonly_seg{s+1}.png")

        if lowess_available:
            ch_time_wiener_seg = x0 + m * (y_s - y0) + b + wiener_resid_all
            ch_lowess_seg, ch_corr_seg = _build_corrected_stages(
                ch_time_wiener_seg, lowess_resid, y_s, clean_mask
            )
            expected_ch192 = x0 + m * (y_s - y0) + b + lowess_resid
            ch_time_wiener[mask] = ch_time_wiener_seg
        else:
            ch_time_wiener_seg = x0 + m * (y_s - y0) + b + resid
            ch_lowess_seg, ch_corr_seg = ch_time_wiener_seg.copy(), _force_unit_slope(ch_time_wiener_seg.copy(), y_s, clean_mask)
            expected_ch192 = x0 + m * (y_s - y0) + b
            ch_time_wiener[mask] = ch_time_wiener_seg

        ch_time_cal_lowess[mask] = ch_lowess_seg
        ch_time_cal[mask] = ch_corr_seg
        ch_time_expected[mask] = expected_ch192
        resid_cal = ch_corr_seg - y_s
        resid_cal -= np.mean(resid_cal[clean_mask])
        m_final, _ = np.polyfit(y_s[clean_mask], ch_corr_seg[clean_mask], 1)
        log(f"Segment {s+1}: final unit-slope correction applied, fitted slope={m_final:.6f}")
        all_resid_cal.append(resid_cal)

        cal_med = np.median(resid_cal)
        cal_mad = np.median(np.abs(resid_cal - cal_med))
        cal_cut = 3 * 1.4826 * cal_mad if cal_mad > 0 else np.std(resid_cal) * 3
        cal_clip = np.abs(resid_cal - cal_med) < cal_cut
        clean_cal = resid_cal[cal_clip]

        sig_cal = plot_t_diff(
            clean_cal.tolist(), f"{prefix}_residuals_cal_seg{s+1}.png",
            f"[LOWESS] Residuals (post-cal): Seg {s+1} "
            f"(±3MAD, N={len(clean_cal)}/{len(resid_cal)})",
            nbins=min(nbins, max(30, len(clean_cal) // 15)),
            xlabel="Calibrated residual (ps)"
        )
        seg_sigmas_cal.append(sig_cal)
        if sig_cal is not None:
            log(f"Segment {s+1} post-cal residual sigma = {sig_cal:.2f} ps")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_s, resid_cal, s=2, alpha=0.3, color="steelblue")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_xlabel("Trigger time (ps)")
        ax.set_ylabel("Calibrated residual (ps)")
        ax.set_title(f"[LOWESS] Seg {s+1}: calibrated residual vs trigger_time  "
                     f"(N={len(resid_cal)})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_vs_trig_cal_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_vs_trig_cal_seg{s+1}.png")

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
        ax.set_title(f"[LOWESS] Seg {s+1}: cal residual clean-only  "
                     f"(N={int(clean_mask.sum())})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_resid_cal_fitonly_seg{s+1}.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_resid_cal_fitonly_seg{s+1}.png")

    fig, axes = plt.subplots(1, n_seg, figsize=(7 * n_seg, 5), squeeze=False)
    for s, mask in enumerate(seg_masks):
        ax = axes[0][s]
        x_s, y_s = ch_time_cal[mask], mcp_trig[mask]
        ax.scatter(y_s, x_s, s=4, alpha=0.4, color=colors[s % len(colors)])
        if len(x_s) >= 2:
            x0t, y0t = float(np.mean(x_s)), float(np.mean(y_s))
            mt, bt = np.polyfit(y_s - y0t, x_s - x0t, 1)
            t_line = np.linspace(y_s.min(), y_s.max(), 200)
            ax.plot(t_line, mt * (t_line - y0t) + bt + x0t, color="red", linewidth=2,
                    label=f"slope={mt:.6f}")
            ax.legend(fontsize=8)
        ax.set_xlabel("MCP trigger_time (ps)")
        ax.set_ylabel(f"Channel {ch} final cal (ps)")
        ax.set_title(f"Segment {s+1} (LOWESS + unit-slope cal, {len(x_s)} events)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_scatter_segments_cal.png", dpi=150)
    plt.close()
    log(f"Saved: {prefix}_scatter_segments_cal.png")

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
        ax.set_title(f"[LOWESS] Pre-cal residuals: {n_seg} segments")
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
        ax.set_title(f"[LOWESS] Post-cal residuals: {n_seg} segments")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    ch_time_cal_lowess_orig = ch_time_cal_lowess.copy()
    ch_time_cal_orig = ch_time_cal.copy()
    ch_time_expected_orig = ch_time_expected.copy()
    ch_time_wiener_orig = ch_time_wiener.copy()
    for s, mask_val in enumerate(seg_masks):
        ch_time_cal_lowess_orig[mask_val] += seg_offsets[s]
        ch_time_cal_orig[mask_val] += seg_offsets[s]
        ch_time_expected_orig[mask_val] += seg_offsets[s]
        ch_time_wiener_orig[mask_val] += seg_offsets[s]

    val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
              & np.isfinite(ch_time) & np.isfinite(mcp_trig)
              & val_bar_ok)
    if val_ok.sum() >= 10:
        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_lowess = (val_time[val_ok] - ch_time_cal_lowess_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_expected = (val_time[val_ok] - ch_time_expected_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        val_energy_ok = val_energy[val_ok]
        dt_before = val_time[val_ok] - ch_time[val_ok]
        dt_lowess = val_time[val_ok] - ch_time_cal_lowess_orig[val_ok]
        dt_expected = val_time[val_ok] - ch_time_expected_orig[val_ok]

        def _mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        labels = ["Before cal", "LOWESS correction", "Expected ch192 Time"]
        _save_inverse_energy_scatter(
            prefix, val_energy_ok, [dt_before, dt_lowess, dt_expected], val_ch, ch,
            "[LOWESS] Validation inverse-energy", labels
        )

        delta_before_walk = _apply_walk_correction(delta_before, val_energy_ok, f"{prefix}_walk_before", f"Walk Fit: Before cal (ch{val_ch})")
        delta_lowess_walk = _apply_walk_correction(delta_lowess, val_energy_ok, f"{prefix}_walk_lowess", f"Walk Fit: LOWESS correction (ch{val_ch})")
        delta_expected_walk = _apply_walk_correction(delta_expected, val_energy_ok, f"{prefix}_walk_expected", f"Walk Fit: Expected ch192 (ch{val_ch})")

        cb = _mad_clip(delta_before_walk)
        cl = _mad_clip(delta_lowess_walk)
        ce = _mad_clip(delta_expected_walk)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − MCP_walk (ps)"
        nb = _validation_fit_nbins()
        res_b = _gauss_fit_hist(ax1, cb, nb, "tab:blue", val_xlabel,
                                "Before cal (walk corr)", len(delta_before_walk))
        res_l = _gauss_fit_hist(ax2, cl, nb, "tab:orange",
                                f"[ch{val_ch}−ch{ch}_lowess] − MCP_walk (ps)",
                                "LOWESS correction (walk corr)", len(delta_lowess_walk))
        res_e = _gauss_fit_hist(ax3, ce, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_expected] − MCP_walk (ps)",
                                "Expected ch192 (walk corr)", len(delta_expected_walk))
        fig.suptitle(f"[LOWESS] Validation (Walk Corrected): ch{val_ch} vs ch{ch}",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(cb)
        sig_l = res_l[1] if res_l else np.std(cl)
        sig_e = res_e[1] if res_e else np.std(ce)
        log(f"  Per-file validation: before σ={sig_b:.2f}, lowess σ={sig_l:.2f}, expected σ={sig_e:.2f} ps")

    return {
        "ch_time_aligned": ch_time_aligned,
        "ch_time_cal_lowess_orig": ch_time_cal_lowess_orig,
        "ch_time_cal_orig": ch_time_cal_orig,
        "ch_time_expected_orig": ch_time_expected_orig,
        "ch_time_wiener_orig": ch_time_wiener_orig,
        "seg_masks": seg_masks,
        "seg_offsets": seg_offsets,
        "seg_slopes": seg_slopes,
        "seg_sigmas": seg_sigmas,
        "seg_sigmas_cal": seg_sigmas_cal,
        "all_resid": all_resid,
        "all_resid_cal": all_resid_cal,
        "n_seg": n_seg,
    }


def main():
    args = parse_args()
    paths = []
    for pattern in args.file:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No valid input files found.")
        sys.exit(1)
    log(f"[LOWESS] Processing {len(paths)} file(s)")

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
                log("  Skipping (no valid MCP entries)")
                continue
            per_file_data.append((path, extract_ch_times(f, cfg, mcp_map)))

    if not per_file_data:
        print("No valid files processed.")
        sys.exit(0)

    prefix = args.out_prefix
    nbins = args.nbins
    ch = args.channel
    val_ch = args.val_channel
    prefix_dir = os.path.dirname(prefix) or "."
    prefix_base = os.path.basename(prefix)

    combined = {k: [] for k in data_keys + ["ch_time_cal_lowess_orig",
                                            "ch_time_cal_orig", "ch_time_aligned", "ch_time_expected_orig", "ch_time_wiener_orig"]}
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
        log(f"  [LOWESS] File: {basename}  ({n_f} events)")
        log(f"  Output dir: {file_dir}/")
        log(f"{'='*60}")
        if n_f < 2:
            log("  Skipping (not enough events)")
            continue

        result = _per_file_analysis_lowess(
            fch, fmcp, fmpk, fval, fbar, feng, file_prefix, ch, val_ch, nbins,
            lowess_frac=args.lowess_frac, lowess_it=args.lowess_it, lowess_delta=args.lowess_delta,
            wiener_size=args.wiener_size
        )
        for k in data_keys:
            combined[k].append(data[k])
        combined["ch_time_cal_lowess_orig"].append(result["ch_time_cal_lowess_orig"])
        combined["ch_time_cal_orig"].append(result["ch_time_cal_orig"])
        combined["ch_time_expected_orig"].append(result["ch_time_expected_orig"])
        combined["ch_time_wiener_orig"].append(result["ch_time_wiener_orig"])
        combined["ch_time_aligned"].append(result["ch_time_aligned"])
        all_resid_global.extend(result["all_resid"])
        all_resid_cal_global.extend(result["all_resid_cal"])
        all_file_summary.append((basename, result["n_seg"], n_f,
                                 result["seg_slopes"], result["seg_sigmas"],
                                 result["seg_sigmas_cal"]))

    for k in combined:
        combined[k] = np.concatenate(combined[k]) if combined[k] else np.array([])

    n = len(combined["ch_time"])
    log(f"\n[LOWESS] Total matched events across all files: {n}")
    if n < 2:
        print("Not enough matched events to produce combined plots.")
        sys.exit(0)

    ch_time = combined["ch_time"]
    ch_time_aligned = combined["ch_time_aligned"]
    ch_time_cal_lowess_orig = combined["ch_time_cal_lowess_orig"]
    ch_time_cal_orig = combined["ch_time_cal_orig"]
    ch_time_expected_orig = combined["ch_time_expected_orig"]
    ch_time_wiener_orig = combined["ch_time_wiener_orig"]
    mcp_trig = combined["mcp_trig_time"]
    mcp_peak = combined["mcp_peak_time"]
    val_time = combined["val_time"]
    val_bar_ok = combined["val_bar_ok"].astype(bool)
    val_energy = combined["val_energy"]

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
                    color=seg_colors[s % len(seg_colors)], label=f"Seg {s+1} ({len(clean_s)} evts)")
        ax.set_xlabel("Residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title(f"[LOWESS] Pre-cal residuals: all {len(all_resid_global)} segments combined")
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
                    color=seg_colors[si % len(seg_colors)], label=f"Seg {si+1} ({len(clean_ci)} evts)")
        ax.set_xlabel("Calibrated residual (ps)")
        ax.set_ylabel("Counts")
        ax.set_title("[LOWESS] Post-cal residuals: all segments combined")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_residuals_combined_cal.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_residuals_combined_cal.png")

    dt = ch_time_aligned - mcp_trig
    dt_wrapped = ((dt + 3125.0) % 6250.0) - 3125.0
    sig_wrap = plot_t_diff(
        dt_wrapped.tolist(), f"{prefix}_dt_wrapped.png",
        f"[LOWESS] Δt = ch{ch} − trigger_time (wrapped ±3125 ps)",
        nbins=nbins, xlabel=f"Wrapped Δt (ps)"
    )
    if sig_wrap is not None:
        log(f"Wrapped dt sigma = {sig_wrap:.2f} ps")

    log(f"\n=== [LOWESS] Validation: ch{val_ch} vs ch{ch} ===")
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
            ax.set_title(f"ch{val_ch} energy distribution (validation events, N={len(val_e_finite)})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{prefix}_val_energy.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_val_energy.png")

        delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        delta_wiener_corr = (val_time[val_ok] - ch_time_cal_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
        
        val_energy_ok = val_energy[val_ok]
        dt_before = val_time[val_ok] - ch_time[val_ok]
        dt_wiener_corr = val_time[val_ok] - ch_time_cal_orig[val_ok]
        
        labels = ["Before cal", "Wiener Only (Final)"]

        def mad_clip(arr):
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
            return arr[np.abs(arr - med) < cut]

        _save_inverse_energy_scatter(
            prefix, val_energy_ok, [dt_before, dt_wiener_corr], val_ch, ch,
            "[Wiener Only] Validation inverse-energy", labels
        )

        delta_before_walk = _apply_walk_correction(delta_before, val_energy_ok, f"{prefix}_walk_before", f"Walk Fit: Before cal (ch{val_ch})")
        delta_wiener_corr_walk = _apply_walk_correction(delta_wiener_corr, val_energy_ok, f"{prefix}_walk_wiener_final", f"Walk Fit: Wiener Only Final (ch{val_ch})")
        
        clean_before = mad_clip(delta_before_walk)
        clean_wiener_corr = mad_clip(delta_wiener_corr_walk)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        val_xlabel = f"[ch{val_ch}−ch{ch}] − MCP_walk (ps)"
        nb = _validation_fit_nbins()
        res_b = _gauss_fit_hist(ax1, clean_before, nb, "tab:blue", val_xlabel,
                                "Before cal", len(delta_before_walk))
        res_wc = _gauss_fit_hist(ax2, clean_wiener_corr, nb, "tab:green",
                                f"[ch{val_ch}−ch{ch}_wiener_cal] − MCP_walk (ps)",
                                "Wiener Only (Final)", len(delta_wiener_corr_walk))
        
        fig.suptitle(f"[Wiener Only] Validation (Walk Corrected): ch{val_ch} vs ch{ch}",
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(f"{prefix}_validation.png", dpi=150)
        plt.close(fig)
        log(f"Saved: {prefix}_validation.png")
        sig_b = res_b[1] if res_b else np.std(clean_before)
        sig_wc = res_wc[1] if res_wc else np.std(clean_wiener_corr)
        
        log(f"Before cal: σ={sig_b:.2f} ps")
        log(f"Wiener Only (Final): σ={sig_wc:.2f} ps")

    print("\n" + "=" * 60)
    print(f"  [LOWESS] ch{ch} vs MCP trigger_time  —  Summary")
    print("=" * 60)
    print(f"  LOWESS frac          : {args.lowess_frac}")
    print(f"  LOWESS it            : {args.lowess_it}")
    print(f"  LOWESS delta         : {args.lowess_delta}")
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
                print(f"      Linear fit slope     : {slope_s:.6f}")
            if sig_s is not None:
                print(f"      Pre-cal resid σ      : {sig_s:.2f} ps")
            if sig_c is not None:
                print(f"      Post-final resid σ   : {sig_c:.2f} ps")
    print("-" * 60)
    if sig_wrap is not None:
        print(f"  Wrapped Δt σ        : {sig_wrap:.2f} ps")
    print("=" * 60)


if __name__ == "__main__":
    main()
