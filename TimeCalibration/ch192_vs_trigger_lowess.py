#!/usr/bin/env python3
"""
Side-branch calibration test: 

 residual correction variant.

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

import uproot
import awkward as ak

# ---- Reuse from the main workflow & original script ----
from bar_helpers import find_data_tree, gauss, log
from bar_processing import _mcp_internal_dt_selector
from bar_plotting import plot_t_diff

from ch192_vs_trigger import (
    build_mcp_map,
    detect_segments,
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
    p.add_argument("--out-prefix", default="ch192_vs_trig_lowess",
                   help="Prefix for output plots (default: ch192_vs_trig_lowess)")
    p.add_argument("--nbins", type=int, default=120)
    p.add_argument("--val-channels", type=int, nargs="+", default=[137, 150],
                   help="Validation channels to process (default: 137 150)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--lowess-frac", type=float, default=0.15,
                   help="LOWESS neighborhood fraction (default: 0.15)")
    p.add_argument("--lowess-it", type=int, default=3,
                   help="LOWESS robust iterations (default: 3)")
    p.add_argument("--lowess-delta", type=float, default=0.0,
                   help="LOWESS delta optimization parameter (default: 0.0)")
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

def _per_file_analysis_lowess(data_dict, prefix, ch, val_channels, nbins,
                              lowess_frac=0.15, lowess_it=3, lowess_delta=0.0):
    ch_time = data_dict["ch_time"]
    mcp_trig = data_dict["mcp_trig_time"]
    mcp_peak = data_dict["mcp_peak_time"]
    val_bar_ok = data_dict["val_bar_ok"]

    # Add pseudo-channel 'bar' if 137 and 150 are present
    v_channels = list(val_channels)
    if 137 in val_channels and 150 in val_channels:
        data_dict["val_time_bar"] = (data_dict["val_time_137"] + data_dict["val_time_150"]) / 2.0
        data_dict["val_energy_bar"] = (data_dict["val_energy_137"] + data_dict["val_energy_150"]) / 2.0
        if "bar" not in v_channels:
            v_channels.append("bar")

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
            lowess_clean_sorted = _lowess_smooth(
                y_clean_sorted,
                resid_clean_sorted,
                frac=lowess_frac,
                it=lowess_it,
                delta=lowess_delta,
            )
            interp_func = interp1d(
                y_clean_sorted, lowess_clean_sorted,
                kind="linear", fill_value="extrapolate"
            )
            lowess_resid = interp_func(y_s)
            lowess_available = True
            log(f"Segment {s+1}: LOWESS applied on {n_clean}/{len(x_s)} clean events "
                f"(frac={lowess_frac:.3f}, it={lowess_it})")
        else:
            log(f"Segment {s+1}: too few clean events ({n_clean}) for LOWESS, skipping correction")

        sort_order_all = np.argsort(y_s)
        y_sorted = y_s[sort_order_all]
        lowess_sorted_all = lowess_resid[sort_order_all]
        predicted = x0 + m * (y_s - y0) + b

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_s[~clean_mask], resid[~clean_mask], s=2, alpha=0.3,
                   color="lightcoral", label=f"Outliers ({(~clean_mask).sum()})")
        ax.scatter(y_s[clean_mask], resid[clean_mask], s=2, alpha=0.3,
                   color="steelblue", label=f"Clean ({clean_mask.sum()})")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        if lowess_available:
            ax.plot(y_sorted, lowess_sorted_all, color="red", linewidth=2,
                    label=f"LOWESS (frac={lowess_frac:.3f})")
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
        if lowess_available:
            sort_fit = np.argsort(y_fit)
            ax.plot(y_fit[sort_fit], lowess_resid[clean_mask][sort_fit],
                    color="red", linewidth=2, label="LOWESS drift")
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
            ch_lowess_seg, ch_corr_seg = _build_corrected_stages(
                x_s, lowess_resid, y_s, clean_mask
            )
            expected_ch192 = x0 + m * (y_s - y0) + b + lowess_resid
        else:
            ch_lowess_seg, ch_corr_seg = x_s.copy(), _force_unit_slope(x_s.copy(), y_s, clean_mask)
            expected_ch192 = x0 + m * (y_s - y0) + b

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
    for s, mask_val in enumerate(seg_masks):
        ch_time_cal_lowess_orig[mask_val] += seg_offsets[s]
        ch_time_cal_orig[mask_val] += seg_offsets[s]
        ch_time_expected_orig[mask_val] += seg_offsets[s]

    for val_ch in v_channels:
        val_time = data_dict[f"val_time_{val_ch}"]
        val_energy = data_dict[f"val_energy_{val_ch}"]
        val_ok = (np.isfinite(val_time) & np.isfinite(mcp_peak)
                  & np.isfinite(ch_time) & np.isfinite(mcp_trig)
                  & val_bar_ok)
        if val_ok.sum() >= 10:
            delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
            delta_expected = (val_time[val_ok] - ch_time_expected_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
            val_energy_ok = val_energy[val_ok]
            dt_before = val_time[val_ok] - ch_time[val_ok]
            dt_expected = val_time[val_ok] - ch_time_expected_orig[val_ok]

            def _mad_clip(arr):
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
                return arr[np.abs(arr - med) < cut]

            labels = ["Before cal", "Expected ch192 Time"]
            _save_inverse_energy_scatter(
                f"{prefix}_ch{val_ch}", val_energy_ok, [dt_before, dt_expected], val_ch, ch,
                f"[LOWESS] Validation inverse-energy (ch{val_ch})", labels
            )

            delta_before_walk = _apply_walk_correction(delta_before, val_energy_ok, f"{prefix}_ch{val_ch}_walk_before", f"Walk Fit: Before cal (ch{val_ch})")
            delta_expected_walk = _apply_walk_correction(delta_expected, val_energy_ok, f"{prefix}_ch{val_ch}_walk_expected", f"Walk Fit: Expected ch192 (ch{val_ch})")

            cb = _mad_clip(delta_before_walk)
            ce = _mad_clip(delta_expected_walk)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
            val_xlabel = f"[ch{val_ch}−ch{ch}] − MCP_walk (ps)"
            nb = _validation_fit_nbins()
            res_b = _gauss_fit_hist(ax1, cb, nb, "tab:blue", val_xlabel,
                                    "Before cal (walk corr)", len(delta_before_walk))
            res_e = _gauss_fit_hist(ax3, ce, nb, "tab:green",
                                    f"[ch{val_ch}−ch{ch}_expected] − MCP_walk (ps)",
                                    "Expected ch192 (walk corr)", len(delta_expected_walk))
            fig.suptitle(f"[LOWESS] Validation (Walk Corrected): ch{val_ch} vs ch{ch}",
                         fontsize=13)
            fig.tight_layout()
            fig.savefig(f"{prefix}_ch{val_ch}_validation.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_ch{val_ch}_validation.png")
            sig_b = res_b[1] if res_b else np.std(cb)
            sig_e = res_e[1] if res_e else np.std(ce)
            log(f"  Per-file validation (ch{val_ch}): before σ={sig_b:.2f}, expected σ={sig_e:.2f} ps")

    return {
        "ch_time_aligned": ch_time_aligned,
        "ch_time_cal_lowess_orig": ch_time_cal_lowess_orig,
        "ch_time_cal_orig": ch_time_cal_orig,
        "ch_time_expected_orig": ch_time_expected_orig,
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
        "val_channels": args.val_channels,
        "max_entries": args.max_entries,
        "step_size": args.step_size,
    }

    data_keys = ["ch_time", "mcp_trig_time", "mcp_peak_time", "event_idx",
                 "val_bar_ok"] + [f"val_time_{vc}" for vc in args.val_channels] + [f"val_energy_{vc}" for vc in args.val_channels]
    per_file_data = []
    nw = min(args.nworkers, len(paths))
    if nw > 1:
        log(f"Using {nw} parallel workers")
        with ProcessPoolExecutor(max_workers=nw) as pool:
            futures = {pool.submit(_process_one_file_multi, p, cfg): p for p in paths}
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
            per_file_data.append((path, extract_ch_times_multi(f, cfg, mcp_map)))

    if not per_file_data:
        print("No valid files processed.")
        sys.exit(0)

    prefix = args.out_prefix
    nbins = args.nbins
    ch = args.channel
    prefix_dir = os.path.dirname(prefix) or "."
    prefix_base = os.path.basename(prefix)

    combined = {k: [] for k in data_keys + ["ch_time_cal_lowess_orig",
                                            "ch_time_cal_orig", "ch_time_aligned", "ch_time_expected_orig"]}
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
        data["val_bar_ok"] = data["val_bar_ok"].astype(bool)
        n_f = len(fch)
        log(f"\n{'='*60}")
        log(f"  [LOWESS] File: {basename}  ({n_f} events)")
        log(f"  Output dir: {file_dir}/")
        log(f"{'='*60}")
        if n_f < 2:
            log("  Skipping (not enough events)")
            continue

        result = _per_file_analysis_lowess(
            data, file_prefix, ch, args.val_channels, nbins,
            lowess_frac=args.lowess_frac, lowess_it=args.lowess_it, lowess_delta=args.lowess_delta
        )
        for k in data_keys:
            combined[k].append(data[k])
        combined["ch_time_cal_lowess_orig"].append(result["ch_time_cal_lowess_orig"])
        combined["ch_time_cal_orig"].append(result["ch_time_cal_orig"])
        combined["ch_time_expected_orig"].append(result["ch_time_expected_orig"])
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
    mcp_trig = combined["mcp_trig_time"]
    mcp_peak = combined["mcp_peak_time"]
    val_bar_ok = combined["val_bar_ok"].astype(bool)

    v_channels = list(args.val_channels)
    if 137 in v_channels and 150 in v_channels:
        combined["val_time_bar"] = (combined["val_time_137"] + combined["val_time_150"]) / 2.0
        combined["val_energy_bar"] = (combined["val_energy_137"] + combined["val_energy_150"]) / 2.0
        if "bar" not in v_channels:
            v_channels.append("bar")


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

    for val_ch in v_channels:
        val_time = combined[f"val_time_{val_ch}"]
        val_energy = combined[f"val_energy_{val_ch}"]
        
        log(f"\n=== [LOWESS] Global Validation: ch{val_ch} vs ch{ch} ===")
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
                fig.savefig(f"{prefix}_ch{val_ch}_val_energy.png", dpi=150)
                plt.close(fig)
                log(f"Saved: {prefix}_ch{val_ch}_val_energy.png")

            delta_before = (val_time[val_ok] - ch_time[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
            delta_expected = (val_time[val_ok] - ch_time_expected_orig[val_ok]) - (mcp_peak[val_ok] - mcp_trig[val_ok])
            val_energy_ok = val_energy[val_ok]
            dt_before = val_time[val_ok] - ch_time[val_ok]
            dt_expected = val_time[val_ok] - ch_time_expected_orig[val_ok]
            labels = ["Before cal", "Expected ch192 Time"]

            def mad_clip(arr):
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                cut = 3 * 1.4826 * mad if mad > 0 else np.std(arr) * 3
                return arr[np.abs(arr - med) < cut]

            _save_inverse_energy_scatter(
                f"{prefix}_ch{val_ch}", val_energy_ok, [dt_before, dt_expected], val_ch, ch,
                f"[LOWESS] Validation inverse-energy (ch{val_ch})", labels
            )

            delta_before_walk = _apply_walk_correction(delta_before, val_energy_ok, f"{prefix}_ch{val_ch}_walk_before", f"Walk Fit: Before cal (ch{val_ch})")
            delta_expected_walk = _apply_walk_correction(delta_expected, val_energy_ok, f"{prefix}_ch{val_ch}_walk_expected", f"Walk Fit: Expected ch192 (ch{val_ch})")

            clean_before = mad_clip(delta_before_walk)
            clean_expected = mad_clip(delta_expected_walk)
            fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5))
            val_xlabel = f"[ch{val_ch}−ch{ch}] − MCP_walk (ps)"
            nb = _validation_fit_nbins()
            res_b = _gauss_fit_hist(ax1, clean_before, nb, "tab:blue", val_xlabel,
                                    "Before cal (walk corr)", len(delta_before_walk))
            res_e = _gauss_fit_hist(ax3, clean_expected, nb, "tab:green",
                                    f"[ch{val_ch}−ch{ch}_expected] − MCP_walk (ps)",
                                    "Expected ch192 (walk corr)", len(delta_expected_walk))
            fig.suptitle(f"[LOWESS] Global Validation (Walk Corrected): ch{val_ch} vs ch{ch}",
                         fontsize=13)
            fig.tight_layout()
            fig.savefig(f"{prefix}_ch{val_ch}_validation.png", dpi=150)
            plt.close(fig)
            log(f"Saved: {prefix}_ch{val_ch}_validation.png")
            sig_b = res_b[1] if res_b else np.std(clean_before)
            sig_e = res_e[1] if res_e else np.std(clean_expected)
            log(f"Before cal (walk corr) [ch{val_ch}]: σ={sig_b:.2f} ps")
            log(f"Expected ch192 (walk corr) [ch{val_ch}]: σ={sig_e:.2f} ps")

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



def extract_ch_times_multi(root_file, cfg, mcp_map):
    from bar_helpers import find_data_tree, log
    import awkward as ak
    import math
    import numpy as np
    
    tree_name = find_data_tree(root_file)
    if tree_name is None:
        log("No data tree found")
        return None

    tree = root_file[tree_name]
    ch_target = cfg["channel"]
    val_channels = cfg.get("val_channels", [137, 150])
    branches = [cfg["branch_channel"], cfg["branch_time"]]
    branch_energy = cfg.get("branch_energy", "energy")
    has_energy = branch_energy in tree.keys()
    if has_energy:
        branches.append(branch_energy)

    max_entries = cfg.get("max_entries") or tree.num_entries
    step = cfg["step_size"]

    out = {
        "ch_time": [], "mcp_trig_time": [], "mcp_peak_time": [], 
        "event_idx": [], "val_bar_ok": []
    }
    for vc in val_channels:
        out[f"val_time_{vc}"] = []
        out[f"val_energy_{vc}"] = []

    for start in range(0, max_entries, step):
        stop = min(start + step, max_entries)
        arrays = tree.arrays(branches, library="ak", entry_start=start, entry_stop=stop)
        ch_arr = arrays[cfg["branch_channel"]]
        t_arr = arrays[cfg["branch_time"]]
        
        n_evt = len(ch_arr)
        global_idx = np.arange(start, start + n_evt, dtype=np.int64)

        for local_i in range(n_evt):
            evt_idx = int(global_idx[local_i])
            if evt_idx not in mcp_map:
                continue

            ch_list = ak.to_numpy(ch_arr[local_i])
            positions = np.where(ch_list == ch_target)[0]
            if len(positions) == 0:
                continue
            pos = positions[0]

            t_list = ak.to_numpy(t_arr[local_i])
            if pos >= len(t_list):
                continue
            ch_t = float(t_list[pos])
            if not math.isfinite(ch_t):
                continue

            tt, pt = mcp_map[evt_idx]
            out["ch_time"].append(ch_t)
            out["mcp_trig_time"].append(tt)
            out["mcp_peak_time"].append(pt)
            out["event_idx"].append(evt_idx)

            required_channels = {137, 150, 234, 243, ch_target}
            ch_set = set(ch_list.tolist())
            out["val_bar_ok"].append(ch_set == required_channels)
            
            e_arr = ak.to_numpy(arrays[branch_energy][local_i]) if has_energy else []

            for vc in val_channels:
                val_positions = np.where(ch_list == vc)[0]
                if len(val_positions) > 0:
                    val_pos = val_positions[0]
                    val_t = float(t_list[val_pos]) if val_pos < len(t_list) else float('nan')
                    val_e = float(e_arr[val_pos]) if (has_energy and val_pos < len(e_arr)) else float('nan')
                else:
                    val_t = float('nan')
                    val_e = float('nan')
                out[f"val_time_{vc}"].append(val_t)
                out[f"val_energy_{vc}"].append(val_e)

    return {k: np.array(v, dtype=float) for k, v in out.items() if isinstance(v, list)}

def _process_one_file_multi(path, cfg):
    import uproot
    f = uproot.open(path)
    from ch192_vs_trigger import build_mcp_map
    mcp_map = build_mcp_map(f, cfg)
    if not mcp_map:
        return None
    return extract_ch_times_multi(f, cfg, mcp_map)

if __name__ == "__main__":
    main()
