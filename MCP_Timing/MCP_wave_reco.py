#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
MCP Waveform Reconstruction

This script performs peak reconstruction on a single channel waveform CSV,
identifying and fitting peaks using Gaussian models with parabolic refinement
for sub-sample precision.

Key Features:
- Gaussian peak fitting with parabolic refinement for sub-sample precision
- Amplitude threshold filtering
- Automatic CSV file detection or manual file specification
- Diagnostic plots for waveforms, fits, and peak distributions

Input CSV Format:
    Expects waveform data in *_data.csv files with columns:
    - Segment: Segment identifier (integer)
    - Time_s: Sample timestamp in seconds
    - Voltage_V: Signal amplitude in volts

Output:
    - peaks_<basename>.csv : Results with peak times (relative to trigger), amplitudes, and metadata per segment
    - Multiple diagnostic plots: waveforms, fits, histograms per segment

Usage Examples:
    # Auto-detect *_data.csv in directory
    python MCP_wave_reco.py --dir ./trc_out --out-dir ./results
    
    # Specify file explicitly
    python MCP_wave_reco.py --csv raw_C1_0004237_0000001_6347_data.csv --out-dir ./out
    
    # With amplitude threshold and limited plotting
    python MCP_wave_reco.py --csv raw_C1_0004237_0000001_6347_data.csv --min-amp 0.01 --plot-first 3

Author: Licheng Zhang (licheng.zhang@cern.ch)
Date: 2026-01
"""
import argparse, os, glob, re
import numpy as np
import pandas as pd

# Force non-interactive backend so figures are saved, not shown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ============================================================================
# Helper Functions
# ============================================================================

def gaussian(x, amp, mu, sigma, baseline):
    """
    Gaussian function for peak fitting.
    
    Parameters:
        x: Independent variable (time in ns)
        amp: Peak amplitude (signal height above baseline)
        mu: Peak center position (mean)
        sigma: Peak width (standard deviation)
        baseline: Constant baseline offset
    
    Returns:
        Gaussian function value: baseline + amp * exp(-0.5 * ((x-mu)/sigma)^2)
    
    Note:
        Uses np.maximum(sigma, 1e-15) to prevent division by zero
    """
    return baseline + amp * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-15))**2)

def parabolic_refine(x, y, i):
    """
    Refine peak position using parabolic interpolation around maximum.
    
    Fits a parabola through three points (i-1, i, i+1) to estimate
    the true peak position with sub-sample precision.
    
    Parameters:
        x: Array of x-coordinates (e.g., time)
        y: Array of y-values (e.g., amplitude)
        i: Index of approximate peak maximum
    
    Returns:
        Refined x-position of peak center (float)
    
    Note:
        If i is at array edge or parabola is degenerate, returns x[i]
    """
    # Edge cases: can't interpolate at boundaries
    if i <= 0 or i >= len(y)-1:
        return x[i]
    
    # Three-point parabolic fit
    y0, y1, y2 = y[i-1], y[i], y[i+1]
    denom = (y0 - 2*y1 + y2)
    
    # Check for degenerate parabola (flat or numerical issue)
    if abs(denom) < 1e-20:
        return x[i]
    
    # Calculate fractional offset from peak position
    delta = 0.5*(y0 - y2)/denom
    dt = (x[i+1] - x[i]) if i+1 < len(x) else (x[i] - x[i-1])
    return x[i] + delta*dt

def fit_largest_peak(t_ns, a, plot=False, title="", save_path=None):
    """
    Fit the largest peak in a waveform using a Gaussian model.
    
    Algorithm:
        1. Find all peaks (both positive and negative) using scipy.signal.find_peaks
        2. Select the peak with largest absolute amplitude
        3. Estimate initial parameters (baseline, amplitude, width)
        4. Define fit window around peak (±2.5 × estimated width)
        5. Perform Gaussian fit using scipy.optimize.curve_fit
        6. Optionally save diagnostic plots
    
    Parameters:
        t_ns: Array of time values in nanoseconds (can be relative or absolute)
        a: Array of amplitude values
        plot: If True, generate diagnostic plots showing fit
        title: Plot title string
        save_path: Path to save plot (if plot=True)
    
    Returns:
        Dictionary with keys:
            - peak_time_ns: Fitted peak center position (same units as t_ns)
            - peak_amp: Fitted amplitude (above baseline)
            - peak_sigma_ns: Fitted Gaussian width (sigma)
            - baseline: Fitted baseline level
            - fit_success: Boolean indicating if fit converged
    
    Peak Finding:
        Uses prominence threshold of 0.3 × amplitude range to find significant peaks.
        Searches for both positive and negative peaks (handles inverted signals).
    
    Fit Window:
        Extends ±2.5 × estimated_width around peak to capture full shape.
        Minimum window size is 20 samples.
    
    Note:
        Returns all NaN values if:
        - Too few samples (< 10)
        - Non-finite values present
        - Zero time range
        - No peaks found
        - Fit fails to converge
    """
    out = dict(peak_time_ns=np.nan, peak_amp=np.nan, peak_sigma_ns=np.nan,
               baseline=np.nan, fit_success=False)

    # Validation: need sufficient data and valid values
    if len(t_ns) < 10 or len(a) < 10: return out
    if not (np.all(np.isfinite(t_ns)) and np.all(np.isfinite(a))): return out
    if (t_ns[-1] - t_ns[0]) <= 0: return out  # Time must be increasing

    # Calculate prominence threshold for peak detection
    # Use 5% of amplitude range to find significant peaks, minimum 1e-3
    amp_range = np.ptp(a)  # Peak-to-peak amplitude
    if amp_range == 0: return out  # Flat signal
    prominence = max(0.05 * amp_range, 1e-3)
    # Find peaks in both polarities (positive and negative signals)
    idx_pos, _ = find_peaks(a, prominence=prominence)
    idx_neg, _ = find_peaks(-a, prominence=prominence)
    
    # Combine all candidates with their absolute amplitudes
    cand = [(i, abs(a[i])) for i in idx_pos] + [(i, abs(a[i])) for i in idx_neg]
    
    # If no peaks found, optionally plot and return
    if not cand:
        if plot and save_path:
            fig, ax = plt.subplots(1,1,figsize=(10,4))
            ax.plot(t_ns, a)
            ax.set_title(title + " (no peaks found)")
            ax.set_xlabel("Time (ns)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)
        return out
    # Select largest peak by absolute amplitude
    cand.sort(key=lambda x: x[1], reverse=True)
    i0 = cand[0][0]  # Index of largest peak

    # Estimate baseline from first few samples
    baseline_est = float(np.median(a[:min(10, len(a))]))
    
    # Estimate peak width by finding half-maximum points
    half = baseline_est + 0.5*(a[i0] - baseline_est)
    L = i0  # Left edge of peak
    while L > 0 and abs(a[L] - half) > 0.6*abs(a[i0]-baseline_est): L -= 1
    R = i0  # Right edge of peak
    while R < len(a)-1 and abs(a[R] - half) > 0.6*abs(a[i0]-baseline_est): R += 1
    
    # Define fit window: ±2.5 × estimated width, minimum 20 samples
    est_w = max(R - L, 10)
    halfwin = int(max(2.5*est_w, 10))
    s = max(0, i0 - halfwin); e = min(len(a), i0 + halfwin)
    t_win = t_ns[s:e]; y = a[s:e]
    if len(t_win) < 8: return out  # Need minimum samples for fit

    # Initial parameter estimates for Gaussian fit
    mu0 = parabolic_refine(t_ns, a, i0)  # Refined peak position
    base0 = float(np.median([y[0], y[-1]]))  # Baseline from window edges
    amp0 = float(a[i0] - base0)  # Peak amplitude above baseline
    sig0 = max( (t_win[-1]-t_win[0])/10.0, 1e-3 )  # Initial width estimate

    # Shift coordinates to center peak at zero (improves numerical stability)
    xrel = t_win - mu0
    p0 = [amp0, 0.0, sig0, base0]  # [amplitude, mu_relative, sigma, baseline]
    
    # Bounds: amplitude unbounded, mu within window, sigma > 0, baseline unbounded
    bounds = ([-np.inf, xrel[0], 0.0, -np.inf],
              [ np.inf, xrel[-1], (t_win[-1]-t_win[0])*2.0, np.inf])
    
    # Perform nonlinear least-squares fit
    try:
        popt, _ = curve_fit(gaussian, xrel, y, p0=p0, bounds=bounds, maxfev=5000)
        amp, mu_rel, sig, base = [float(v) for v in popt]
        mu_abs = mu0 + mu_rel
        out.update(peak_time_ns=mu_abs, peak_amp=amp, peak_sigma_ns=sig,
                   baseline=base, fit_success=True)

        if plot and save_path:
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,7))
            ax1.plot(t_ns, a, label="waveform")
            ax1.axvline(mu_abs, linestyle='--', label=f"fit μ={mu_abs:.3f} ns")
            ax1.axvspan(t_win[0], t_win[-1], color='yellow', alpha=0.15, label="fit window")
            ax1.legend(); ax1.set_title(title); ax1.set_xlabel("Time (ns)"); ax1.set_ylabel("Amp"); ax1.grid(True, alpha=0.3)
            xx = np.linspace(t_win[0], t_win[-1], 400)
            ax2.plot(t_win, y, 'o', ms=3, label="data")
            ax2.plot(xx, gaussian(xx - mu0, *popt), '-', label="Gaussian fit")
            ax2.axhline(base, linestyle=':', label=f"baseline={base:.3g}")
            ax2.axvline(mu_abs, linestyle='--', label="μ")
            ax2.set_xlabel("Time (ns)"); ax2.set_ylabel("Amp"); ax2.grid(True, alpha=0.3); ax2.legend()
            plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)
    except Exception:
        pass

    return out

def load_wave_csv(path):
    """
    Expects waveform CSV with columns:
      Segment, Time_s, Voltage_V
    Also loads corresponding _meta.csv for trigger_time and trigger_offset per segment.
    Returns: (dict segment -> (t_ns ndarray, amp ndarray), dict segment -> metadata dict)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if 'segment' not in cols or 'time_s' not in cols or 'voltage_v' not in cols:
        available = ", ".join(df.columns)
        raise ValueError(
            f"{path}: expected columns Segment, Time_s, Voltage_V. "
            f"Available: {available}"
        )

    # Load meta.csv
    meta_path = path.replace('_data.csv', '_meta.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    # Read meta.csv manually since it has unquoted commas in values
    meta_dict = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if ',' not in line: continue
            # Split only on the first comma
            parts = line.split(',', 1)
            if len(parts) == 2:
                key, value = parts
                meta_dict[key.lower()] = value
    
    if 'trigger_time' not in meta_dict or 'trigger_offset' not in meta_dict:
        available_meta = list(meta_dict.keys())
        raise ValueError(
            f"{meta_path}: expected keys trigger_time, trigger_offset. "
            f"Available: {available_meta}"
        )
    
    # Parse trigger_time and trigger_offset (semicolon-separated strings)
    trigger_time_str = meta_dict['trigger_time']
    trigger_offset_str = meta_dict['trigger_offset']
    
    trigger_times = np.array([float(x) for x in trigger_time_str.strip('[]').split(';')])
    trigger_offsets = np.array([float(x) for x in trigger_offset_str.strip('[]').split(';')])
    
    if len(trigger_times) != len(trigger_offsets):
        raise ValueError(f"Mismatch in trigger_time ({len(trigger_times)}) and trigger_offset ({len(trigger_offsets)}) lengths")
    
    tns = 1e9 * df[cols['time_s']].astype(float)  # Convert seconds to nanoseconds (relative)
    df['_time_ns'] = tns
    df['_amp'] = df[cols['voltage_v']].astype(float)
    df['_seg'] = df[cols['segment']].astype(int)

    out = {}
    meta = {}
    for seg, g in df.groupby('_seg'):
        g = g.sort_values('_time_ns')  # Sort by time
        seg_idx = int(seg) - 1  # Assuming segments start from 1
        if seg_idx < 0 or seg_idx >= len(trigger_times):
            raise ValueError(f"Segment {seg} out of range for meta data (0-{len(trigger_times)-1})")
        
        # Absolute time: trigger_time + trigger_offset + relative Time_s, in ns
        abs_time_ns = 1e9 * (trigger_times[seg_idx] + trigger_offsets[seg_idx]) + g['_time_ns']
        out[int(seg)] = (abs_time_ns.to_numpy(), g['_amp'].to_numpy())
        meta[int(seg)] = {
            'trigger_time_s': trigger_times[seg_idx],
            'trigger_offset_s': trigger_offsets[seg_idx]
        }
    
    return out, meta

# ---------------- pipeline ----------------

def run_single(csv_path, plot_first=5, out_dir="./out", enable_plots=True,
               min_amp=None):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[i] Loading: {csv_path}")
    w, meta = load_wave_csv(csv_path)

    segments = sorted(w.keys())
    print(f"[i] Segments: {len(segments)}")

    rows = []
    base = os.path.splitext(os.path.basename(csv_path))[0]

    for k, seg in enumerate(segments):
        t_abs, a = w[seg]
        m = meta[seg]
        
        # Time relative to trigger
        trigger_ns = 1e9 * (m['trigger_time_s'] + m['trigger_offset_s'])
        t_rel = t_abs - trigger_ns

        # Plot waveform if enabled
        if enable_plots and k < plot_first:
            fig, ax = plt.subplots(1,1, figsize=(10,4))
            ax.plot(t_rel, a)
            ax.set_title(f"Segment {seg} – waveform")
            ax.set_xlabel("Time (ns, relative to trigger)"); ax.set_ylabel("Amplitude (V)"); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(out_dir, f"waveform_seg{seg}_{base}.png")
            plt.savefig(out_png, dpi=150); plt.close(fig)

        # Fit peak
        r = fit_largest_peak(t_rel, a, plot=(enable_plots and k < plot_first),
                             title=f"Segment {seg} – fit",
                             save_path=os.path.join(out_dir, f"fit_seg{seg}_{base}.png") if k < plot_first else None)

        rows.append({
            'segment': seg,
            'peak_time_ns': r['peak_time_ns'],
            'peak_amp': r['peak_amp'],
            'peak_sigma_ns': r['peak_sigma_ns'],
            'baseline': r['baseline'],
            'fit_success': r['fit_success'],
        })

    res = pd.DataFrame(rows)
    
    # Filter successful fits
    n_total = len(res)
    n_success = int(res['fit_success'].sum())
    print(f"[i] Fit statistics: total={n_total}, success={n_success}")
    
    res = res[res['fit_success']].copy()
    if len(res) == 0:
        raise RuntimeError("No segments with successful fits.")

    # Apply amplitude threshold if provided
    if min_amp is not None:
        res = res[res['peak_amp'].abs() >= float(min_amp)]

    if len(res) == 0:
        raise RuntimeError("No segments remaining after applying amplitude threshold.")

    # Output CSV
    out_csv = os.path.join(out_dir, f"peaks_{base}.csv")
    cols_out = ['segment', 'peak_time_ns', 'peak_amp', 'peak_sigma_ns', 'baseline']
    res[cols_out].to_csv(out_csv, index=False)
    print(f"[ok] Wrote: {out_csv}")

    if enable_plots:
        # Histogram of peak amplitudes
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        ax.hist(res['peak_amp'].to_numpy(), bins=50, edgecolor='black', alpha=0.8)
        mu = float(res['peak_amp'].mean()); sd = float(res['peak_amp'].std())
        ax.set_title(f"Peak Amplitudes\nMean={mu:.3f} V  RMS={sd:.3f} V")
        ax.set_xlabel("Amplitude (V)"); ax.set_ylabel("Counts"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_hist_amp = os.path.join(out_dir, f"hist_amp_{base}.png")
        plt.savefig(out_hist_amp, dpi=150); plt.close(fig)
        print(f"[ok] Wrote: {out_hist_amp}")

        # Histogram of peak times (relative)
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        ax.hist(res['peak_time_ns'].to_numpy(), bins=50, edgecolor='black', alpha=0.8)
        mu = float(res['peak_time_ns'].mean()); sd = float(res['peak_time_ns'].std())
        ax.set_title(f"Peak Times (Relative to Trigger)\nMean={mu:.3f} ns  RMS={sd:.3f} ns")
        ax.set_xlabel("Time (ns)"); ax.set_ylabel("Counts"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_hist_time = os.path.join(out_dir, f"hist_time_{base}.png")
        plt.savefig(out_hist_time, dpi=150); plt.close(fig)
        print(f"[ok] Wrote: {out_hist_time}")
    else:
        print("[i] Plots disabled (--no-plots)")

def main():
    ap = argparse.ArgumentParser(description="MCP Waveform Reconstruction from single channel CSV.")
    ap.add_argument("--dir", default="./out", help="Directory with CSV (default: ./out)")
    ap.add_argument("--csv", help="CSV file to process (overrides --dir search)")
    ap.add_argument("--pattern", default=r".*_data\.csv", help="Regex to find *_data.csv in --dir")
    ap.add_argument("--plot-first", type=int, default=5, help="Save plots for first N events (default: 5)")
    ap.add_argument("--out-dir", help="Output directory (default: same as --dir)")
    ap.add_argument("--no-plots", action="store_true", help="Disable all plot generation")
    ap.add_argument("--min-amp", type=float, default=None,
                    help="Minimum peak amplitude magnitude (|amp|) to include event (default: no threshold)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.dir
    os.makedirs(out_dir, exist_ok=True)
    enable_plots = not args.no_plots

    if args.csv:
        csv_path = args.csv
    else:
        # Find a single CSV in the directory
        files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files in {args.dir}")
        # Filter out non-data files
        import re
        wf = [f for f in files if re.search(args.pattern, os.path.basename(f))]
        if not wf:
            raise FileNotFoundError(f"No waveform CSVs in {args.dir} (avoiding coincidence files)")
        if len(wf) > 1:
            print(f"[warn] Multiple CSVs found: {wf}, using the first one: {wf[0]}")
        csv_path = wf[0]

    run_single(csv_path, plot_first=args.plot_first, out_dir=out_dir, enable_plots=enable_plots,
               min_amp=args.min_amp)

if __name__ == "__main__":
    main()
