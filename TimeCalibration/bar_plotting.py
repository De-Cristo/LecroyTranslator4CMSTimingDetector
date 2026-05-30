#!/usr/bin/env python3
"""
Plotting functions for bar-level timing analysis.

All visualisation routines extracted from bar_analysis.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import moyal

from bar_helpers import gauss


def _add_entry_box(ax, n_entries):
    """Draw entry-count annotation in the top-right corner."""
    ax.text(
        0.98, 0.96, f"Entries: {int(n_entries)}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.85),
    )


# ──────────────────────────────────────────────────────────────────
# 1-D  T_L − T_R  histogram  +  Gaussian fit
# ──────────────────────────────────────────────────────────────────

def plot_t_diff(vals, out_path, title, nbins=100, hist_range=None,
               xlabel="T_left − T_right  (ps)", color="teal"):
    """Plot 1-D histogram of T_L − T_R with Gaussian fit."""
    if not vals:
        print(f"No valid points for {title}.")
        return None

    arr = np.asarray(vals, dtype=float)
    plt.figure(figsize=(7, 5))
    counts, bins, _ = plt.hist(arr, bins=nbins, range=hist_range,
                                alpha=0.75,
                                color=color, edgecolor="white")

    # ---- Gaussian fit ----
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask]
    if x_fit.size >= 3:
        # Find the peak bin
        max_idx = np.argmax(y_fit)
        peak_x = x_fit[max_idx]
        a0 = float(y_fit[max_idx])
        
        # Determine an initial window around the peak
        # For phase differences and time differences, 500 ps is a safe wide initial search window
        init_window = 500.0
        mask_init = (x_fit >= peak_x - init_window) & (x_fit <= peak_x + init_window)
        
        if np.sum(mask_init) >= 3:
            mu0 = peak_x
            
            # Estimate sigma from the FWHM inside the window
            y_window = y_fit[mask_init]
            x_window = x_fit[mask_init]
            half_max = a0 / 2.0
            above_half = x_window[y_window >= half_max]
            if len(above_half) >= 2:
                sig0 = (above_half[-1] - above_half[0]) / 2.355  # approx sigma from FWHM
            else:
                sig0 = 100.0
                
            sig0 = max(sig0, 5.0) # prevent 0 sigma

            try:
                # First pass fit
                popt, _ = curve_fit(gauss, x_window, y_window,
                                    p0=[a0, mu0, sig0], maxfev=10000)
                a1, mu1, sig1 = popt
                sig1 = abs(sig1)
                
                # Second pass: tighten window to ±2 sigma
                mask_strict = (x_fit >= mu1 - 2.5 * sig1) & (x_fit <= mu1 + 2.5 * sig1)
                if np.sum(mask_strict) >= 3:
                    popt2, _ = curve_fit(gauss, x_fit[mask_strict], y_fit[mask_strict],
                                         p0=[a1, mu1, sig1], maxfev=10000)
                    a_f, mu_f, sig_f = popt2
                    sig_f = abs(sig_f)
                else:
                    a_f, mu_f, sig_f = a1, mu1, sig1

                # Draw the fitted curve
                x_line = np.linspace(mu_f - 4*sig_f, mu_f + 4*sig_f, 400)
                plt.plot(x_line, gauss(x_line, a_f, mu_f, sig_f),
                         color="red", linewidth=2,
                         label=f"Gaussian: μ={mu_f:.3g}, σ={sig_f:.3g}")
                plt.legend()
            except Exception as e:
                print(f"Fit failed for {title}: {e}")
        else:
            print(f"Not enough points near peak for {title}.")

    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)
    _add_entry_box(plt.gca(), len(arr))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    
    if 'sig_f' in locals() and sig_f is not None:
        return float(sig_f)
    return None


# ──────────────────────────────────────────────────────────────────
# 1-D  segmented histogram  (splits on macroscopic time gaps)
# ──────────────────────────────────────────────────────────────────

def plot_t_diff_segmented(vals, out_path, title_base, nbins=100,
                          xlabel="\\langle T \\rangle - T_{peak}  (ps)", color="teal", gap_threshold=1e10):
    """Plot 1-D histogram of continuous time with macroscopic gaps, segmented by time leaps."""
    if not vals:
        print(f"No valid points for {title_base}.")
        return None

    # Sort the values to find gaps
    arr = np.sort(np.asarray(vals, dtype=float))
    diffs = np.diff(arr)
    # 1e10 ps = 10 ms jump between events is essentially a new segment run
    split_indices = np.where(diffs > gap_threshold)[0] + 1
    segments = np.split(arr, split_indices)

    n_seg = len(segments)
    fig, axes = plt.subplots(n_seg, 1, figsize=(7, 4 * n_seg))
    if n_seg == 1:
        axes = [axes]

    fitted_sigmas = []

    for i, (seg, ax) in enumerate(zip(segments, axes)):
        # filtering outliers inside the local segment (±3 sigma)
        mu = float(np.mean(seg))
        sig = float(np.std(seg, ddof=1))
        if sig > 0:
            lo = mu - 3.0 * sig
            hi = mu + 3.0 * sig
            seg_clean = seg[(seg >= lo) & (seg <= hi)]
            hist_range = (lo, hi)
        else:
            seg_clean = seg
            hist_range = (mu - 0.5, mu + 0.5)

        counts, bins, _ = ax.hist(seg_clean, bins=nbins, range=hist_range, alpha=0.75, color=color, edgecolor="white")

        # Gaussian fit
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        mask = counts > 0
        x_fit = bin_centers[mask]
        y_fit = counts[mask]
        sig_f = None
        if x_fit.size >= 3:
            max_idx = np.argmax(y_fit)
            peak_x = x_fit[max_idx]
            a0 = float(y_fit[max_idx])

            init_window = 500.0
            mask_init = (x_fit >= peak_x - init_window) & (x_fit <= peak_x + init_window)
            if np.sum(mask_init) >= 3:
                mu0 = peak_x
                y_window = y_fit[mask_init]
                x_window = x_fit[mask_init]
                half_max = a0 / 2.0
                above_half = x_window[y_window >= half_max]
                if len(above_half) >= 2:
                    sig0 = (above_half[-1] - above_half[0]) / 2.355
                else:
                    sig0 = 100.0
                sig0 = max(sig0, 5.0)

                try:
                    popt, _ = curve_fit(gauss, x_window, y_window, p0=[a0, mu0, sig0], maxfev=10000)
                    a1, mu1, sig1 = popt
                    sig1 = abs(sig1)

                    mask_strict = (x_fit >= mu1 - 2.5 * sig1) & (x_fit <= mu1 + 2.5 * sig1)
                    if np.sum(mask_strict) >= 3:
                        popt2, _ = curve_fit(gauss, x_fit[mask_strict], y_fit[mask_strict], p0=[a1, mu1, sig1], maxfev=10000)
                        a_f, mu_f, sig_f = popt2
                        sig_f = abs(sig_f)
                    else:
                        a_f, mu_f, sig_f = a1, mu1, sig1

                    x_line = np.linspace(mu_f - 4*sig_f, mu_f + 4*sig_f, 400)
                    ax.plot(x_line, gauss(x_line, a_f, mu_f, sig_f), color="red", linewidth=2, label=f"Fit: μ={mu_f:.3g}, σ={sig_f:.3g}")
                    ax.legend()
                except Exception:
                    pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        title_suffix = f"(Segment {i+1}/{n_seg})" if n_seg > 1 else ""
        ax.set_title(f"{title_base} {title_suffix}".strip())
        _add_entry_box(ax, len(seg_clean))
        ax.grid(True, alpha=0.3)
        fitted_sigmas.append(sig_f)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved segmented plot: {out_path}")

    valid_sigmas = [s for s in fitted_sigmas if s is not None]
    if valid_sigmas:
        return sum(valid_sigmas) / len(valid_sigmas)
    return None


# ──────────────────────────────────────────────────────────────────
# 1-D  aligned segmented histogram
# ──────────────────────────────────────────────────────────────────

def plot_t_diff_aligned(vals, out_path, title, nbins=100,
                        xlabel="Aligned \\langle T \\rangle - T_{peak,abs}  (ps)", color="teal", gap_threshold=1e10):
    """
    Plot 1-D histogram of continuous time with macroscopic gaps aligned by median offset.
    For each segment, O_s = median(<T> - T_peak,seg).
    The plotted value is (<T> - T_peak) - O_s.
    """
    if not vals:
        print(f"No valid points for {title}.")
        return None

    arr = np.sort(np.asarray(vals, dtype=float))
    diffs = np.diff(arr)
    # Identify macroscopic segments
    split_indices = np.where(diffs > gap_threshold)[0] + 1
    segments = np.split(arr, split_indices)

    aligned_vals = []

    for seg in segments:
        if len(seg) > 0:
            o_s = np.median(seg)
            aligned_seg = seg - o_s
            aligned_vals.extend(aligned_seg.tolist())

    # filtering outliers on the combined aligned data
    mu = float(np.mean(aligned_vals))
    sig = float(np.std(aligned_vals, ddof=1))
    if sig > 0:
        lo = mu - 3.0 * sig
        hi = mu + 3.0 * sig
        aligned_clean = [x for x in aligned_vals if lo <= x <= hi]
        hist_range = (lo, hi)
    else:
        aligned_clean = aligned_vals
        hist_range = (mu - 0.5, mu + 0.5)

    return plot_t_diff(aligned_clean, out_path, title, nbins=nbins,
                       hist_range=hist_range, xlabel=xlabel, color=color)


# ──────────────────────────────────────────────────────────────────
# 1-D  energy histogram  +  optional Landau (Moyal) fit
# ──────────────────────────────────────────────────────────────────

def plot_energy(vals, out_path, title, nbins=100, hist_range=None, fit_landau=True):
    """Plot 1-D energy histogram and optionally perform a Landau (Moyal) fit.
    Returns the lower and upper cut thresholds derived from the fit, or (None, None).
    """
    if not vals:
        print(f"No valid energy values for {title}.")
        return None, None

    arr = np.asarray(vals, dtype=float)
    plt.figure(figsize=(7, 5))
    counts, bin_edges, _ = plt.hist(arr, bins=nbins, range=hist_range, alpha=0.75,
                                    color="darkgoldenrod", edgecolor="white")
    
    mu_raw = float(np.mean(arr))
    rms_raw = float(np.std(arr, ddof=1))
    
    mpv, scale = None, None
    if fit_landau:
        # ---- Landau (Moyal) Binned Peak Fit ----
        # To avoid the long energy tail skewing the peak, fit only around the maximum bin.
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        max_bin_idx = np.argmax(counts)
        peak_x = bin_centers[max_bin_idx]
        max_count = counts[max_bin_idx]
        
        # Define a fitting window around the peak
        # Left boundary: where counts rise to ~20% of max
        # Right boundary: where counts drop to ~40% of max
        left_idx = max_bin_idx
        while left_idx > 0 and counts[left_idx] > 0.2 * max_count:
            left_idx -= 1
            
        right_idx = max_bin_idx
        while right_idx < len(counts) - 1 and counts[right_idx] > 0.4 * max_count:
            right_idx += 1
            
        # Ensure we have a valid window
        if right_idx - left_idx > 3:
            fit_x = bin_centers[left_idx:right_idx]
            fit_y = counts[left_idx:right_idx]
            
            def moyal_scaled(x, amp, loc, scale):
                return amp * moyal.pdf(x, loc=loc, scale=scale)
                
            try:
                # Initial guess: Amplitude = max_count * scale_guess, loc = peak_x, scale = typical width
                p0 = [max_count * 20.0, peak_x, 20.0]
                popt, _ = curve_fit(moyal_scaled, fit_x, fit_y, p0=p0, bounds=([0, 0, 0.1], [np.inf, np.inf, np.inf]))
                amp, mpv, scale = popt
                
                # Plot the continuous fit curve over the full range
                x_line = np.linspace(np.min(arr), np.max(arr), 300)
                y_line = moyal_scaled(x_line, amp, mpv, scale)
                plt.plot(x_line, y_line, color='black', linewidth=2, label=f'Landau Fit\nMPV={mpv:.1f}, Scale={scale:.1f}')
                
                # Calculate standard cut thresholds 
                # (adjusting based on the user's reference image which shows a tighter cut)
                cut_lo = mpv - 1.0 * scale
                cut_hi = mpv + 3.0 * scale
                
                # Draw vertical lines for the cut window
                plt.axvline(cut_lo, color='black', linestyle='--', alpha=0.8, label='Cut Range')
                plt.axvline(cut_hi, color='black', linestyle='--', alpha=0.8)
                
                plt.legend(loc='upper right')
            except Exception as e:
                print(f"Binned Moyal fit failed for {title}: {e}")
        
    plt.xlabel("Energy  [a.u.]")
    plt.ylabel("Counts")
    
    title_params = f"Raw Mean={mu_raw:.1f}  RMS={rms_raw:.1f}"
    if mpv is not None:
        title_params += f"\nFit MPV={mpv:.1f} Scale={scale:.1f}"
    plt.title(f"{title}\n{title_params}")
    _add_entry_box(plt.gca(), len(arr))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    
    if fit_landau and mpv is not None and scale is not None:
        return cut_lo, cut_hi
    return None, None


# ──────────────────────────────────────────────────────────────────
# 2-D  phase-vs-energy  scatter  +  polynomial time-walk fit
# ──────────────────────────────────────────────────────────────────

def plot_phi_vs_energy(phi_vals, energy_vals, out_path, title, ylabel="$\\phi - \\phi_{peak}$ (ps)", fit_poly=True):
    """Plot 2-D scatter (hist2d) of phase diff vs energy, and optionally fit a polynomial curve."""
    if not phi_vals or not energy_vals or len(phi_vals) != len(energy_vals):
        print(f"No valid pairs for {title}.")
        return None, None

    arr_p = np.asarray(phi_vals, dtype=float)
    arr_e = np.asarray(energy_vals, dtype=float)

    mask = np.isfinite(arr_p) & np.isfinite(arr_e)
    arr_p = arr_p[mask]
    arr_e = arr_e[mask]

    if len(arr_p) < 3:
        print(f"Not enough valid points for {title}.")
        return None, None

    # Filter outliers on the phi axis by finding the densest vertical core
    hist, bin_edges = np.histogram(arr_p, bins=100)
    max_bin = np.argmax(hist)
    peak_phi = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2.0
    
    # Calculate a local standard deviation around the peak to avoid background tails
    local_mask = np.abs(arr_p - peak_phi) < 500.0
    if np.sum(local_mask) > 10:
        sig_core = np.std(arr_p[local_mask], ddof=1)
    else:
        sig_core = np.std(arr_p, ddof=1)
        
    window = max(2.0 * sig_core, 150.0) # enforce a minimum 150ps search window
    valid = np.abs(arr_p - peak_phi) <= window
    arr_p = arr_p[valid]
    arr_e = arr_e[valid]

    if len(arr_p) < 3:
        print(f"Not enough points after outlier rejection for {title}.")
        return None, None

    # ---- 1D Profile Generation ----
    # Define energy bins for the profile
    num_bins = 40
    e_min, e_max = np.min(arr_e), np.max(arr_e)
    bin_edges = np.linspace(e_min, e_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    profile_e = []
    profile_p = []
    profile_err = []
    
    a, b, c = None, None, None
    
    for i in range(num_bins):
        # Find points within the current energy bin
        in_bin = (arr_e >= bin_edges[i]) & (arr_e < bin_edges[i+1])
        if np.sum(in_bin) >= 5: # Require at least 5 points in a bin to calculate statistics
            p_in_bin = arr_p[in_bin]
            mean_p = np.mean(p_in_bin)
            std_p = np.std(p_in_bin, ddof=1)
            # Standard error of the mean
            err_p = std_p / np.sqrt(len(p_in_bin))
            
            profile_e.append(bin_centers[i])
            profile_p.append(mean_p)
            profile_err.append(err_p)
            
    profile_e = np.array(profile_e)
    profile_p = np.array(profile_p)
    profile_err = np.array(profile_err)

    # ---- Polynomial fit for time-walk calibration on the 1D Profile ----
    def poly_fit(x, a, b, c):
        return a * x**2 + b * x + c
        
    if fit_poly and len(profile_e) >= 3:
        try:
            # Fit profile_e as X, profile_p as Y, weighted by inverse square error
            # Prevent zero division if err is exactly 0
            sigma_w = np.where(profile_err > 0, profile_err, 1.0)
            popt, _ = curve_fit(poly_fit, profile_e, profile_p, sigma=sigma_w, absolute_sigma=True, method='trf', loss='soft_l1')
            a, b, c = popt
        except Exception as e:
            print(f"Profile polynomial fit failed for {title}: {e}")
    elif fit_poly:
        print(f"Not enough valid profile bins ({len(profile_e)}) for {title} to perform fit.")

    # ---- Plotting and Fit Display ----
    plt.figure(figsize=(8, 6))
    # Use hist2d for a rectangular mesh instead of hexbin
    # Using a coarser mesh (25 bins) so that sparse data has a smoother distribution
    h, xedges, yedges, image = plt.hist2d(arr_e, arr_p, bins=25, cmap='viridis', cmin=1)
    plt.colorbar(image, label='Counts')
    
    # Plot the profile points with red error bars
    if len(profile_e) > 0:
        plt.errorbar(profile_e, profile_p, yerr=profile_err, fmt='o', color='black', ecolor='red', capsize=0, zorder=5, markersize=4, label='Profile Mean')
    
    if fit_poly and a is not None and b is not None and c is not None:
        x_line = np.linspace(e_min, e_max, 100)
        y_line = poly_fit(x_line, a, b, c)
        plt.plot(x_line, y_line, color='red', linewidth=2, 
                 label=f"Fit: $\\phi = {a:.3e} \\cdot E^2 + {b:.3e} \\cdot E + {c:.2f}$")
        plt.legend()
        title = f"{title}\nCalib: $\\phi_{{corr}} = \\phi_{{raw}} - ({a:.3e} \\cdot E^2 + {b:.3e} \\cdot E + {c:.2f})$"

    # Enforce a wider Y-axis range (3x the selection window) for visual consistency
    plt.ylim(peak_phi - 3.0 * window, peak_phi + 3.0 * window)

    plt.xlabel("Energy")
    plt.ylabel(ylabel)
    plt.title(title)
    _add_entry_box(plt.gca(), len(arr_p))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    
    return a, b, c


# ──────────────────────────────────────────────────────────────────
# 2-D  phi_L vs phi_R  (wrap diagnostic)
# ──────────────────────────────────────────────────────────────────

def plot_phi_l_vs_phi_r(phi_l_vals, phi_r_vals, out_path, title,
                        bins=80, hist_range=(0.0, 6250.0)):
    """Plot 2-D histogram of raw phi_L vs phi_R to diagnose wrap issues."""
    if not phi_l_vals or not phi_r_vals or len(phi_l_vals) != len(phi_r_vals):
        print(f"No valid pairs for {title}.")
        return

    arr_l = np.asarray(phi_l_vals, dtype=float)
    arr_r = np.asarray(phi_r_vals, dtype=float)
    mask = np.isfinite(arr_l) & np.isfinite(arr_r)
    arr_l = arr_l[mask]
    arr_r = arr_r[mask]

    if len(arr_l) < 3:
        print(f"Not enough valid points for {title}.")
        return

    plt.figure(figsize=(7, 6))
    h, xedges, yedges, image = plt.hist2d(
        arr_l, arr_r, bins=bins, range=[hist_range, hist_range], cmap="viridis", cmin=1
    )
    plt.colorbar(image, label="Counts")
    plt.xlabel("phi_L  (ps)")
    plt.ylabel("phi_R  (ps)")
    plt.title(title)
    _add_entry_box(plt.gca(), len(arr_l))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────
# 2-D  correlation diagnostic
# ──────────────────────────────────────────────────────────────────

def plot_correlation_2d(x_vals, y_vals, out_path, title, xlabel, ylabel,
                        bins=60, hist_range=None):
    """Plot event-by-event 2D correlation (no fit)."""
    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        print(f"No valid pairs for {title}.")
        return None, None

    arr_x = np.asarray(x_vals, dtype=float)
    arr_y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x = arr_x[mask]
    arr_y = arr_y[mask]

    if len(arr_x) < 3:
        print(f"Not enough valid points for {title}.")
        return None, None

    plt.figure(figsize=(7, 6))
    if hist_range is None:
        plt.hist2d(arr_x, arr_y, bins=bins, cmap="viridis", cmin=1)
    else:
        plt.hist2d(arr_x, arr_y, bins=bins, range=hist_range, cmap="viridis", cmin=1)
    plt.colorbar(label="Counts")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    _add_entry_box(plt.gca(), len(arr_x))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    return None
