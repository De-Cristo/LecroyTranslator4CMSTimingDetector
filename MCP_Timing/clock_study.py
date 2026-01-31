#!/usr/bin/env python3
print('[start] clock_study.py executing', flush=True)
"""
clock_study.py

Clock edge detection and timing-quality diagnostics for single-channel waveform CSVs.

This script detects clock edges using zero-crossing and template/correlation methods,
produces per-event precise edge timestamps, fits multi-edge linear models to extract
per-event clock phase (t0) and period (Tclk), and saves diagnostic plots and CSVs.

Input CSV Format:
    Expects waveform data in *_data.csv files with columns:
    - Segment: Segment identifier (integer)
    - Time_s: Sample timestamps in seconds
    - Voltage_V: Signal amplitude in volts

Metadata:
    If a matching *_meta.csv (LeCroy-style) exists alongside the waveform CSV, the
    script will attempt to read arrays named "trigger_time" and "trigger_offset"
    and apply per-segment timing corrections before reconstruction.

Usage Examples:
    # Analyze a single CSV file (fast, useful for debugging)
    python3 clock_study.py --input raw_C2_0004237_0000001_6347_data.csv --out-dir ./clock_out --method template --plot-first 5

    # Scan a directory of *_data.csv files and write results
    python3 clock_study.py --dir ./trc_out --out-dir ./clock_out --method template

Notes:
    - The --drop-last-edge option now accepts an integer N (default 0). It drops the
      last N template-detected edges per event before fitting the linear model.
    - Time columns in waveform CSVs are in seconds and are converted to nanoseconds
      internally (Time_s * 1e9).

Author: Licheng Zhang (licheng.zhang@cern.ch)
Date: 2026-01
"""

# Top-level imports required by the script
import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simple local helpers adapted to updated CSV format
# The old dependency on plot_four_channels.py was removed — this script now
# supports the new tidy CSV with columns: Segment, Time_s, Voltage_V

def read_meta_file(meta_path):
    """Parse a LeCroy-style _meta.csv and return a dict.
    Bracketed lists like [a;b;c] are converted to numpy arrays of floats when possible.
    """
    import csv
    meta = {}
    if not os.path.isfile(meta_path):
        return meta
    with open(meta_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # join remainder columns in case values contain commas
            if len(row) >= 2:
                key = row[0].strip()
                val = ",".join(row[1:]).strip()
            else:
                continue
            if val.startswith('[') and val.endswith(']'):
                inner = val[1:-1]
                parts = [p for p in inner.split(';') if p != '']
                try:
                    arr = np.array([float(p) for p in parts], dtype=float)
                    meta[key] = arr
                except Exception:
                    meta[key] = parts
            else:
                # try numeric
                try:
                    meta[key] = float(val)
                except Exception:
                    meta[key] = val
    return meta


# Update load_wave_csv to apply metadata time correction when available
def load_wave_csv(path):
    """
    Read CSV files produced in the new format with columns:
      Segment, Time_s, Voltage_V
    Returns (waves, meta) where waves is dict: eventNo -> (tns, amp)
    and meta is dict: eventNo -> metadata dict (may include trigger_time_s, trigger_offset_s, shift_ns)
    """
    print(f"[debug] load_wave_csv: reading {path}", flush=True)
    df = pd.read_csv(path, low_memory=False)
    cols = {c.lower(): c for c in df.columns}

    # Expecting 'segment','time_s','voltage_v'
    if not ('segment' in cols and 'time_s' in cols and 'voltage_v' in cols):
        available = ','.join(df.columns)
        raise ValueError(f"Unexpected CSV columns for {path}. Found: {available}")

    seg_col = cols['segment']
    time_col = cols['time_s']
    volt_col = cols['voltage_v']

    waves = {}
    meta = {}
    grouped = df.groupby(seg_col)
    for seg, g in grouped:
        try:
            g = g.sort_values(time_col)
            t_s = g[time_col].astype(float).to_numpy()
            t_ns = t_s * 1e9
            amps = g[volt_col].astype(float).to_numpy()
            evt = int(seg)
            waves[evt] = (t_ns, amps)
            meta[evt] = {}
        except Exception as e:
            print(f"[warn] Skipping segment {seg} in {path}: {e}", flush=True)
            continue

    print(f"[debug] load_wave_csv: loaded events={len(waves)} from {path}", flush=True)

    # Try to find an accompanying _meta.csv and apply per-segment time corrections
    dirn = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    # common candidate patterns: base + '_meta.csv', replace '_data' with '_meta' if present
    candidates = [os.path.join(dirn, base + '_meta.csv')]
    if '_data' in base:
        candidates.append(os.path.join(dirn, base.replace('_data', '_meta') + '.csv'))
    # also try base without suffix + '_meta.csv'
    candidates.append(os.path.join(dirn, base + '.meta.csv'))
    candidates.append(os.path.join(dirn, base + ' _meta.csv'))

    meta_found = None
    for p in candidates:
        if os.path.isfile(p):
            meta_found = p
            break
    if meta_found is None:
        # also try finding any file that ends with '_meta.csv' and shares a common prefix
        for fname in os.listdir(dirn):
            if fname.lower().endswith('_meta.csv') and base.split('_data')[0] in fname:
                meta_found = os.path.join(dirn, fname)
                break

    if meta_found is not None:
        print(f"[info] Found meta file for {path}: {meta_found}", flush=True)
        meta_global = read_meta_file(meta_found)
        # Prefer arrays trigger_time and trigger_offset if present
        trig_arr = None
        offset_arr = None
        if 'trigger_time' in meta_global:
            trig_arr = meta_global['trigger_time']
        if 'trigger_offset' in meta_global:
            offset_arr = meta_global['trigger_offset']
        # Apply per-event correction: t_ns += (trigger_time + trigger_offset) * 1e9
        if isinstance(trig_arr, np.ndarray) or isinstance(offset_arr, np.ndarray):
            n_events = len(waves)
            for evt in sorted(list(waves.keys())):
                tns, amps = waves[evt]
                # default shift 0
                shift_s = 0.0
                try:
                    idx = int(evt) - 1  # meta arrays are usually 0-based list corresponding to segment index
                    if isinstance(trig_arr, np.ndarray) and idx >= 0 and idx < len(trig_arr):
                        shift_s = float(trig_arr[idx])
                    if isinstance(offset_arr, np.ndarray) and idx >= 0 and idx < len(offset_arr):
                        shift_s = shift_s + float(offset_arr[idx])
                except Exception as e:
                    print(f"[warn] Could not read trigger/offset for event {evt}: {e}", flush=True)
                if shift_s != 0.0:
                    shift_ns = shift_s * 1e9
                    tns = tns + shift_ns
                    waves[evt] = (tns, amps)
                    meta[evt]['trigger_time_s'] = float(trig_arr[idx]) if (isinstance(trig_arr, np.ndarray) and idx >= 0 and idx < len(trig_arr)) else None
                    meta[evt]['trigger_offset_s'] = float(offset_arr[idx]) if (isinstance(offset_arr, np.ndarray) and idx >= 0 and idx < len(offset_arr)) else None
                    meta[evt]['shift_ns'] = float(shift_ns)
        else:
            print(f"[info] Meta file found but no trigger_time/trigger_offset arrays present; meta keys: {list(meta_global.keys())}", flush=True)
    else:
        print(f"[info] No meta file found for {path}", flush=True)

    return waves, meta


def find_all_channel_groups(dir_path):
    """
    Find CSV files in a directory. Returns a list of file paths (strings).
    This is a simplified replacement for the original grouping logic and
    returns every .csv file found (excluding coincidence files).
    """
    import glob
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory not found: {dir_path}")
    files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {dir_path}")

    # Filter out obvious coincidence / double-underscore files
    def is_coincidence(fname):
        b = os.path.basename(fname).lower()
        return b.startswith('coincidence_') or '__' in b

    waveform_files = [f for f in files if not is_coincidence(f)]
    return waveform_files

# End of local helpers


def detect_zero_crossing(t_rel, a, polarity='rising', zero_line_override=None):
    """Detect first zero crossing relative to zero line (avg(max,min)).
    polarity: 'rising' (neg->pos) or 'falling' (pos->neg) or 'both' (choose first either)
    Returns: (edge_time_ns_rel, sample_index, frac) or (np.nan, -1, np.nan) on failure
    """
    # zero line
    if zero_line_override is not None:
        z = float(zero_line_override)
    else:
        z = 0.5 * (np.nanmax(a) + np.nanmin(a))
    v = a - z
    if polarity == 'rising':
        idx = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    elif polarity == 'falling':
        idx = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
    else:
        idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
        idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
        idx = np.sort(np.concatenate((idx_r, idx_f)))

    if len(idx) == 0:
        return np.nan, -1, z
    j = int(idx[0])
    a0 = a[j]; a1 = a[j+1]
    t0 = float(t_rel[j]); t1 = float(t_rel[j+1])
    denom = (a1 - a0)
    if denom == 0:
        frac = 0.5
    else:
        frac = (z - a0) / denom
        frac = float(frac)
    frac = max(0.0, min(1.0, frac))
    t_edge = t0 + frac * (t1 - t0)
    return t_edge, j, z


def detect_zero_crossings(t_rel, a, polarity='rising', zero_line_override=None):
    """Detect all zero crossings relative to zero line (avg(max,min)).
    Returns a list of (t_edge, sample_index, zero_line). Empty list if none.
    """
    if zero_line_override is not None:
        z = float(zero_line_override)
    else:
        z = 0.5 * (np.nanmax(a) + np.nanmin(a))
    v = a - z
    if polarity == 'rising':
        idxs = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    elif polarity == 'falling':
        idxs = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
    else:
        idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
        idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
        idxs = np.sort(np.concatenate((idx_r, idx_f)))

    out = []
    for j in idxs:
        a0 = a[j]; a1 = a[j+1]
        t0 = float(t_rel[j]); t1 = float(t_rel[j+1])
        denom = (a1 - a0)
        if denom == 0:
            frac = 0.5
        else:
            frac = (z - a0) / denom
            frac = float(frac)
        frac = max(0.0, min(1.0, frac))
        t_edge = t0 + frac * (t1 - t0)
        out.append((t_edge, int(j), float(z)))
    return out


def detect_zero_crossings_typed(t_rel, a, zero_line_override=None):
    """Detect all zero crossings and annotate their type ('rising'|'falling').
    Returns list of (t_edge, sample_index, zero_line, crossing_type).
    """
    if zero_line_override is not None:
        z = float(zero_line_override)
    else:
        z = 0.5 * (np.nanmax(a) + np.nanmin(a))
    v = a - z
    idx_r = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    idx_f = np.where((v[:-1] > 0) & (v[1:] <= 0))[0]
    out = []
    for j in idx_r:
        a0 = a[j]; a1 = a[j+1]
        t0 = float(t_rel[j]); t1 = float(t_rel[j+1])
        denom = (a1 - a0)
        if denom == 0:
            frac = 0.5
        else:
            frac = (z - a0) / denom
            frac = float(frac)
        frac = max(0.0, min(1.0, frac))
        t_edge = t0 + frac * (t1 - t0)
        out.append((t_edge, int(j), float(z), 'rising'))
    for j in idx_f:
        a0 = a[j]; a1 = a[j+1]
        t0 = float(t_rel[j]); t1 = float(t_rel[j+1])
        denom = (a1 - a0)
        if denom == 0:
            frac = 0.5
        else:
            frac = (z - a0) / denom
            frac = float(frac)
        frac = max(0.0, min(1.0, frac))
        t_edge = t0 + frac * (t1 - t0)
        out.append((t_edge, int(j), float(z), 'falling'))
    # sort by time
    out.sort(key=lambda x: x[0])
    return out


def process_group(group, out_dir, plot_first=5, polarity='rising', method='zero', zero_line_override=None,
                 high_jitter_threshold_ps=None, template_min_corr=None, drop_last_edge=0,
                 min_edge_spacing_ns=None):
    """Process one channel group mapping or a direct filepath string.
    If a string is passed, it is treated as the input CSV to analyze.
    The function is not tied to any channel name (e.g. C4) and will use the first
    filepath found if a dict is passed.
    method: 'zero' | 'template'
    zero_line_override: optional fixed amplitude (same units as waveform) to use instead of avg(max,min)
    """
    # Accept either a direct filepath string, or a dict mapping channel names -> file paths
    if isinstance(group, str):
        infile = group
    elif isinstance(group, dict):
        # pick first string-valued entry from the dict
        vals = [v for v in group.values() if isinstance(v, str)]
        if len(vals) >= 1:
            infile = vals[0]
            if len(vals) > 1:
                print(f"[info] process_group: dict provided, using first file: {infile}")
        else:
            print(f"[warn] Group contains no file paths, skipping: {list(group.values())}")
            return None, None, None
    else:
        print(f"[warn] Unsupported group type ({type(group)}), skipping")
        return None, None, None

    print(f"[info] process_group: loading input file: {infile}")
    waves, meta = load_wave_csv(infile)
    # print(f"[info] Loaded waves: events={len(waves)}; meta keys={list(meta.keys()) if isinstance(meta, dict) else 'N/A'}")

    # Separate containers for zero and template results
    rows_zero = []
    rows_template_edges = []
    rows_template_fit = []
    nplots = 0
    base = os.path.splitext(os.path.basename(infile))[0]
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    run_zero = method in ('zero', 'template')
    zero_plots_enabled = (method == 'zero')

    # --- ZERO-CROSSING METHOD (optional) ---
    if run_zero:
        n_fit_plots = 0
        edges_rejected_lowcorr = 0
        for evt in sorted(waves.keys()):
            tns, a = waves[evt]
            # Make times relative to first sample (like two_channel_coincidence)
            t_rel = tns - tns[0]
            try:
                edges = detect_zero_crossings(t_rel, a, polarity=polarity, zero_line_override=zero_line_override)
            except Exception:
                edges = []

            if len(edges) == 0:
                zero_line = float(zero_line_override) if zero_line_override is not None else 0.5 * (np.nanmax(a) + np.nanmin(a))
                # record one NaN row if no edges
                rows_zero.append({
                    'eventNo': int(evt),
                    'edge_time_ns_rel': np.nan,
                    'edge_sample': -1,
                    'zero_line': zero_line,
                    'method': 'zero_cross_avg_maxmin',
                    'source_file': os.path.basename(infile)
                })
            else:
                for (t_edge, samp_idx, zero_line) in edges:
                    rows_zero.append({
                        'eventNo': int(evt),
                        'edge_time_ns_rel': float(t_edge),
                        'edge_sample': int(samp_idx),
                        'zero_line': float(zero_line),
                        'method': 'zero_cross_avg_maxmin',
                        'source_file': os.path.basename(infile)
                    })

            # Diagnostic plot for first events: mark all detected edges
            if zero_plots_enabled and nplots < plot_first:
                fig, ax = plt.subplots(1,1,figsize=(8,3))
                ax.plot(t_rel, a, label='waveform')
                zero_line_plot = 0.5 * (np.nanmax(a) + np.nanmin(a))
                ax.axhline(zero_line_plot, color='gray', linestyle='--', label='zero_line')
                for (t_edge, samp_idx, _) in edges:
                    ax.axvline(t_edge, color='red', linestyle='--')
                    ax.scatter([t_edge], [np.interp(t_edge, t_rel, a)], color='red')
                ax.set_title(f"Event {evt} clock edges (count={len(edges)})")
                ax.set_xlabel('Time (ns, rel)'); ax.set_ylabel('Amplitude')
                ax.legend(loc='best'); ax.grid(True, alpha=0.25)
                plt.tight_layout()
                out_png = os.path.join(plots_dir, f"clock_evt{evt}_{base}.png")
                plt.savefig(out_png, dpi=150); plt.close(fig)
                nplots += 1

    # At this point rows_zero may contain zero-crossing results

    # === Inter-edge differences (x1-x0, x2-x1, ...) across events for this file ===
    # Compute diffs from zero-crossing rows only when zero method was used for output
    if method == 'zero':
        df_zero_tmp = pd.DataFrame(rows_zero) if len(rows_zero) > 0 else pd.DataFrame(columns=['eventNo','edge_time_ns_rel'])
        diffs_all = []
        for evt, g_evt in df_zero_tmp.groupby('eventNo'):
            times = g_evt['edge_time_ns_rel'].dropna().to_numpy()
            if len(times) <= 1:
                continue
            times_sorted = np.sort(times)
            dd = np.diff(times_sorted)
            # filter non-positive or nan diffs
            dd = dd[~np.isnan(dd)]
            dd = dd[dd > 0]
            if len(dd) > 0:
                diffs_all.extend(dd.tolist())

        if len(diffs_all) > 0:
            diffs_all = np.asarray(diffs_all)
            nbins = 100
            fig, ax = plt.subplots(1,1,figsize=(6,4))
            counts, bins, _ = ax.hist(diffs_all, bins=nbins, color='C0', alpha=0.7)
            ax.set_xlabel('Inter-edge interval (ns)')
            ax.set_ylabel('Counts')
            ax.set_title(f'Inter-edge intervals — {base} (N={len(diffs_all)})')

            # Gaussian overlay using mean/std
            mu = float(np.mean(diffs_all))
            sigma = float(np.std(diffs_all, ddof=1)) if len(diffs_all) > 1 else float(np.std(diffs_all))
            x = np.linspace(bins[0], bins[-1], 400)
            bin_width = bins[1] - bins[0]
            scale = len(diffs_all) * bin_width
            gauss = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, scale * gauss, color='red', lw=2, label=f'Gaussian μ={mu:.3f} ns σ={sigma:.3f} ns')
            ax.legend(loc='best')
            plt.tight_layout()
            out_hist = os.path.join(plots_dir, f'clock_interedge_hist_{base}.png')
            plt.savefig(out_hist, dpi=150); plt.close(fig)

            # Zoomed histogram around 6.0 - 6.5 ns to inspect precision
            xmin, xmax = 6.0, 6.5
            in_range = diffs_all[(diffs_all >= xmin) & (diffs_all <= xmax)]
            if len(in_range) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                counts, bins, patches = ax.hist(in_range, bins=50, color='C0', alpha=0.8)
                ax.set_xlim(xmin, xmax)
                ax.set_xlabel('Inter-edge interval (ns)')
                ax.set_ylabel('Counts')
                ax.set_title(f'Inter-edge intervals zoom {xmin}-{xmax} ns — {base} (N={len(in_range)})')

                # Fit (mean/std) for zoomed data and overlay Gaussian
                mu_z = float(np.mean(in_range))
                sigma_z = float(np.std(in_range, ddof=1)) if len(in_range) > 1 else float(np.std(in_range))
                xz = np.linspace(xmin, xmax, 400)
                bin_width_z = bins[1] - bins[0]
                scale_z = len(in_range) * bin_width_z
                gauss_z = (1.0 / (sigma_z * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xz - mu_z) / sigma_z) ** 2)
                ax.plot(xz, scale_z * gauss_z, color='red', lw=2, label=f'Gaussian μ={mu_z:.6f} ns σ={sigma_z:.6f} ns')
                ax.legend(loc='best')

                plt.tight_layout()
                out_zoom = os.path.join(plots_dir, f'clock_interedge_hist_zoom_{base}.png')
                plt.savefig(out_zoom, dpi=150)
                plt.close(fig)
                print(f"[ok] Saved zoomed histogram: {out_zoom}")

                # Save zoom stats
                zoom_stats_path = os.path.join(out_dir, f'clock_interedge_stats_zoom_{base}.csv')
                zoom_stats_df = pd.DataFrame({'n_intervals_zoom': [len(in_range)], 'mean_ns_zoom': [mu_z], 'std_ns_zoom': [sigma_z]})
                zoom_stats_df.to_csv(zoom_stats_path, index=False, float_format='%.12g')
                print(f"[ok] Saved zoom stats: {zoom_stats_path}")
            else:
                print(f"[warn] No inter-edge intervals in {xmin}-{xmax} ns for {base}")

            # Save stats
            stats_path = os.path.join(out_dir, f'clock_interedge_stats_{base}.csv')
            stats_df = pd.DataFrame({'n_intervals': [len(diffs_all)], 'mean_ns': [mu], 'std_ns': [sigma]})
            stats_df.to_csv(stats_path, index=False, float_format='%.9g')
        else:
            print(f"[warn] No inter-edge intervals found for {base}")

    # ------------------ Template/correlation method ------------------

    template_pre_ns = 1.0
    template_post_ns = 5.0
    template_max_cycles = 200

    def build_template_from_edges(waves, meta, polarity='rising', pre_ns=1.0, post_ns=5.0, max_cycles=200, zero_line_override=None):
        """Build template by averaging many edge-centered snippets.
        Returns template array and time axis (ns) relative to edge center (start of window).
        """
        snippets = []
        dt_ns = None
        for i, evt in enumerate(sorted(waves.keys())):
            if i >= max_cycles:
                break
            tns, a = waves[evt]
            t_rel = tns - tns[0]
            # estimate dt
            if dt_ns is None:
                if len(t_rel) > 1:
                    dt_ns = float(t_rel[1] - t_rel[0])
                else:
                    continue
            edges = detect_zero_crossings(t_rel, a, polarity=polarity, zero_line_override=zero_line_override)
            # use typed detector when both polarities are possible
            if polarity == 'both':
                edges_typed = detect_zero_crossings_typed(t_rel, a, zero_line_override=zero_line_override)
                # we will handle selection of rising vs falling templates outside
                edges = edges_typed
            else:
                edges = detect_zero_crossings(t_rel, a, polarity=polarity, zero_line_override=zero_line_override)
            if len(edges) == 0:
                continue
            # take first edge for template building (if typed, edges[0] may include type)
            if polarity == 'both':
                t_edge = edges[0][0]
            else:
                t_edge, _, _ = edges[0]
            start = t_edge - pre_ns
            stop = t_edge + post_ns
            # produce uniform-length snippet
            n_points = int(round((post_ns + pre_ns) / dt_ns)) + 1
            ts = np.linspace(start, stop, n_points)
            # interpolate amplitude onto ts
            try:
                snippet = np.interp(ts, t_rel, a)
            except Exception:
                continue
            # normalize snippet amplitude
            if np.nanstd(snippet) == 0:
                continue
            snippet = (snippet - np.mean(snippet)) / np.std(snippet)
            snippets.append(snippet)

        print(f"[info] build_template_from_edges: collected snippets={len(snippets)} dt_ns={dt_ns}")
        if len(snippets) == 0:
            return None, None, None, None
        T = np.mean(np.vstack(snippets), axis=0)
        # time axis relative to window start
        t_axis = np.linspace(-pre_ns, post_ns, len(T))
        return T, t_axis, dt_ns, None


    def cross_correlate_align(snippet, template, dt_ns):
        """Compute cross-correlation alignment between snippet and template.
        Returns shift_ns (positive means snippet occurs later than template), sub-sample lag,
        raw peak value, and the peak normalized by snippet length.
        """
        s = snippet.copy()
        T = template.copy()
        # normalize
        if np.nanstd(s) == 0 or np.nanstd(T) == 0:
            return np.nan, np.nan, np.nan, np.nan
        s_norm = (s - np.mean(s)) / np.std(s)
        T_norm = (T - np.mean(T)) / np.std(T)
        R = np.correlate(s_norm, T_norm, mode='full')
        i0 = int(np.argmax(R))
        N = len(s_norm)
        lag = i0 - (N - 1)  # integer lag in samples
        # quadratic refinement around peak
        if 1 <= i0 < len(R) - 1:
            y0, y1, y2 = R[i0-1], R[i0], R[i0+1]
            denom = (y0 - 2*y1 + y2)
            if denom != 0:
                p = 0.5 * (y0 - y2) / denom
            else:
                p = 0.0
        else:
            p = 0.0
        lag_refined = lag + p
        # we define shift_ns such that positive => s is later than template
        shift_ns = -lag_refined * dt_ns
        peak = R[i0]
        peak_norm = peak / float(N) if N > 0 else np.nan
        return float(shift_ns), float(lag_refined), float(peak), float(peak_norm)


    # Use the newer per-edge template handling later; initialize variables here
    template = None
    t_axis = None
    dt_ns = None
    # placeholders for rising/falling templates (built further down)
    template_r = None; t_axis_r = None; dt_r = None
    template_f = None; t_axis_f = None; dt_f = None
    chosen_template_polarity = None
    any_template = False
    high_jitter_threshold_ns = None
    if high_jitter_threshold_ps is not None:
        try:
            high_jitter_threshold_ns = float(high_jitter_threshold_ps) * 1e-3
        except Exception:
            high_jitter_threshold_ns = None

    # If method == 'template' the newer block below will build rising/falling templates
    if method == 'template':
        try:
            # If user requested polarity='both', build separate rising and falling templates
            if polarity == 'both':
                # rising
                tpl_r = None
                try:
                    tpl_r, t_axis_r, dt_r, _ = build_template_from_edges(
                        waves, meta, polarity='rising', pre_ns=template_pre_ns, post_ns=template_post_ns,
                        max_cycles=template_max_cycles, zero_line_override=zero_line_override)
                    template_r = tpl_r
                except Exception:
                    template_r = None
                # falling
                tpl_f = None
                try:
                    tpl_f, t_axis_f, dt_f, _ = build_template_from_edges(
                        waves, meta, polarity='falling', pre_ns=template_pre_ns, post_ns=template_post_ns,
                        max_cycles=template_max_cycles, zero_line_override=zero_line_override)
                    template_f = tpl_f
                except Exception:
                    template_f = None
                # If only one exists, mark chosen_template_polarity for fallback behavior
                if template_r is not None and template_f is None:
                    chosen_template_polarity = 'rising'
                elif template_f is not None and template_r is None:
                    chosen_template_polarity = 'falling'
                else:
                    chosen_template_polarity = None  # both available or none
            else:
                tpl, t_axis_tpl, dt_tpl, _ = build_template_from_edges(
                    waves, meta, polarity=polarity, pre_ns=template_pre_ns, post_ns=template_post_ns,
                    max_cycles=template_max_cycles, zero_line_override=zero_line_override)
                if tpl is not None:
                    if polarity == 'rising':
                        template_r, t_axis_r, dt_r = tpl, t_axis_tpl, dt_tpl
                        template_f = None; t_axis_f = None; dt_f = None
                    else:
                        template_f, t_axis_f, dt_f = tpl, t_axis_tpl, dt_tpl
                        template_r = None; t_axis_r = None; dt_r = None
                    chosen_template_polarity = polarity
                else:
                    template_r = template_f = None
                    chosen_template_polarity = None
            # final chosen aggregated template presence
            any_template = (template_r is not None) or (template_f is not None)
        except Exception as e:
            print(f"[warn] Template building failed for {base}: {e}")
            template_r = template_f = None
            t_axis_r = t_axis_f = None
            dt_r = dt_f = None
            chosen_template_polarity = None
            any_template = False

    if method == 'template' and (not any_template):
        print(f"[warn] No template could be built for {base}; skipping template mode.", flush=True)

    if any_template and method == 'template':
        # Save available templates (rising/falling)
        try:
            if template_r is not None:
                tpl_png_r = os.path.join(plots_dir, f"clock_template_rising_{base}.png")
                fig, ax = plt.subplots(1,1,figsize=(6,3))
                ax.plot(t_axis_r, template_r, lw=1)
                ax.set_xlabel('Time (ns, rel to edge)'); ax.set_ylabel('Normalized amplitude')
                ax.set_title(f'Template (rising) for {base}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(tpl_png_r, dpi=150); plt.close(fig)
                np.save(os.path.join(out_dir, f'clock_template_rising_{base}.npy'), template_r)
                print(f"[ok] Saved rising template png and numpy: {tpl_png_r}")
            if template_f is not None:
                tpl_png_f = os.path.join(plots_dir, f"clock_template_falling_{base}.png")
                fig, ax = plt.subplots(1,1,figsize=(6,3))
                ax.plot(t_axis_f, template_f, lw=1)
                ax.set_xlabel('Time (ns, rel to edge)'); ax.set_ylabel('Normalized amplitude')
                ax.set_title(f'Template (falling) for {base}')
                ax.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(tpl_png_f, dpi=150); plt.close(fig)
                np.save(os.path.join(out_dir, f'clock_template_falling_{base}.npy'), template_f)
                print(f"[ok] Saved falling template png and numpy: {tpl_png_f}")
        except Exception as e:
            print(f"[warn] Failed saving template visuals: {e}")

        # Align each snippet to appropriate template, timestamp all edges, and fit multi-edge template per event
        n_overlay = 0
        high_jitter_plot_count = 0
        # counter for how many per-event linear-fit plots we've saved
        template_linfit_plots = 0
        for evt in sorted(waves.keys()):
            tns, a = waves[evt]
            t_rel = tns - tns[0]
            # always use typed detection when templates for both polarities may be used
            if polarity == 'both' or (template_r is not None and template_f is not None):
                snippet_edges = detect_zero_crossings_typed(t_rel, a, zero_line_override=zero_line_override)
            else:
                # prefer chosen_template_polarity if specified
                detect_pol = chosen_template_polarity if chosen_template_polarity is not None else polarity
                snippet_edges = detect_zero_crossings(t_rel, a, polarity=detect_pol, zero_line_override=zero_line_override)
            if len(snippet_edges) == 0:
                continue
            precise_times = []
            last_time_kept = None
            for edge_idx, ed in enumerate(snippet_edges):
                # ed may be (t_edge, samp_idx, zero_line) or (t_edge, samp_idx, zero_line, type)
                if len(ed) == 3:
                    t_edge, samp_idx, zero_line = ed
                    edge_type = chosen_template_polarity
                else:
                    t_edge, samp_idx, zero_line, edge_type = ed
                # select template corresponding to edge_type
                if edge_type == 'rising':
                    cur_template = template_r
                    cur_t_axis = t_axis_r
                    cur_dt = dt_r
                elif edge_type == 'falling':
                    cur_template = template_f
                    cur_t_axis = t_axis_f
                    cur_dt = dt_f
                else:
                    # fallback to whichever template exists
                    if template_r is not None:
                        cur_template = template_r; cur_t_axis = t_axis_r; cur_dt = dt_r
                    elif template_f is not None:
                        cur_template = template_f; cur_t_axis = t_axis_f; cur_dt = dt_f
                    else:
                        cur_template = None
                if cur_template is None:
                    continue
                ts = t_edge + cur_t_axis
                try:
                    snippet = np.interp(ts, t_rel, a)
                except Exception:
                    continue
                if np.nanstd(snippet) == 0:
                    continue
                snippet_norm = (snippet - np.mean(snippet)) / np.std(snippet)
                shift_ns, lag, peak, peak_norm = cross_correlate_align(snippet_norm, cur_template, cur_dt)
                if np.isnan(shift_ns):
                    continue
                if template_min_corr is not None:
                    try:
                        if np.isnan(peak_norm) or peak_norm < template_min_corr:
                            edges_rejected_lowcorr += 1
                            continue
                    except Exception:
                        edges_rejected_lowcorr += 1
                        continue
                precise_time = t_edge + shift_ns
                if min_edge_spacing_ns is not None:
                    try:
                        if last_time_kept is not None and (precise_time - last_time_kept) < min_edge_spacing_ns:
                            continue
                    except Exception:
                        pass
                rows_template_edges.append({
                    'eventNo': int(evt),
                    'edge_index': int(edge_idx),
                    'rough_time_ns': float(t_edge),
                    'precise_time_ns': float(precise_time),
                    'shift_ns': float(shift_ns),
                    'lag_samples': float(lag),
                    'xcorr_peak': float(peak),
                    'xcorr_peak_norm': float(peak_norm),
                    'zero_line': float(zero_line),
                    'edge_type': edge_type,
                    'source_file': os.path.basename(infile)
                })
                precise_times.append(float(precise_time))
                last_time_kept = float(precise_time)

                if n_overlay < plot_first:
                    fig, ax = plt.subplots(1,1,figsize=(7,3))
                    ax.plot(cur_t_axis, cur_template, label=f'template ({edge_type})', alpha=0.8)
                    ax.plot(cur_t_axis - shift_ns, snippet_norm, label='snippet (aligned)', alpha=0.9)
                    ax.axvline(0.0, color='gray', linestyle='--', label='edge reference')
                    ax.set_xlabel('Time (ns, rel to template reference)'); ax.set_ylabel('Normalized amplitude')
                    ax.set_title(f'Event {evt} edge {edge_idx} template alignment')
                    ax.legend(loc='best'); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    out_ov = os.path.join(plots_dir, f"clock_evt{evt}_edge{edge_idx}_template_overlay_{base}.png")
                    plt.savefig(out_ov, dpi=150); plt.close(fig)
                    n_overlay += 1

            if len(precise_times) == 0:
                continue

            precise_times = np.asarray(precise_times)
            edge_indices_full = np.arange(len(precise_times))
            precise_times_used = precise_times.copy()
            edge_indices_used = edge_indices_full.copy()
            # drop_last_edge now accepts an integer number of edges to drop from the end
            try:
                n_drop = int(drop_last_edge) if drop_last_edge is not None else 0
            except Exception:
                n_drop = 0
            if n_drop > 0 and len(precise_times_used) >= 2:
                # ensure at least one edge remains when possible
                if n_drop >= len(precise_times_used):
                    if len(precise_times_used) > 1:
                        n_drop = len(precise_times_used) - 1
                    else:
                        n_drop = 0
                if n_drop > 0:
                    precise_times_used = precise_times_used[:-n_drop]
                    edge_indices_used = edge_indices_used[:-n_drop]

            n_edges_total = len(precise_times)
            n_edges_used = len(precise_times_used)
            if n_edges_used >= 2:
                slope, intercept = np.polyfit(edge_indices_used, precise_times_used, 1)
                fit_vals = intercept + slope * edge_indices_used
                residuals = precise_times_used - fit_vals
                sigma_single = float(np.std(residuals, ddof=1)) if n_edges_used > 1 else np.nan
            else:
                slope = np.nan
                intercept = float(precise_times_used[0]) if n_edges_used > 0 else np.nan
                sigma_single = np.nan
            sigma_t0 = sigma_single / np.sqrt(n_edges_used) if (n_edges_used > 0 and not np.isnan(sigma_single)) else np.nan

            # Save a linear-fit diagnostic plot for the first `plot_first` events
            try:
                if template_linfit_plots < plot_first:
                    fig, ax = plt.subplots(1,1,figsize=(6,4))
                    ax.scatter(edge_indices_used, precise_times_used, color='C0', label='precise times')
                    if n_edges_used >= 2:
                        ax.plot(edge_indices_used, fit_vals, color='C1', label=f'fit: t0={intercept:.3f} ns, Tclk={slope:.4f} ns')
                    ax.set_xlabel('Edge index n_j')
                    ax.set_ylabel('Time t_j (ns)')
                    ax.set_title(f'Event {evt} template linear fit')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                    plt.tight_layout()
                    fit_png = os.path.join(plots_dir, f"clock_evt{evt}_template_linfit_{base}.png")
                    plt.savefig(fit_png, dpi=150); plt.close(fig)
                    template_linfit_plots += 1
            except Exception:
                pass

            rows_template_fit.append({
                'eventNo': int(evt),
                'n_edges_total': int(n_edges_total),
                'n_edges_used': int(n_edges_used),
                't0_ns': float(intercept),
                'tclk_ns': float(slope) if not np.isnan(slope) else np.nan,
                'sigma_single_edge_ns': sigma_single,
                'sigma_t0_ns': sigma_t0,
                'source_file': os.path.basename(infile)
            })

            is_high_jitter = (
                high_jitter_threshold_ns is not None
                and not np.isnan(sigma_t0)
                and sigma_t0 >= high_jitter_threshold_ns
            )
            if is_high_jitter and high_jitter_plot_count < plot_first:
                fig, (ax_wave, ax_fit_plot) = plt.subplots(1, 2, figsize=(12, 4))
                ax_wave.plot(t_rel, a, label='waveform')
                edge_amp = np.interp(precise_times, t_rel, a, left=np.nan, right=np.nan)
                ax_wave.scatter(precise_times, edge_amp, color='red', zorder=5, label='precise edges')
                for tt in precise_times:
                    ax_wave.axvline(tt, color='red', alpha=0.3)
                ax_wave.set_xlabel('Time (ns, rel)')
                ax_wave.set_ylabel('Amplitude')
                ax_wave.set_title(f'Event {evt} waveform (σ_t0={sigma_t0*1e3:.1f} ps)')
                ax_wave.legend(loc='best'); ax_wave.grid(True, alpha=0.3)

                ax_fit_plot.scatter(edge_indices_used, precise_times_used, color='C0', label='precise times (fit)')
                if n_edges_used >= 2:
                    ax_fit_plot.plot(edge_indices_used, fit_vals, color='C1', label=f'fit Tclk={slope:.4f} ns')
                ax_fit_plot.set_xlabel('Edge index n_j')
                ax_fit_plot.set_ylabel('Time t_j (ns)')
                ax_fit_plot.set_title('Linear fit diagnostics')
                ax_fit_plot.grid(True, alpha=0.3)
                ax_fit_plot.legend(loc='best')

                plt.tight_layout()
                dbg_png = os.path.join(plots_dir, f"clock_evt{evt}_highjitter_{base}.png")
                plt.savefig(dbg_png, dpi=150)
                plt.close(fig)
                high_jitter_plot_count += 1

        if template_min_corr is not None and edges_rejected_lowcorr > 0:
            print(f"[info] Rejected {edges_rejected_lowcorr} template edges in {base} with corr < {template_min_corr}")

            if n_edges_used >= 2 and n_fit_plots < plot_first:
                fig, ax = plt.subplots(1,1,figsize=(6,4))
                ax.scatter(edge_indices_used, precise_times_used, color='C0', label='precise times')
                ax.plot(edge_indices_used, fit_vals, color='C1', label=f'fit: t0={intercept:.3f} ns, Tclk={slope:.4f} ns')
                ax.set_xlabel('Edge index n_j')
                ax.set_ylabel('Time t_j (ns)')
                ax.set_title(f'Event {evt} template linear fit')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                plt.tight_layout()
                fit_png = os.path.join(plots_dir, f"clock_evt{evt}_template_linfit_{base}.png")
                plt.savefig(fit_png, dpi=150)
                plt.close(fig)
                n_fit_plots += 1

    # Prepare DataFrames to return (and also save per-file CSVs)
    df_zero = pd.DataFrame(rows_zero) if len(rows_zero) > 0 else pd.DataFrame(columns=['eventNo','edge_time_ns_rel'])
    df_template_edges = pd.DataFrame(rows_template_edges) if len(rows_template_edges) > 0 else pd.DataFrame(columns=['eventNo','edge_index','precise_time_ns'])
    df_template_fit = pd.DataFrame(rows_template_fit) if len(rows_template_fit) > 0 else pd.DataFrame(columns=['eventNo','n_edges_total','n_edges_used','t0_ns','tclk_ns','sigma_single_edge_ns','sigma_t0_ns'])

    print(f"[info] process_group done: rows_zero={len(df_zero)} rows_template_edges={len(df_template_edges)} rows_template_fit={len(df_template_fit)} for {base}")

    # Save per-file CSVs for clarity
    if method == 'zero' and len(df_zero) > 0:
        csv_zero_path = os.path.join(out_dir, f'clock_edges_zero_{base}.csv')
        df_zero.to_csv(csv_zero_path, index=False, float_format='%.9g')
        print(f"[ok] Saved per-file zero-cross CSV: {csv_zero_path}")
    if len(df_template_edges) > 0:
        csv_tpl_edges_path = os.path.join(out_dir, f'clock_edges_template_precise_{base}.csv')
        df_template_edges.to_csv(csv_tpl_edges_path, index=False, float_format='%.9g')
        print(f"[ok] Saved per-file template precise-edge CSV: {csv_tpl_edges_path}")
    if len(df_template_fit) > 0:
        csv_tpl_fit_path = os.path.join(out_dir, f'clock_template_fit_{base}.csv')
        df_template_fit.to_csv(csv_tpl_fit_path, index=False, float_format='%.9g')
        print(f"[ok] Saved per-file template fit CSV: {csv_tpl_fit_path}")

    return df_zero, df_template_edges, df_template_fit


def main():
    ap = argparse.ArgumentParser(description='Clock edge study (zero-cross avg(max,min) method)')
    ap.add_argument('--dir', required=False, help='Directory with waveform CSV files')
    # Accept a single input file; keep --file as alias for compatibility
    ap.add_argument('--input', '--file', dest='input', required=False, help='Single CSV file to analyze (faster debug)')
    ap.add_argument('--out-dir', default='./clock_out', help='Output directory')
    ap.add_argument('--plot-first', type=int, default=5, help='How many event diagnostics to save')
    ap.add_argument('--polarity', choices=['rising','falling','both'], default='rising')
    ap.add_argument('--method', choices=['zero', 'template'], default='zero', help="Method to detect clock edges")
    ap.add_argument('--fixed-zero-line', type=float, default=None, help='Override zero-cross threshold with a fixed amplitude (same units as waveform)')
    ap.add_argument('--high-jitter-threshold-ps', type=float, default=None, help='If set, save waveform+fit plots for events whose σ_t0 exceeds this threshold (ps)')
    ap.add_argument('--template-min-corr', type=float, default=None, help='Minimum normalized cross-correlation peak (0-1) required to keep a template edge')
    ap.add_argument('--min-edge-spacing-ns', type=float, default=1.0, help='Minimum spacing between template edges (ns) to accept multiple edges in one event')
    ap.add_argument('--drop-last-edge', type=int, default=0, help='Drop the last N template edges per event before fitting (useful to avoid wrap-around artifacts)')

    args = ap.parse_args()
    print(f"[info] Starting clock_study.py with args: {vars(args)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.dir and not args.input:
        raise ValueError("Either --dir or --input/--file must be specified.")
    if args.dir and args.input:
        raise ValueError("Specify only one of --dir or --input/--file.")

    # Build groups list either from single file or directory
    if args.input:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"File not found: {args.input}")
        groups = [args.input]
    else:
        groups = find_all_channel_groups(args.dir)
        if not groups:
            raise FileNotFoundError('No channel groups found')

    print(f"[info] Found {len(groups)} groups/files. Example: {groups[:2]}", flush=True)

    all_dfs_zero = []
    all_dfs_template_edges = []
    all_dfs_template_fit = []
    for i, grp in enumerate(groups):
        print(f"[info] Processing group {i+1}/{len(groups)}: {grp}", flush=True)
        try:
            df_zero, df_template_edges, df_template_fit = process_group(
                grp,
                args.out_dir,
                plot_first=args.plot_first,
                polarity=args.polarity,
                method=args.method,
                zero_line_override=args.fixed_zero_line,
                high_jitter_threshold_ps=args.high_jitter_threshold_ps,
                template_min_corr=args.template_min_corr,
                drop_last_edge=args.drop_last_edge,
                min_edge_spacing_ns=args.min_edge_spacing_ns,
            )
            print(
                f"[info] process_group returned: df_zero_rows={(len(df_zero) if df_zero is not None else 'None')}, "
                f"df_template_edge_rows={(len(df_template_edges) if df_template_edges is not None else 'None')}, "
                f"df_template_fit_rows={(len(df_template_fit) if df_template_fit is not None else 'None')}",
                flush=True,
            )
        except Exception as e:
            print(f"[error] process_group failed for {grp}: {e}", flush=True)
            continue
        if df_zero is not None:
            all_dfs_zero.append(df_zero)
        if df_template_edges is not None:
            all_dfs_template_edges.append(df_template_edges)
        if df_template_fit is not None:
            all_dfs_template_fit.append(df_template_fit)

    if not all_dfs_zero and not all_dfs_template_edges and not all_dfs_template_fit:
        print('[warn] No clock data produced', flush=True)
        return

    out_df_zero = pd.DataFrame()
    if len(all_dfs_zero) > 0:
        out_df_zero = pd.concat(all_dfs_zero, ignore_index=True)
        if args.method == 'zero':
            out_csv_zero = os.path.join(args.out_dir, 'clock_edges_zero_cross.csv')
            out_df_zero.to_csv(out_csv_zero, index=False, float_format='%.9g')
            print(f"[ok] Wrote summary CSV (zero-cross): {out_csv_zero} (rows={len(out_df_zero)})")

            # Basic summary plot: histogram of edge times
            plt.figure(figsize=(6,4))
            vals = out_df_zero['edge_time_ns_rel'].dropna().values
            if len(vals) > 0:
                plt.hist(vals, bins=100)
                plt.xlabel('Edge time (ns, rel)'); plt.ylabel('Counts'); plt.title('Clock edge times (relative)')
                plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, 'clock_edge_hist_zero_cross.png'), dpi=150); plt.close()
                print(f"[ok] Saved histogram of zero-cross edges")
            else:
                print('[warn] No detected edges to plot')

    out_df_template_edges = pd.DataFrame()
    if len(all_dfs_template_edges) > 0:
        out_df_template_edges = pd.concat(all_dfs_template_edges, ignore_index=True)
        if args.method == 'template' and not out_df_template_edges.empty:
            out_precise_csv = os.path.join(args.out_dir, 'clock_edges_template_precise.csv')
            out_df_template_edges.to_csv(out_precise_csv, index=False, float_format='%.9g')
            print(f"[ok] Wrote precise edge CSV (template): {out_precise_csv} (rows={len(out_df_template_edges)})")

    out_df_template_fit = pd.DataFrame()
    if len(all_dfs_template_fit) > 0:
        out_df_template_fit = pd.concat(all_dfs_template_fit, ignore_index=True)
        if args.method == 'template' and not out_df_template_fit.empty:
            out_fit_csv = os.path.join(args.out_dir, 'clock_template_fit_results.csv')
            out_df_template_fit.to_csv(out_fit_csv, index=False, float_format='%.9g')
            print(f"[ok] Wrote template fit CSV: {out_fit_csv} (rows={len(out_df_template_fit)})")

            # Basic diagnostics: histograms of fitted clock period and sigma_t0
            tclk_vals = out_df_template_fit['tclk_ns'].dropna().to_numpy()
            if len(tclk_vals) > 0:
                mask_zoom = (tclk_vals >= 6.1) & (tclk_vals <= 6.4)
                tclk_zoom = tclk_vals[mask_zoom]
                if len(tclk_zoom) > 0:
                    plt.figure(figsize=(6,4))
                    counts, bins, _ = plt.hist(tclk_zoom, bins=60, color='C1', alpha=0.75, label='data')
                    mu_tclk = float(np.mean(tclk_zoom))
                    sigma_tclk = float(np.std(tclk_zoom, ddof=1)) if len(tclk_zoom) > 1 else float(np.std(tclk_zoom))
                    x = np.linspace(6.1, 6.4, 400)
                    bw = bins[1] - bins[0]
                    gauss = (1.0 / (sigma_tclk * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu_tclk) / sigma_tclk) ** 2)
                    plt.plot(x, len(tclk_zoom) * bw * gauss, color='black', lw=2, label=f'Gaussian μ={mu_tclk:.6f} ns σ={sigma_tclk:.6f} ns')
                    plt.xlabel('Fitted clock period Tclk (ns)'); plt.ylabel('Counts')
                    plt.xlim(6.1, 6.4)
                    plt.title(f'Template fit clock period (6.1-6.4 ns) — entries={len(tclk_zoom)}')
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out_dir, 'clock_template_fit_tclk_hist.png'), dpi=150)
                    plt.close()
                    print(f"[ok] Saved clock period histogram (template fit). Mean={mu_tclk:.6f} ns, σ={sigma_tclk:.6f} ns, entries={len(tclk_zoom)}")
                else:
                    print('[warn] No Tclk entries in 6.1-6.4 ns range for histogram')

            sigma_vals = out_df_template_fit['sigma_t0_ns'].dropna().to_numpy()
            if len(sigma_vals) > 0:
                sigma_ps = sigma_vals * 1e3
                mask_zoom = (sigma_ps >= 0.0) & (sigma_ps <= 50.0)
                sigma_zoom = sigma_ps[mask_zoom]
                if len(sigma_zoom) > 0:
                    plt.figure(figsize=(6,4))
                    counts, bins, _ = plt.hist(sigma_zoom, bins=60, color='C3', alpha=0.75, label='data')
                    mu_sigma = float(np.mean(sigma_zoom))
                    sigma_sigma = float(np.std(sigma_zoom, ddof=1)) if len(sigma_zoom) > 1 else float(np.std(sigma_zoom))
                    x = np.linspace(0.0, 50.0, 400)
                    bw = bins[1] - bins[0]
                    gauss = (1.0 / (sigma_sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu_sigma) / sigma_sigma) ** 2)
                    plt.plot(x, len(sigma_zoom) * bw * gauss, color='black', lw=2, label=f'Gaussian μ={mu_sigma:.3f} ps σ={sigma_sigma:.3f} ps')
                    plt.xlabel('Event t0 jitter σ (ps)'); plt.ylabel('Counts')
                    plt.xlim(0.0, 50.0)
                    plt.title(f'Template fit event jitter (σ_t0) 0-50 ps — entries={len(sigma_zoom)}')
                    plt.legend(loc='best')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out_dir, 'clock_template_fit_sigma_t0_hist.png'), dpi=150)
                    plt.close()
                    print(f"[ok] Saved t0 jitter histogram (template fit). Mean={mu_sigma:.3f} ps, σ={sigma_sigma:.3f} ps, entries={len(sigma_zoom)}")
                else:
                    print('[warn] No sigma_t0 entries in 0-50 ps range for histogram')


    if args.method == 'template' and out_df_template_fit.empty and out_df_template_edges.empty:
        print('[warn] Template mode selected but no template data were produced.', flush=True)

if __name__ == '__main__':
    import traceback, sys
    try:
        main()
    except Exception as e:
        print(f"[fatal] Exception in main: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
