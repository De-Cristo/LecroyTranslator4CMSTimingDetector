#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from clock_study_calc import (
    build_template_from_edges,
    collect_interedge_diffs,
    compute_zero_line,
    cross_correlate_align,
    detect_zero_crossings,
    detect_zero_crossings_typed,
    fit_edge_times,
    normalize_trace,
)
from clock_study_plots import (
    save_detected_edges_plot,
    save_high_jitter_plot,
    save_interval_histograms,
    save_linear_fit_plot,
    save_template_artifact,
    save_template_overlay_plot,
    save_template_summary_outputs,
    save_zero_summary_outputs,
)

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
        # Apply per-event trigger_offset correction only
        # (Time_s already contains the absolute time from the scope;
        #  trigger_time is embedded in Time_s, so only trigger_offset is needed)
        if isinstance(trig_arr, np.ndarray) or isinstance(offset_arr, np.ndarray):
            n_events = len(waves)
            for evt in sorted(list(waves.keys())):
                tns, amps = waves[evt]
                # default shift 0
                shift_s = 0.0
                try:
                    idx = int(evt) - 1  # meta arrays are usually 0-based list corresponding to segment index
                    if isinstance(offset_arr, np.ndarray) and idx >= 0 and idx < len(offset_arr):
                        shift_s = float(offset_arr[idx])
                except Exception as e:
                    print(f"[warn] Could not read trigger_offset for event {evt}: {e}", flush=True)
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


def process_group(group, out_dir, plot_first=5, polarity='rising', method='zero', zero_line_override=None,
                 high_jitter_threshold_ps=None, template_min_corr=None, drop_last_edge=0,
                 min_edge_spacing_ns=None, disable_template_correction=False, save_csv_details=False):
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
            return None, None, None, None
    else:
        print(f"[warn] Unsupported group type ({type(group)}), skipping")
        return None, None, None, None

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
    edges_rejected_lowcorr = 0
    if run_zero:
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
                out_png = os.path.join(plots_dir, f"clock_evt{evt}_{base}.png")
                save_detected_edges_plot(
                    t_rel,
                    a,
                    edges,
                    out_png,
                    evt,
                    compute_zero_line(a, zero_line_override=zero_line_override),
                )
                nplots += 1

    # At this point rows_zero may contain zero-crossing results

    # === Inter-edge differences (x1-x0, x2-x1, ...) across events for this file ===
    # Compute diffs from zero-crossing rows only when zero method was used for output
    rows_zero_fit = []
    zero_linfit_plots = 0
    
    if method == 'zero':
        df_zero_tmp = pd.DataFrame(rows_zero) if len(rows_zero) > 0 else pd.DataFrame(columns=['eventNo','edge_time_ns_rel'])
        diffs_all = collect_interedge_diffs(df_zero_tmp, 'edge_time_ns_rel')

        if len(diffs_all) > 0:
            save_interval_histograms(
                diffs_all,
                plots_dir,
                out_dir,
                base,
                hist_prefix='clock_interedge_hist',
                stats_prefix='clock_interedge_stats',
                color='C0',
                title_prefix='Inter-edge intervals',
                save_stats=save_csv_details,
            )
        else:
            print(f"[warn] No inter-edge intervals found for {base}")

        # === Multi-edge linear fit for Zero-Crossing method ===
        for evt, g_evt in df_zero_tmp.groupby('eventNo'):
            fit_result = fit_edge_times(g_evt['edge_time_ns_rel'].dropna().to_numpy())
            if fit_result is None:
                continue

            rows_zero_fit.append({
                'eventNo': int(evt),
                'n_edges_total': fit_result['n_edges_total'],
                'n_edges_used': fit_result['n_edges_used'],
                't0_ns': fit_result['t0_ns'],
                'tclk_ns': fit_result['tclk_ns'],
                'sigma_single_edge_ns': fit_result['sigma_single_edge_ns'],
                'sigma_t0_ns': fit_result['sigma_t0_ns'],
                'sigma_t_ave_ns': fit_result['sigma_t_ave_ns'],
                't_ave_ps': fit_result['t_ave_ps'],
                'source_file': os.path.basename(infile)
            })

            # Diagnostic plot for the first `plot_first` events
            try:
                if zero_linfit_plots < plot_first:
                    fit_png = os.path.join(plots_dir, f"clock_evt{evt}_zero_linfit_{base}.png")
                    fit_label = None
                    if fit_result['n_edges_used'] >= 2:
                        fit_label = (
                            f"fit: t0={fit_result['t0_ns']:.3f} ns, "
                            f"Tclk={fit_result['tclk_ns']:.4f} ns"
                        )
                    save_linear_fit_plot(
                        fit_result['edge_indices_used'],
                        fit_result['precise_times_used'],
                        fit_result['fit_vals'],
                        fit_png,
                        f'Event {evt} zero-cross linear fit',
                        'zero-cross times',
                        fit_label=fit_label,
                    )
                    zero_linfit_plots += 1
            except Exception:
                pass

    # ------------------ Template/correlation method ------------------

    template_pre_ns = 1.0
    template_post_ns = 5.0
    template_max_cycles = 200
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
        try:
            if template_r is not None:
                save_template_artifact(
                    template_r, t_axis_r, plots_dir, out_dir, base, 'rising', save_csv_details
                )
            if template_f is not None:
                save_template_artifact(
                    template_f, t_axis_f, plots_dir, out_dir, base, 'falling', save_csv_details
                )
        except Exception as e:
            print(f"[warn] Failed saving template visuals: {e}")

        n_overlay = 0
        high_jitter_plot_count = 0
        template_linfit_plots = 0
        for evt in sorted(waves.keys()):
            tns, a = waves[evt]
            t_rel = tns - tns[0]
            if polarity == 'both' or (template_r is not None and template_f is not None):
                snippet_edges = detect_zero_crossings_typed(t_rel, a, zero_line_override=zero_line_override)
            else:
                detect_pol = chosen_template_polarity if chosen_template_polarity is not None else polarity
                snippet_edges = detect_zero_crossings(t_rel, a, polarity=detect_pol, zero_line_override=zero_line_override)
            if len(snippet_edges) == 0:
                continue
            precise_times = []
            last_time_kept = None
            for edge_idx, ed in enumerate(snippet_edges):
                if len(ed) == 3:
                    t_edge, _, zero_line = ed
                    edge_type = chosen_template_polarity
                else:
                    t_edge, _, zero_line, edge_type = ed
                if edge_type == 'rising':
                    cur_template = template_r
                    cur_t_axis = t_axis_r
                    cur_dt = dt_r
                elif edge_type == 'falling':
                    cur_template = template_f
                    cur_t_axis = t_axis_f
                    cur_dt = dt_f
                else:
                    if template_r is not None:
                        cur_template = template_r
                        cur_t_axis = t_axis_r
                        cur_dt = dt_r
                    elif template_f is not None:
                        cur_template = template_f
                        cur_t_axis = t_axis_f
                        cur_dt = dt_f
                    else:
                        cur_template = None
                if cur_template is None:
                    continue
                ts = t_edge + cur_t_axis
                try:
                    snippet = np.interp(ts, t_rel, a)
                except Exception:
                    continue
                snippet_norm = normalize_trace(snippet)
                if snippet_norm is None:
                    continue
                
                if not disable_template_correction:
                    shift_ns, lag, peak, peak_norm = cross_correlate_align(snippet_norm, cur_template, cur_dt)
                else:
                    shift_ns, lag, peak, peak_norm = 0.0, 0.0, 1.0, 1.0

                if np.isnan(shift_ns):
                    continue
                if template_min_corr is not None and not disable_template_correction:
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
                    out_ov = os.path.join(plots_dir, f"clock_evt{evt}_edge{edge_idx}_template_overlay_{base}.png")
                    save_template_overlay_plot(
                        cur_t_axis,
                        cur_template,
                        snippet_norm,
                        shift_ns,
                        out_ov,
                        evt,
                        edge_idx,
                        edge_type,
                    )
                    n_overlay += 1

            if len(precise_times) == 0:
                continue

            fit_result = fit_edge_times(precise_times, drop_last=drop_last_edge)
            if fit_result is None:
                continue

            try:
                if template_linfit_plots < plot_first:
                    fit_png = os.path.join(plots_dir, f"clock_evt{evt}_template_linfit_{base}.png")
                    fit_label = None
                    if fit_result['n_edges_used'] >= 2:
                        fit_label = (
                            f"fit: t0={fit_result['t0_ns']:.3f} ns, "
                            f"Tclk={fit_result['tclk_ns']:.4f} ns"
                        )
                    save_linear_fit_plot(
                        fit_result['edge_indices_used'],
                        fit_result['precise_times_used'],
                        fit_result['fit_vals'],
                        fit_png,
                        f'Event {evt} template linear fit',
                        'precise times',
                        fit_label=fit_label,
                    )
                    template_linfit_plots += 1
            except Exception:
                pass

            rows_template_fit.append({
                'eventNo': int(evt),
                'n_edges_total': fit_result['n_edges_total'],
                'n_edges_used': fit_result['n_edges_used'],
                't0_ns': fit_result['t0_ns'],
                'tclk_ns': fit_result['tclk_ns'],
                'sigma_single_edge_ns': fit_result['sigma_single_edge_ns'],
                'sigma_t0_ns': fit_result['sigma_t0_ns'],
                'sigma_t_ave_ns': fit_result['sigma_t_ave_ns'],
                't_ave_ps': fit_result['t_ave_ps'],
                't_event_ref_ns': float(tns[0]) if (tns is not None and len(tns) > 0) else np.nan,
                't0_abs_ns': (
                    float(tns[0] + fit_result['t0_ns'])
                    if (tns is not None and len(tns) > 0 and not np.isnan(fit_result['t0_ns']))
                    else np.nan
                ),
                'source_file': os.path.basename(infile)
            })

            is_high_jitter = (
                high_jitter_threshold_ns is not None
                and not np.isnan(fit_result['sigma_t0_ns'])
                and fit_result['sigma_t0_ns'] >= high_jitter_threshold_ns
            )
            if is_high_jitter and high_jitter_plot_count < plot_first:
                dbg_png = os.path.join(plots_dir, f"clock_evt{evt}_highjitter_{base}.png")
                save_high_jitter_plot(
                    t_rel,
                    a,
                    fit_result['precise_times_all'],
                    fit_result['edge_indices_used'],
                    fit_result['precise_times_used'],
                    fit_result['fit_vals'],
                    fit_result['tclk_ns'],
                    fit_result['sigma_t0_ns'],
                    dbg_png,
                    evt,
                )
                high_jitter_plot_count += 1

        if template_min_corr is not None and edges_rejected_lowcorr > 0:
            print(f"[info] Rejected {edges_rejected_lowcorr} template edges in {base} with corr < {template_min_corr}")

    # Prepare DataFrames to return (and also save per-file CSVs)
    df_zero = pd.DataFrame(rows_zero) if len(rows_zero) > 0 else pd.DataFrame(columns=['eventNo','edge_time_ns_rel'])
    df_zero_fit = pd.DataFrame(rows_zero_fit) if len(rows_zero_fit) > 0 else pd.DataFrame(columns=['eventNo','n_edges_total','n_edges_used','t0_ns','tclk_ns','sigma_single_edge_ns','sigma_t0_ns','sigma_t3_ns'])
    df_template_edges = pd.DataFrame(rows_template_edges) if len(rows_template_edges) > 0 else pd.DataFrame(columns=['eventNo','edge_index','precise_time_ns'])
    df_template_fit = pd.DataFrame(rows_template_fit) if len(rows_template_fit) > 0 else pd.DataFrame(columns=['eventNo','n_edges_total','n_edges_used','t0_ns','tclk_ns','sigma_single_edge_ns','sigma_t0_ns','sigma_t3_ns','t_event_ref_ns','t0_abs_ns'])

    # === Inter-edge differences for Template method ===
    if method == 'template' and not df_template_edges.empty:
        diffs_tpl = collect_interedge_diffs(df_template_edges, 'precise_time_ns')
        if len(diffs_tpl) > 0:
            save_interval_histograms(
                diffs_tpl,
                plots_dir,
                out_dir,
                base,
                hist_prefix='clock_interedge_hist_template',
                stats_prefix='clock_interedge_stats_template',
                color='C2',
                title_prefix='Template Inter-edge intervals',
                save_stats=save_csv_details,
            )
        else:
            print(f"[warn] No template inter-edge intervals found for {base}")

    print(f"[info] process_group done: rows_zero={len(df_zero)} rows_zero_fit={len(df_zero_fit)} rows_template_edges={len(df_template_edges)} rows_template_fit={len(df_template_fit)} for {base}")

    # Save per-file CSVs for clarity
    if save_csv_details:
        if method == 'zero' and len(df_zero) > 0:
            csv_zero_path = os.path.join(out_dir, f'clock_edges_zero_{base}.csv')
            df_zero.to_csv(csv_zero_path, index=False, float_format='%.9g')
            print(f"[ok] Saved per-file zero-cross CSV: {csv_zero_path}")
            if len(df_zero_fit) > 0:
                csv_zero_fit_path = os.path.join(out_dir, f'clock_zero_fit_{base}.csv')
                df_zero_fit.to_csv(csv_zero_fit_path, index=False, float_format='%.9g')
                print(f"[ok] Saved per-file zero-cross fit CSV: {csv_zero_fit_path}")
        if len(df_template_edges) > 0:
            csv_tpl_edges_path = os.path.join(out_dir, f'clock_edges_template_precise_{base}.csv')
            df_template_edges.to_csv(csv_tpl_edges_path, index=False, float_format='%.9g')
            print(f"[ok] Saved per-file template precise-edge CSV: {csv_tpl_edges_path}")
        if len(df_template_fit) > 0:
            csv_tpl_fit_path = os.path.join(out_dir, f'clock_template_fit_{base}.csv')
            df_template_fit.to_csv(csv_tpl_fit_path, index=False, float_format='%.9g')
            print(f"[ok] Saved per-file template fit CSV: {csv_tpl_fit_path}")

    return df_zero, df_template_edges, df_template_fit, df_zero_fit


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
    ap.add_argument('--disable-template-correction', action='store_true', help='If set, use the rough zero-cross time without applying the template cross-correlation shift')
    ap.add_argument('--min-edge-spacing-ns', type=float, default=1.0, help='Minimum spacing between template edges (ns) to accept multiple edges in one event')
    ap.add_argument('--drop-last-edge', type=int, default=0, help='Drop the last N template edges per event before fitting (useful to avoid wrap-around artifacts)')
    ap.add_argument('--save-csv-details', action='store_true', help='If set, save detailed intermediate CSV files for clock checking')

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
    all_dfs_zero_fit = []
    all_dfs_template_edges = []
    all_dfs_template_fit = []
    for i, grp in enumerate(groups):
        print(f"[info] Processing group {i+1}/{len(groups)}: {grp}", flush=True)
        try:
            df_zero, df_template_edges, df_template_fit, df_zero_fit = process_group(
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
                disable_template_correction=args.disable_template_correction,
                save_csv_details=args.save_csv_details,
            )
            print(
                f"[info] process_group returned: df_zero_rows={(len(df_zero) if df_zero is not None else 'None')}, "
                f"df_zero_fit_rows={(len(df_zero_fit) if df_zero_fit is not None else 'None')}, "
                f"df_template_edge_rows={(len(df_template_edges) if df_template_edges is not None else 'None')}, "
                f"df_template_fit_rows={(len(df_template_fit) if df_template_fit is not None else 'None')}",
                flush=True,
            )
        except Exception as e:
            print(f"[error] process_group failed for {grp}: {e}", flush=True)
            continue
        if df_zero is not None:
            all_dfs_zero.append(df_zero)
        if df_zero_fit is not None:
            all_dfs_zero_fit.append(df_zero_fit)
        if df_template_edges is not None:
            all_dfs_template_edges.append(df_template_edges)
        if df_template_fit is not None:
            all_dfs_template_fit.append(df_template_fit)

    if not all_dfs_zero and not all_dfs_template_edges and not all_dfs_template_fit and not all_dfs_zero_fit:
        print('[warn] No clock data produced', flush=True)
        return

    out_df_zero = pd.concat(all_dfs_zero, ignore_index=True) if all_dfs_zero else pd.DataFrame()
    out_df_zero_fit = pd.concat(all_dfs_zero_fit, ignore_index=True) if all_dfs_zero_fit else pd.DataFrame()
    out_df_template_edges = (
        pd.concat(all_dfs_template_edges, ignore_index=True) if all_dfs_template_edges else pd.DataFrame()
    )
    out_df_template_fit = (
        pd.concat(all_dfs_template_fit, ignore_index=True) if all_dfs_template_fit else pd.DataFrame()
    )

    if args.method == 'zero':
        save_zero_summary_outputs(
            out_df_zero,
            out_df_zero_fit,
            args.out_dir,
            save_csv_details=args.save_csv_details,
        )

    if args.method == 'template':
        save_template_summary_outputs(
            out_df_template_edges,
            out_df_template_fit,
            args.out_dir,
            save_csv_details=args.save_csv_details,
        )

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
