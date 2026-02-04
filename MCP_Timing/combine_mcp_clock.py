#!/usr/bin/env python3
"""
combine_mcp_clock.py

Combine MCP peak reconstruction with clock template-fit t0 results.

Usage example:
  python3 combine_mcp_clock.py \
    --mcp ../trc_out/raw_C1_0004237_0000001_6349_data.csv \
    --clock ../trc_out/raw_C2_0004237_0000001_6347_data.csv \
    --out-dir ../trc_out_MCP_reco/ \
    --clock-plot-first 0

The script will run the clock template fitter (via importing process_group
from clock_study.py) to obtain per-event absolute t0 (column `t0_abs_ns`) and
will run the MCP peak fitter (using functions from MCP_wave_reco.py) to
produce a peaks CSV augmented with a new column `t0_abs_ns` for each segment.

If no t0 is available for an event the value will be NaN.

Author: Licheng Zhang (licheng.zhang@cern.ch)
Date: 2026-01
"""
import os
import argparse
import numpy as np
import pandas as pd

# Import helpers from existing modules in the workspace
try:
    from MCP_wave_reco import load_wave_csv as mcp_load_wave_csv, fit_largest_peak
except Exception as e:
    raise ImportError(f"Failed to import MCP helpers from MCP_wave_reco.py: {e}")

try:
    # Import the process_group function from clock_study which returns the template-fit DataFrame
    from clock_study import process_group as clock_process_group
except Exception as e:
    raise ImportError(f"Failed to import clock_study.process_group: {e}")


def build_t0_map_from_clock(clock_csv, out_dir, plot_first=0, polarity='rising',
                            method='template', template_min_corr=None, drop_last_edge=0,
                            min_edge_spacing_ns=None, high_jitter_threshold_ps=None):
    """Run clock template processing and return mapping eventNo -> t0_abs_ns.
    Uses process_group to compute per-event template fits and reads the resulting
    DataFrame returned by the function.
    """
    # process_group returns (df_zero, df_template_edges, df_template_fit)
    _, _, df_fit = clock_process_group(
        clock_csv,
        out_dir,
        plot_first=plot_first,
        polarity=polarity,
        method=method,
        zero_line_override=None,
        high_jitter_threshold_ps=high_jitter_threshold_ps,
        template_min_corr=template_min_corr,
        drop_last_edge=drop_last_edge,
        min_edge_spacing_ns=min_edge_spacing_ns,
    )

    if df_fit is None or df_fit.empty:
        print("[warn] No template-fit results returned by clock processing; t0 map will be empty")
        return {}

    # Expect column 't0_abs_ns' added by clock_study
    if 't0_abs_ns' not in df_fit.columns:
        print("[warn] clock template-fit DataFrame has no 't0_abs_ns' column; t0 map will be empty")
        return {}

    # Build mapping eventNo -> t0_abs_ns
    df_map = df_fit.set_index('eventNo')['t0_abs_ns']
    t0_map = df_map.to_dict()
    return t0_map


def build_mcp_peaks_with_t0(mcp_csv, t0_map, out_dir, plot_first=5, min_amp=None):
    """Run MCP peak reconstruction (using load_wave_csv + fit_largest_peak) and
    attach t0_abs_ns for each event (segment) using t0_map.

    Returns the DataFrame written to CSV.
    """
    w, meta = mcp_load_wave_csv(mcp_csv)
    segments = sorted(w.keys())
    rows = []
    base = os.path.splitext(os.path.basename(mcp_csv))[0]

    for k, seg in enumerate(segments):
        t_abs, a = w[seg]
        m = meta.get(seg, {})
        # reconstruct trigger reference in ns if present in meta
        trig_s = m.get('trigger_time_s', None)
        off_s = m.get('trigger_offset_s', None)
        if trig_s is not None and off_s is not None:
            trigger_ns = 1e9 * (float(trig_s) + float(off_s))
        else:
            # fallback: treat first sample as event reference
            trigger_ns = float(t_abs[0]) if len(t_abs) > 0 else 0.0

        # time relative to trigger (same convention as MCP_wave_reco.run_single)
        t_rel = t_abs - trigger_ns

        # Fit largest peak
        r = fit_largest_peak(t_rel, a, plot=False, title=f"Segment {seg} – fit", save_path=None)

        # lookup t0_abs for this event/segment (event numbers are integers matching seg)
        t0_abs = t0_map.get(int(seg), np.nan)

        rows.append({
            'segment': int(seg),
            'peak_time_ns': r['peak_time_ns'],
            'peak_amp': r['peak_amp'],
            'peak_sigma_ns': r['peak_sigma_ns'],
            'baseline': r['baseline'],
            'fit_success': bool(r['fit_success']),
            't0_abs_ns': float(t0_abs) if (t0_abs is not None and not np.isnan(t0_abs)) else np.nan,
            # also store times in picoseconds with higher numeric precision
            'peak_time_ps': (float(r['peak_time_ns']) * 1000.0) if (r.get('peak_time_ns') is not None and not np.isnan(r.get('peak_time_ns'))) else np.nan,
            't0_abs_ps': (float(t0_abs) * 1000.0) if (t0_abs is not None and not np.isnan(t0_abs)) else np.nan,
        })

    df = pd.DataFrame(rows)
    # Apply amplitude threshold if requested
    if min_amp is not None:
        df = df[df['peak_amp'].abs() >= float(min_amp)].copy()

    if df.empty:
        print("[warn] No MCP peak rows after filtering; writing empty CSV")

    out_csv = os.path.join(out_dir, f"peaks_{base}_with_t0.csv")
    os.makedirs(out_dir, exist_ok=True)
    cols_out = ['segment', 'peak_time_ns', 'peak_time_ps', 'peak_amp', 'peak_sigma_ns', 'baseline', 'fit_success', 't0_abs_ns', 't0_abs_ps']
    # use higher precision for float formatting so ps columns show more digits
    df.to_csv(out_csv, index=False, columns=cols_out, float_format='%.12g')
    print(f"[ok] Wrote augmented MCP peaks CSV: {out_csv} (rows={len(df)})")
    return df


def main():
    ap = argparse.ArgumentParser(description='Combine MCP peak recon and clock t0 into one CSV per event')
    ap.add_argument('--mcp', required=True, help='MCP waveform CSV (e.g. raw_C1_..._data.csv)')
    ap.add_argument('--clock', required=True, help='Clock waveform CSV (e.g. raw_C2_..._data.csv)')
    ap.add_argument('--out-dir', default='./combined_out', help='Output directory')
    ap.add_argument('--clock-plot-first', type=int, default=0, help='Number of clock diagnostic plots to save')
    ap.add_argument('--clock-polarity', choices=['rising','falling','both'], default='rising')
    ap.add_argument('--clock-template-min-corr', type=float, default=None)
    ap.add_argument('--clock-drop-last-edge', type=int, default=0)
    ap.add_argument('--clock-min-edge-spacing-ns', type=float, default=1.0)
    ap.add_argument('--clock-high-jitter-threshold-ps', type=float, default=None)
    ap.add_argument('--plot-first', type=int, default=5, help='Number of MCP plots (unused here, kept for compatibility)')
    ap.add_argument('--min-amp', type=float, default=None, help='Minimum peak amplitude to keep')

    args = ap.parse_args()

    # run clock processing
    print(f"[info] Running clock processing on {args.clock}")
    t0_map = build_t0_map_from_clock(
        args.clock,
        args.out_dir,
        plot_first=args.clock_plot_first,
        polarity=args.clock_polarity,
        method='template',
        template_min_corr=args.clock_template_min_corr,
        drop_last_edge=args.clock_drop_last_edge,
        min_edge_spacing_ns=args.clock_min_edge_spacing_ns,
        high_jitter_threshold_ps=args.clock_high_jitter_threshold_ps,
    )

    # run MCP peak reconstruction and attach t0
    print(f"[info] Running MCP peak reconstruction on {args.mcp}")
    df_peaks = build_mcp_peaks_with_t0(args.mcp, t0_map, args.out_dir, plot_first=args.plot_first, min_amp=args.min_amp)

    print('[done] combine_mcp_clock completed')


if __name__ == '__main__':
    main()
