#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Apply mapping results to original ROOT events: read mapping (mapping_results_segN.csv),
read peaks CSVs (one per meta file), and append peak time/amplitude as an entry for a
given channel (default 192) to the ROOT tree, preserving all other branches.

Usage examples:
  # Process a single mapping file (one segment):
  python3 apply_mapping_add_peaks.py \
    --root /eos/cms/.../4237/1_e.root \
    --mapping ../trc_out_sync/mapping_results_0004237_0000001_6347.csv \
    --peaks-dir ../trc_out_MCP_reco \
    --peaks-pattern "peaks_raw_C1_*_{suffix}_data.csv" \
    --channel 192 \
    --out out_with_peaks_seg6347.root

  # Process all mapping files (all segments) in a directory or via glob — script merges all mappings
  # into the single output (keeps original ROOT entry ordering):
  python3 apply_mapping_add_peaks.py \
    --root /eos/cms/.../4237/1_e.root \
    --mapping "../trc_out_sync/mapping_results_*.csv" \
    --peaks-dir ../trc_out_MCP_reco \
    --peaks-pattern "peaks_raw_C1_*_{suffix}_data.csv" \
    --channel 192 \
    --out out_with_peaks_all_segments.root

Notes:
- The mapping CSV (mapping_results_segN.csv) must contain a row with key 'trigger_to_root' whose
  value is a JSON list (may contain nulls). 'trigger_to_root'[j] = i means trigger-index j maps to
  root-cluster index i (index into the cluster entries used when producing the mapping).
- The peaks CSV files are searched by pattern: the script substitutes {suffix} with the numeric
  suffix extracted from the mapping filename if possible. If no match is found the script will
  try to find any peaks CSV in the provided peaks-dir and match by order.
- The script writes a new ROOT file with the same branches as the input tree, except the
  chosen channel/time/energy branches are extended by the new peak entries for matched events.

This is a best-effort utility; verify the output before using in production.

Author: Licheng Zhang (licheng.zhang@cern.ch)
Date: 2026-01
"""

import argparse
import ast
import csv
import json
import os
import re
from pathlib import Path

try:
    import uproot
    import awkward as ak
    import numpy as np
    import pandas as pd
except Exception as e:
    print('Missing dependency:', e)
    print('Install: pip install uproot awkward numpy pandas')
    raise


def parse_list_field(s):
    if pd.isna(s):
        return []
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def find_peaks_file_for_suffix(peaks_dir, pattern_template, suffix):
    # pattern_template should include '{suffix}' placeholder
    patt = pattern_template.format(suffix=suffix)
    matches = sorted(Path(peaks_dir).parent.glob(os.path.basename(patt))) if os.path.dirname(patt) else sorted(Path(peaks_dir).glob(patt))
    # allow absolute path in pattern
    if not matches:
        matches = sorted(Path(peaks_dir).glob(patt)) if Path(peaks_dir).is_dir() else []
    return matches[0] if matches else None


def infer_peak_columns(df_peaks):
    # Try to guess columns for time and amplitude
    cols = [c.lower() for c in df_peaks.columns]
    time_col = None
    amp_col = None
    for c in df_peaks.columns:
        lc = c.lower()
        if time_col is None and ('time' in lc or 'peak_time' in lc or 't_' in lc):
            time_col = c
        if amp_col is None and ('amp' in lc or 'amplitude' in lc or 'height' in lc or 'peak' in lc):
            amp_col = c
    # fallback to first two columns
    if time_col is None and len(df_peaks.columns) >= 1:
        time_col = df_peaks.columns[0]
    if amp_col is None and len(df_peaks.columns) >= 2:
        amp_col = df_peaks.columns[1]
    return time_col, amp_col


def expand_mapping_paths(mapping_args):
    paths = []
    for item in mapping_args:
        if any(ch in item for ch in ['*', '?', '[']):
            paths.extend(sorted(Path().glob(item)))
        else:
            paths.append(Path(item))
    # de-dup while preserving order
    seen = set()
    ordered = []
    for p in paths:
        s = str(p)
        if s not in seen:
            ordered.append(p)
            seen.add(s)
    return ordered


def load_mapped_root_idx(mapping_csv_path):
    return load_mapping_value(mapping_csv_path, "mapped_root_idx") or []


def load_mapping_value(mapping_csv_path, key_name):
    with open(mapping_csv_path, "r", newline='') as fh:
        reader = csv.reader(fh)
        for parts in reader:
            if not parts or len(parts) < 2:
                continue
            key = parts[0].strip()
            if key != key_name:
                continue
            value_json = parts[1].strip()
            try:
                value = json.loads(value_json)
            except Exception:
                value = None
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    try:
                        return ast.literal_eval(value)
                    except Exception:
                        return None
            return value
    return None


def find_peaks_file_for_mapping(peaks_dir, pattern_template, mapping_path):
    base = Path(mapping_path).name
    m = re.search(r'(\d{7}_\d{7}_\d{4})', base)
    token = m.group(1) if m else None
    candidates = sorted(Path(peaks_dir).glob(pattern_template))
    if token:
        for c in candidates:
            if token in c.name:
                return c
    if len(candidates) == 1:
        return candidates[0]
    return None


def main():
    p = argparse.ArgumentParser(description='Apply mapping and insert peak time/amp into ROOT tree as channel entries')
    p.add_argument('--root', required=True, help='Input ROOT file path')
    p.add_argument('--mapping', required=True, nargs='+', help='Mapping CSV(s) produced by Febd_synchronizor (mapping_results_{suffix}.csv). Accepts a glob or multiple files')
    p.add_argument('--peaks-dir', required=True, help='Directory containing peaks CSV files')
    p.add_argument('--peaks-pattern', default='peaks_raw_C1_*_{suffix}_data.csv', help='Pattern to find peaks CSV; use {suffix} placeholder that will be replaced by numeric suffix from mapping filename')
    p.add_argument('--channel', type=int, default=192, help='Channel ID to add peaks to')
    p.add_argument('--out', required=True, help='Output ROOT file path')
    p.add_argument('--branch-channel', default='channelID', help='Branch name for channel list in ROOT tree')
    p.add_argument('--branch-time', default='time', help='Branch name for time list in ROOT tree')
    p.add_argument('--branch-energy', default='energy', help='Branch name for energy list in ROOT tree')
    args = p.parse_args()

    root_path = args.root
    mapping_path = args.mapping
    peaks_dir = args.peaks_dir
    pattern_template = args.peaks_pattern
    out_root = args.out

    # Step 1: copy input ROOT to output unchanged (ignore mapping/peaks for now)
    print('Step 1: copying input ROOT to output (no mapping applied)')
    # Simple byte-copy avoids uproot limitations when tree contains nested/jagged records.
    import shutil
    out_path = out_root
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(root_path, out_path)
    print('Copied input ROOT to', out_path, '(bytewise, unchanged)')

    # Step 2/3: add MCP tree with mapped_root_idx and peaks as branches
    print('Step 2/3: adding MCP tree with mapped_root_idx + peak_time/peak_amp')
    mapping_paths = expand_mapping_paths(mapping_path)
    if not mapping_paths:
        raise FileNotFoundError('No mapping CSVs matched the provided --mapping arguments')
    all_indices = []
    all_peak_time = []
    all_peak_amp = []
    all_peak_phi = []
    all_trigger_time = []
    for mp in mapping_paths:
        idx_list = load_mapped_root_idx(mp) or []
        trigger_to_root = load_mapping_value(mp, "trigger_to_root") or []
        root_to_trigger = load_mapping_value(mp, "root_to_trigger") or []
        peaks_file = find_peaks_file_for_mapping(peaks_dir, pattern_template, mp)
        if peaks_file is None:
            raise FileNotFoundError(f'No peaks CSV matched for mapping {mp}')
        df_peaks = pd.read_csv(peaks_file)

        # Determine time column (require ps)
        if "peak_time_ps" in df_peaks.columns:
            time_col = "peak_time_ps"
            time_in_ps = True
        else:
            raise KeyError(f'Missing peak_time_ps column in peaks file: {peaks_file}')

        # t0 may be present (added by combine_mcp_clock); prefer ps
        if "t0_abs_ps" in df_peaks.columns:
            t0_col = "t0_abs_ps"
            t0_in_ps = True
        elif "t0_abs_ns" in df_peaks.columns:
            t0_col = "t0_abs_ns"
            t0_in_ps = False
        else:
            # t0 missing — we'll compute phi as NaN later
            t0_col = None
            t0_in_ps = False

        # trigger time (ps) from combine_mcp_clock
        if "trigger_time_ps" in df_peaks.columns:
            trig_col = "trigger_time_ps"
            trig_in_ps = True
        elif "trigger_time_s" in df_peaks.columns:
            trig_col = "trigger_time_s"
            trig_in_ps = False
        else:
            trig_col = None
            trig_in_ps = False

        if "segment" not in df_peaks.columns:
            raise KeyError(f'Missing "segment" column in peaks file: {peaks_file}')
        if "peak_amp" not in df_peaks.columns:
            raise KeyError(f'Missing peak_amp column in peaks file: {peaks_file}')

        # Build mapping: segment -> (peak_time_ps, peak_amp, t0_abs_ps, trigger_time_ps)
        seg_to_peak = {}
        for _, row in df_peaks.iterrows():
            try:
                seg = int(row["segment"])
            except Exception:
                continue
            # peak time in ps
            pt = row.get(time_col, np.nan)
            if not time_in_ps and not pd.isna(pt):
                pt = float(pt) * 1000.0
            # amplitude
            pa = row.get("peak_amp", np.nan)
            # t0 in ps if present
            if t0_col is not None:
                t0v = row.get(t0_col, np.nan)
                if not t0_in_ps and not pd.isna(t0v):
                    t0v = float(t0v) * 1000.0
            else:
                t0v = np.nan
            # trigger_time in ps if present
            if trig_col is not None:
                trv = row.get(trig_col, np.nan)
                if not trig_in_ps and not pd.isna(trv):
                    trv = float(trv) * 1e12
            else:
                trv = np.nan
            seg_to_peak[int(seg)] = (
                pt if not pd.isna(pt) else np.nan,
                pa if not pd.isna(pa) else np.nan,
                t0v if not pd.isna(t0v) else np.nan,
                trv if not pd.isna(trv) else np.nan,
            )

        # Append in trigger order (same order used to build mapped_root_idx)
        local_peak_time = []
        local_peak_amp = []
        local_peak_phi = []
        local_trigger_time = []
        for j_trigger, i_root in enumerate(trigger_to_root):
            if i_root is None or (isinstance(i_root, float) and np.isnan(i_root)):
                continue
            i_root_int = int(i_root)
            if 0 <= i_root_int < len(root_to_trigger):
                trig_idx = root_to_trigger[i_root_int]
            else:
                trig_idx = j_trigger
            if trig_idx is None or (isinstance(trig_idx, float) and np.isnan(trig_idx)):
                seg = j_trigger + 1
            else:
                seg = int(trig_idx) + 1
            peak = seg_to_peak.get(seg, (np.nan, np.nan, np.nan, np.nan))
            peak_time_ps, peak_amp, t0_abs_ps, trigger_time_ps = peak
            # compute phi: (peak_time_ps - t0_abs_ps) mod 6250 (ps)
            if np.isnan(peak_time_ps) or np.isnan(t0_abs_ps):
                phi = np.nan
            else:
                try:
                    phi = float((peak_time_ps - t0_abs_ps) % 6250.0)
                except Exception:
                    phi = np.nan
            local_peak_time.append(peak_time_ps)
            local_peak_amp.append(peak_amp)
            local_peak_phi.append(phi)
            local_trigger_time.append(trigger_time_ps)

        if len(idx_list) != len(local_peak_time):
            print(f'Warning: mapped_root_idx length {len(idx_list)} != peaks length {len(local_peak_time)} for {mp}')

        all_indices.extend(idx_list)
        all_peak_time.extend(local_peak_time)
        all_peak_amp.extend(local_peak_amp)
        all_peak_phi.extend(local_peak_phi)
        all_trigger_time.extend(local_trigger_time)

    if len(all_indices) == 0:
        print('Warning: mapped_root_idx is empty; MCP/index will be an empty branch')
    with uproot.update(out_path) as f:
        f["MCP"] = {
            "index": np.array(all_indices, dtype=np.int64),
            # write peak_time in picoseconds (ps)
            "peak_time": np.array(all_peak_time, dtype=np.float64),
            "peak_amp": np.array(all_peak_amp, dtype=np.float64),
            "phi_peak": np.array(all_peak_phi, dtype=np.float64),
            "trigger_time": np.array(all_trigger_time, dtype=np.float64),
        }
    print('Added MCP tree with', len(all_indices), 'entries')


if __name__ == '__main__':
    main()
