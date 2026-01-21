#!/usr/bin/env python3
"""
Simple ROOT 'data' tree explorer and quick HEP-style plots.

Usage:
    python3 read_root_explore.py path/to/file.root --channel 160 --outdir ./plots

Requirements:
    pip install uproot awkward numpy matplotlib

What the script does:
- Opens the ROOT file and finds a tree named 'data' (or lists available trees if not found).
- Prints basic tree info and branch names.
- Tries to infer branches for channel id, channel index, time and energy.
- For a requested channel ID (e.g. 160) it maps per-event channel entries to the corresponding time/energy using the channelIdx mapping and collects energy values.
- Produces an energy histogram saved to outdir (PNG).

Note: this script uses awkward arrays via uproot/awkward. It is defensive: it will try to pick reasonable branch names but prints what it finds so you can adapt branch names explicitly if needed.
"""

import argparse
import os
import sys
import math

try:
    import uproot
    import awkward as ak
    import numpy as np
    import matplotlib.pyplot as plt
    import concurrent.futures
    import csv
    import glob
    import re
    from concurrent.futures import ThreadPoolExecutor
except Exception as e:
    print("Missing Python dependency:", e)
    print("Install required packages: pip install uproot awkward numpy matplotlib")
    sys.exit(1)


def pick_branch(candidates, keys):
    # return first branch name from keys that contains any candidate substring
    for cand in candidates:
        for k in keys:
            if cand in k.lower():
                return k
    return None

def process_event(i):
    try:
        ev_ch = ak.to_list(arrays[ch_branch][i]) if ch_branch in keys else []
        ev_idx = ak.to_list(arrays[idx_branch][i]) if idx_branch in keys else []
        ev_time = ak.to_list(arrays[time_branch][i]) if time_branch in keys else []
        ev_energy = ak.to_list(arrays[energy_branch][i]) if energy_branch in keys else []

        if required_chs and not required_chs.issubset(set(ev_ch)):
            return None

        return [i, json.dumps(ev_ch), json.dumps(ev_idx), json.dumps(ev_time), json.dumps(ev_energy)]
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(description='Explore ROOT tree "data" and dump branches to CSV/JSON')
    p.add_argument('file', help='ROOT file path')
    p.add_argument('--branch-channel', help='Explicit branch name for channelID entries (e.g. channelID)')
    p.add_argument('--branch-idx', help='Explicit branch name for channel index mapping (e.g. channelIdx)')
    p.add_argument('--branch-time', help='Explicit branch name for time array (e.g. time)')
    p.add_argument('--branch-energy', help='Explicit branch name for energy array (e.g. energy)')
    p.add_argument('--dump-csv', help='If provided, dump per-event/channel entries to this CSV file (no mapping).')
    p.add_argument('--dump-mapped-csv', help='If provided, dump per-channel mapped rows to this CSV file (using channelIdx[channelID] mapping).')
    p.add_argument('--dump-simple-csv', help='If provided, dump mapped rows as simple CSV: entry,channelID,time,energy')
    p.add_argument('--dump-json', help='If provided, dump per-event nested JSON lines to this file.')
    p.add_argument('--dump-df', help='If provided, convert tree (or first --max-entries) to a pandas DataFrame and save as pickle.')
    p.add_argument('--require-channels', nargs='+', type=int, help='List of channelID values that must be present in an event to include it (e.g. --require-channels 192 156)')
    p.add_argument('--max-entries', type=int, default=1000, help='Max number of events to dump')
    p.add_argument('--channel', type=int, help='Channel ID to plot time distribution for (uses channelIdx mapping)')
    p.add_argument('--outdir', help='Output directory to save plots (default: current directory)')
    p.add_argument('--workers', type=int, default=1, help='Number of worker threads for per-event processing (default: 1)')
    p.add_argument('--meta-dir', help='Directory containing meta CSV files for correlation (e.g. raw_C1_*_meta.csv)')
    p.add_argument('--verbose', action='store_true', help='Enable verbose debug prints')
    args = p.parse_args()

    if not os.path.exists(args.file):
        print('File not found:', args.file)
        sys.exit(2)

    f = uproot.open(args.file)
    # find tree named 'data' (prefer the one with largest cycle, e.g. 'data;4')
    all_keys = list(f.keys())
    if getattr(args, 'verbose', False):
        print('Opened ROOT file:', args.file)
        print('Top-level keys count:', len(all_keys))
    data_keys = [k for k in all_keys if k.startswith('data')]
    tree_name = None
    if data_keys:
        import re
        best = None
        best_cycle = -1
        for k in data_keys:
            m = re.match(r'data(?:;(\d+))?$', k)
            cycle = int(m.group(1)) if m and m.group(1) else 0
            if cycle > best_cycle:
                best_cycle = cycle
                best = k
        tree_name = best
    else:
        # fallback: pick any TTree-like object
        tnames = [k for k, v in f.items() if hasattr(v, 'num_entries')]
        print('No top-level "data" tree found. Available TTrees/objects:')
        for k in tnames:
            print('  ', k)
        if not tnames:
            print('No TTrees found in file. Exiting.')
            sys.exit(3)
        tree_name = tnames[0]
        print('Using first available tree:', tree_name)

    tree = f[tree_name]
    n_entries = tree.num_entries
    print(f'Tree "{tree_name}" has {n_entries} entries')

    # list branches
    keys = list(tree.keys())
    print('\nBranches found (sample):')
    for k in keys[:200]:
        print('  ', k)

    arrays = tree.arrays(library='ak')
    if getattr(args, 'verbose', False):
        print('Loaded arrays from tree into memory (awkward)')
        # use previously computed `keys` from the tree for branch count
        print('Number of branches loaded:', len(keys))

    # helper to extract branches for a single event index as plain python lists
    def get_event(idx):
        try:
            ev_ch = ak.to_list(arrays[ch_branch][idx]) if (ch_branch and ch_branch in keys) else []
        except Exception:
            ev_ch = []
        try:
            ev_idx = ak.to_list(arrays[idx_branch][idx]) if (idx_branch and idx_branch in keys) else []
        except Exception:
            ev_idx = []
        try:
            ev_time = ak.to_list(arrays[time_branch][idx]) if (time_branch and time_branch in keys) else []
        except Exception:
            ev_time = []
        try:
            ev_energy = ak.to_list(arrays[energy_branch][idx]) if (energy_branch and energy_branch in keys) else []
        except Exception:
            ev_energy = []
        if getattr(args, 'verbose', False) and idx < 3:
            # print a small sample for the first few events
            print(f'[debug] get_event idx={idx} len(ch)={len(ev_ch)} len(idx)={len(ev_idx)} len(time)={len(ev_time)}')
        return ev_ch, ev_idx, ev_time, ev_energy

    # Optional event filter: require certain channelIDs present in event channel list
    required_chs = set(args.require_channels) if getattr(args, 'require_channels', None) else None

    # Optional quick CSV dump: write per-event channelID lists so you can inspect immediately (no pandas/pickle)
    dump_df_arg = getattr(args, 'dump_df', None)
    if dump_df_arg:
        import csv, json
        from concurrent.futures import ThreadPoolExecutor

        max_e = min(args.max_entries if args.max_entries is not None else n_entries, n_entries)
        out_csv = dump_df_arg
        print(f"Dumping up to {max_e} events' channelID lists to CSV (parallelized): {out_csv}")

        # prefer exact branch names unless user overrides
        local_ch_branch = args.branch_channel if args.branch_channel else ('channelID' if 'channelID' in keys else None)
        local_idx_branch = args.branch_idx if args.branch_idx else ('channelIdx' if 'channelIdx' in keys else None)
        local_time_branch = args.branch_time if args.branch_time else ('time' if 'time' in keys else None)
        local_energy_branch = args.branch_energy if args.branch_energy else ('energy' if 'energy' in keys else None)

        def process_event(i):
            try:
                ev_ch = ak.to_list(arrays[local_ch_branch][i]) if local_ch_branch in keys else []
            except Exception:
                ev_ch = []
            try:
                ev_idx = ak.to_list(arrays[local_idx_branch][i]) if local_idx_branch in keys else []
            except Exception:
                ev_idx = []
            try:
                ev_time = ak.to_list(arrays[local_time_branch][i]) if local_time_branch in keys else []
            except Exception:
                ev_time = []
            try:
                ev_energy = ak.to_list(arrays[local_energy_branch][i]) if local_energy_branch in keys else []
            except Exception:
                ev_energy = []

            if required_chs:
                try:
                    if not required_chs.issubset(set(ev_ch)):
                        return None
                except Exception:
                    return None

            return [i, json.dumps(ev_ch), json.dumps(ev_idx), json.dumps(ev_time), json.dumps(ev_energy)]

        with open(out_csv, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['entry', 'channelID', 'channelIdx', 'time', 'energy'])

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(process_event, range(max_e)):
                    if result:
                        writer.writerow(result)

        print('ChannelID CSV dump completed (with multithreading).')


    # Precompute branch name choices so --dump-csv can use them before further processing
    # Use explicit --branch-channel if provided, otherwise prefer exact 'channelID' (case-insensitive) if present
    if args.branch_channel:
        ch_branch = args.branch_channel
    else:
        ch_branch = None
        for k in keys:
            if k.lower() == 'channelid'.lower():
                ch_branch = k
                break
    # Use explicit overrides if provided, otherwise use exact branch names expected in these files
    idx_branch = args.branch_idx if args.branch_idx else ('channelIdx' if 'channelIdx' in keys else None)
    time_branch = args.branch_time if args.branch_time else ('time' if 'time' in keys else None)
    energy_branch = args.branch_energy if args.branch_energy else ('energy' if 'energy' in keys else None)

    # Do not attempt to search/guess other channel branch names; user treats channelID as the real branch.
    # If ch_branch equals idx_branch that's likely a filename-specific issue; pass through and let user override with --branch-channel.

    # New dump options: raw CSV (existing), mapped CSV (uses channelIdx[channelID] mapping), and JSON lines
    # --dump-csv: raw per-entry alignment (unchanged)
    # --dump-mapped-csv: per-channel mapped rows using channelIdx[channelID]
    # --dump-json: one JSON object per event with nested arrays and mapped results
    
    # Raw CSV dump (unchanged behavior)
    if args.dump_csv:
        import csv
        max_e = args.max_entries if args.max_entries is not None else n_entries
        out_csv = args.dump_csv
        print(f"Dumping up to {max_e} events to CSV: {out_csv} (raw per-entry alignment)")
        with open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['entry', 'chan_entry_idx', 'channelID', 'channelIdx_at_j', 'time_at_j', 'energy_at_j', 'len_channel', 'len_channelIdx', 'len_time', 'len_energy'])
            for i in range(min(n_entries, max_e)):
                ev_ch = ak.to_list(arrays[ch_branch][i]) if (ch_branch and ch_branch in keys) else []
                ev_idx = ak.to_list(arrays[idx_branch][i]) if (idx_branch and idx_branch in keys) else []
                ev_time = ak.to_list(arrays[time_branch][i]) if (time_branch and time_branch in keys) else []
                ev_energy = ak.to_list(arrays[energy_branch][i]) if (energy_branch and energy_branch in keys) else []

                l_ch = len(ev_ch) if ev_ch is not None else 0
                l_idx = len(ev_idx) if ev_idx is not None else 0
                l_time = len(ev_time) if ev_time is not None else 0
                l_energy = len(ev_energy) if ev_energy is not None else 0

                # iterate over channel-entry slots and dump aligned values (may be missing)
                for j in range(max(l_ch, l_time, l_energy)):
                    ch_val = ev_ch[j] if j < l_ch else ''
                    idx_val = ev_idx[j] if j < l_idx else ''
                    t_val = ev_time[j] if j < l_time else ''
                    e_val = ev_energy[j] if j < l_energy else ''
                    writer.writerow([i, j, ch_val, idx_val, t_val, e_val, l_ch, l_idx, l_time, l_energy])

        print('Raw CSV dump completed.')
        # continue, do not return — allow other dump modes if specified

    # Mapped CSV dump: use channelIdx[channelID] to map
    mapped_csv_arg = getattr(args, 'dump_mapped_csv', None)
    if mapped_csv_arg:
        import csv
        max_e = args.max_entries if args.max_entries is not None else n_entries
        out_csv = mapped_csv_arg
        print(f"Dumping up to {max_e} events to mapped CSV: {out_csv} (using channelIdx[channelID] mapping)")
        with open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['entry', 'chan_entry_idx', 'channelID', 'mapped_index', 'time_mapped', 'energy_mapped', 'len_channel', 'len_channelIdx', 'len_time', 'len_energy'])
            for i in range(min(n_entries, max_e)):
                ev_ch = ak.to_list(arrays[ch_branch][i]) if (ch_branch and ch_branch in keys) else []
                ev_idx = ak.to_list(arrays[idx_branch][i]) if (idx_branch and idx_branch in keys) else []
                ev_time = ak.to_list(arrays[time_branch][i]) if (time_branch and time_branch in keys) else []
                ev_energy = ak.to_list(arrays[energy_branch][i]) if (energy_branch and energy_branch in keys) else []

                l_ch = len(ev_ch) if ev_ch is not None else 0
                l_idx = len(ev_idx) if ev_idx is not None else 0
                l_time = len(ev_time) if ev_time is not None else 0
                l_energy = len(ev_energy) if ev_energy is not None else 0

                for j, ch in enumerate(ev_ch):
                    mapped_idx = None
                    t_m = ''
                    e_m = ''
                    try:
                        # channelIdx is an array indexed by channelID
                        if ev_idx is not None and 0 <= ch < l_idx:
                            mapped_idx = ev_idx[ch]
                        else:
                            mapped_idx = -1
                        if mapped_idx is None or mapped_idx < 0:
                            # no mapping
                            mapped_idx = -1
                        else:
                            if 0 <= mapped_idx < l_time:
                                t_m = ev_time[mapped_idx]
                            if 0 <= mapped_idx < l_energy:
                                e_m = ev_energy[mapped_idx]
                    except Exception:
                        mapped_idx = -1
                    writer.writerow([i, j, ch, mapped_idx, t_m, e_m, l_ch, l_idx, l_time, l_energy])
        print('Mapped CSV dump completed.')
        # continue

    # JSON nested dump: one JSON object per event with arrays and mapped info
    dump_json_arg = getattr(args, 'dump_json', None)
    if dump_json_arg:
        import json
        max_e = args.max_entries if args.max_entries is not None else n_entries
        out_json = dump_json_arg
        print(f"Dumping up to {max_e} events to JSON lines: {out_json}")
        with open(out_json, 'w') as jf:
            for i in range(min(n_entries, max_e)):
                ev_ch = ak.to_list(arrays[ch_branch][i]) if (ch_branch and ch_branch in keys) else []
                ev_idx = ak.to_list(arrays[idx_branch][i]) if (idx_branch and idx_branch in keys) else []
                ev_time = ak.to_list(arrays[time_branch][i]) if (time_branch and time_branch in keys) else []
                ev_energy = ak.to_list(arrays[energy_branch][i]) if (energy_branch and energy_branch in keys) else []

                l_ch = len(ev_ch) if ev_ch is not None else 0
                l_idx = len(ev_idx) if ev_idx is not None else 0
                l_time = len(ev_time) if ev_time is not None else 0
                l_energy = len(ev_energy) if ev_energy is not None else 0

                mapped = []
                for j, ch in enumerate(ev_ch):
                    try:
                        if ev_idx is not None and 0 <= ch < l_idx:
                            mapped_idx = ev_idx[ch]
                        else:
                            mapped_idx = -1
                        if mapped_idx is None or mapped_idx < 0:
                            mapped.append({'chan_entry_idx': j, 'channelID': ch, 'mapped_index': -1})
                        else:
                            t_m = ev_time[mapped_idx] if 0 <= mapped_idx < l_time else None
                            e_m = ev_energy[mapped_idx] if 0 <= mapped_idx < l_energy else None
                            mapped.append({'chan_entry_idx': j, 'channelID': ch, 'mapped_index': int(mapped_idx), 'time': t_m, 'energy': e_m})
                    except Exception:
                        mapped.append({'chan_entry_idx': j, 'channelID': ch, 'mapped_index': -1})

                obj = {
                    'entry': int(i),
                    'channel': ev_ch,
                    'channelIdx': ev_idx,
                    'time': ev_time,
                    'energy': ev_energy,
                    'mapped': mapped,
                    'lens': {'len_channel': l_ch, 'len_channelIdx': l_idx, 'len_time': l_time, 'len_energy': l_energy}
                }
                jf.write(json.dumps(obj) + '\n')
        print('JSON dump completed.')
        # continue

    # Simple CSV dump: mapped rows (entry,channelID,time,energy)
    simple_csv_arg = getattr(args, 'dump_simple_csv', None)
    if simple_csv_arg:
        import csv
        max_e = args.max_entries if args.max_entries is not None else n_entries
        out_csv = simple_csv_arg
        print(f"Dumping up to {max_e} events to simple CSV: {out_csv} (entry,channelID,time,energy)")
        with open(out_csv, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(['entry', 'channelID', 'time', 'energy'])
            for i in range(min(n_entries, max_e)):
                ev_ch = ak.to_list(arrays[ch_branch][i]) if (ch_branch and ch_branch in keys) else []
                ev_idx = ak.to_list(arrays[idx_branch][i]) if (idx_branch and idx_branch in keys) else []
                ev_time = ak.to_list(arrays[time_branch][i]) if (time_branch and time_branch in keys) else []
                ev_energy = ak.to_list(arrays[energy_branch][i]) if (energy_branch and energy_branch in keys) else []

                l_ch = len(ev_ch) if ev_ch is not None else 0
                l_idx = len(ev_idx) if ev_idx is not None else 0
                l_time = len(ev_time) if ev_time is not None else 0
                l_energy = len(ev_energy) if ev_energy is not None else 0

                # for each channel entry, dump the mapped time/energy
                for j, ch in enumerate(ev_ch):
                    mapped_idx = None
                    t_m = ''
                    e_m = ''
                    try:
                        if ev_idx is not None and 0 <= ch < l_idx:
                            mapped_idx = ev_idx[ch]
                        else:
                            mapped_idx = -1
                        if mapped_idx is None or mapped_idx < 0:
                            # no mapping
                            mapped_idx = -1
                        else:
                            if 0 <= mapped_idx < l_time:
                                t_m = ev_time[mapped_idx]
                            if 0 <= mapped_idx < l_energy:
                                e_m = ev_energy[mapped_idx]
                    except Exception:
                        mapped_idx = -1
                    writer.writerow([i, ch, t_m, e_m])
        print('Simple CSV dump completed.')

    # Plot requested channel time distribution (uses channelIdx mapping)
    if getattr(args, 'channel', None) is not None:
        target_chan = args.channel
        # require time, idx and channel branches to be present
        if time_branch is None or idx_branch is None or ch_branch is None:
            print('Cannot plot: missing inferred branch(s) (time/channel/channelIdx).')
        else:
            times = []
            max_e = args.max_entries if args.max_entries is not None else n_entries
            max_e = min(n_entries, max_e)

            def process_idx(i):
                try:
                    ev_ch, ev_idx, ev_time, ev_energy = get_event(i)
                    # apply required channels filter if requested
                    if required_chs:
                        try:
                            if not required_chs.issubset(set(ev_ch)):
                                return None
                        except Exception:
                            return None
                    l_idx = len(ev_idx) if ev_idx is not None else 0
                    if ev_idx is not None and 0 <= target_chan < l_idx:
                        mapped_idx = ev_idx[target_chan]
                    else:
                        mapped_idx = -1
                    if mapped_idx is None or mapped_idx < 0:
                        return None
                    if 0 <= mapped_idx < len(ev_time):
                        return ev_time[mapped_idx]
                except Exception:
                    return None
                return None

            indices = list(range(max_e))
            if getattr(args, 'workers', 1) and args.workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for res in ex.map(process_idx, indices):
                        if res is not None:
                            times.append(res)
            else:
                for i in indices:
                    res = process_idx(i)
                    if res is not None:
                        times.append(res)

            if times:
                outdir = args.outdir if getattr(args, 'outdir', None) else '.'
                os.makedirs(outdir, exist_ok=True)
                plt.figure(figsize=(6, 4))
                plt.hist(times, bins=100, histtype='stepfilled', alpha=0.7)
                plt.xlabel('time')
                plt.ylabel('counts')
                plt.title(f'Channel {target_chan} time distribution (n={len(times)})')
                outpng = os.path.join(outdir, f'channel_{target_chan}_time_hist.png')
                plt.tight_layout()
                plt.savefig(outpng)
                plt.close()
                print('Saved time histogram for channel', target_chan, '->', outpng)
            else:
                print('No mapped times found for channel', target_chan)

    # Scatter correlation: match ROOT event times with trigger times from meta CSVs
    meta_dir = getattr(args, 'meta_dir', None)
    if meta_dir and getattr(args, 'channel', None) is not None:
        if not os.path.exists(meta_dir):
            print(f'Meta directory not found: {meta_dir}')
        else:
            # Extract spill number from ROOT filename (e.g. 1_e.root -> spill 1)
            root_basename = os.path.basename(args.file)
            spill_match = re.match(r'(\d+)_e\.root', root_basename)
            if not spill_match:
                print(f'Could not extract spill number from {root_basename}')
            else:
                spill_num = spill_match.group(1)
                spill_padded = spill_num.zfill(7)  # e.g. '0000001'
                target_chan = args.channel
                
                # Find meta CSVs for C1 channel matching this spill
                meta_pattern = os.path.join(meta_dir, f'raw_C1_*_{spill_padded}_*_meta.csv')
                meta_files = sorted(glob.glob(meta_pattern))
                
                if not meta_files:
                    print(f'No meta CSVs found matching pattern: {meta_pattern}')
                else:
                    print(f'Found {len(meta_files)} meta CSV files for spill {spill_num}')
                    
                    # Load trigger_time from each meta CSV
                    meta_data = {}  # {file_suffix: list of trigger_times}
                    for mf in meta_files:
                        try:
                            # Extract file suffix like 13597/13598/13599
                            fname_parts = os.path.basename(mf).replace('_meta.csv', '').split('_')
                            file_suffix = fname_parts[-1]  # e.g. '13597'
                            
                            trigger_times = []
                            with open(mf, 'r') as fh:
                                # Try DictReader first (headered CSV where trigger_time is a column)
                                fh.seek(0)
                                dr = csv.DictReader(fh)
                                fnames = dr.fieldnames
                                if fnames and any(('trigger_time' == (fn or '').strip().lower()) for fn in fnames):
                                    # header contains a trigger_time column
                                    for row in dr:
                                        # find the key exactly (case-insensitive)
                                        v = None
                                        for k in row.keys():
                                            if k and k.strip().lower() == 'trigger_time':
                                                v = row[k]
                                                break
                                        if v is None:
                                            continue
                                        s = str(v)
                                        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", s)
                                        for token in nums:
                                            try:
                                                trigger_times.append(float(token))
                                            except Exception:
                                                continue
                                else:
                                    # Likely a key/value file (rows of Field,Value). Rewind and parse as rows.
                                    fh.seek(0)
                                    rdr = csv.reader(fh)
                                    header = next(rdr, None)
                                    idx_field = None
                                    idx_value = None
                                    if header:
                                        for i, col in enumerate(header):
                                            if col and col.strip().lower() in ('field', 'name', 'key'):
                                                idx_field = i
                                            if col and col.strip().lower() in ('value', 'val', 'data'):
                                                idx_value = i
                                    # If we could not detect indices, fallback to scanning rows and matching first col == 'trigger_time'
                                    if idx_field is None or idx_value is None:
                                        fh.seek(0)
                                        for row in csv.reader(fh):
                                            if len(row) >= 2 and row[0].strip().lower() == 'trigger_time':
                                                s = row[1]
                                                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(s))
                                                for token in nums:
                                                    try:
                                                        trigger_times.append(float(token))
                                                    except Exception:
                                                        continue
                                                break
                                    else:
                                        for row in rdr:
                                            if len(row) <= max(idx_field, idx_value):
                                                continue
                                            if row[idx_field].strip().lower() == 'trigger_time':
                                                s = row[idx_value]
                                                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(s))
                                                for token in nums:
                                                    try:
                                                        trigger_times.append(float(token))
                                                    except Exception:
                                                        continue
                                                break
                            if getattr(args, 'verbose', False):
                                print(f'    Parsed {len(trigger_times)} trigger_time values from {os.path.basename(mf)}')
                            if trigger_times:
                                meta_data[file_suffix] = sorted(trigger_times)
                                print(f'  Loaded {len(trigger_times)} trigger times from {file_suffix}')
                        except Exception as e:
                            print(f'  Error loading {mf}: {e}')
                    
                    if not meta_data:
                        print('No valid trigger_time data found in meta CSVs')
                    else:
                        # Collect ROOT event times (same logic as histogram)
                        root_times = []
                        max_e = args.max_entries if args.max_entries is not None else n_entries
                        max_e = min(n_entries, max_e)
                        
                        for i in range(max_e):
                            ev_ch, ev_idx, ev_time, ev_energy = get_event(i)
                            if required_chs:
                                try:
                                    if not required_chs.issubset(set(ev_ch)):
                                        continue
                                except Exception:
                                    continue
                            l_idx = len(ev_idx) if ev_idx is not None else 0
                            if ev_idx is not None and 0 <= target_chan < l_idx:
                                mapped_idx = ev_idx[target_chan]
                            else:
                                continue
                            if mapped_idx is None or mapped_idx < 0 or not (0 <= mapped_idx < len(ev_time)):
                                continue
                            root_times.append(ev_time[mapped_idx])
                        
                        if root_times:
                            print(f'Collected {len(root_times)} ROOT event times for channel {target_chan}')
                            
                            # Create scatter plot: split ROOT event times into contiguous time-series chunks
                            # and correlate each chunk with the corresponding meta file (in sorted order).
                            outdir = args.outdir if getattr(args, 'outdir', None) else '.'
                            os.makedirs(outdir, exist_ok=True)

                            # Ensure deterministic ordering of meta files
                            sorted_meta = sorted(meta_data.items())
                            n_meta = len(sorted_meta)

                            # Split root_times (time-series) into n_meta contiguous chunks (preserves order)
                            try:
                                root_chunks = list(np.array_split(np.array(root_times), n_meta))
                            except Exception:
                                # fallback to simple Python split
                                chunk_size = max(1, len(root_times) // n_meta)
                                root_chunks = [root_times[i:i+chunk_size] for i in range(0, len(root_times), chunk_size)]
                                # pad/truncate to n_meta
                                if len(root_chunks) < n_meta:
                                    # append empty chunks
                                    for _ in range(n_meta - len(root_chunks)):
                                        root_chunks.append([])
                                elif len(root_chunks) > n_meta:
                                    root_chunks = root_chunks[:n_meta]

                            # Combined overlay plot: all meta-file correlations on one axes
                            fig, ax = plt.subplots(1, 1, figsize=(7, 5))

                            colors = plt.cm.tab10.colors
                            all_counts = []

                            for ax_idx, ((suffix, trigger_times), chunk) in enumerate(zip(sorted_meta, root_chunks)):
                                # Convert trigger times (seconds) -> picoseconds (ps)
                                trigger_times_ps = [float(t) * 1e12 for t in trigger_times]

                                matched_trigger_times = []
                                diffs = []

                                # For each ROOT time in this chunk, find nearest trigger_time from this meta file
                                for rt in chunk:
                                    if trigger_times_ps:
                                        closest_tt = min(trigger_times_ps, key=lambda t: abs(t - rt))
                                        diff = rt - closest_tt
                                        matched_trigger_times.append(closest_tt)
                                    else:
                                        diff = rt
                                        matched_trigger_times.append(0.0)
                                    diffs.append(diff)

                                # Defensive padding
                                if len(matched_trigger_times) < len(diffs):
                                    matched_trigger_times.extend([0.0] * (len(diffs) - len(matched_trigger_times)))

                                # Plot on same axes; use small marker to form vertical bands
                                if len(diffs) > 0:
                                    c = colors[ax_idx % len(colors)]
                                    ax.scatter(matched_trigger_times, diffs, alpha=0.6, s=8, color=c, label=f'{suffix} (n={len(diffs)})')
                                    all_counts.append((suffix, len(diffs)))

                            ax.set_xlabel('trigger_time (ps)')
                            ax.set_ylabel('ROOT time - trigger_time (ps)')
                            ax.set_title(f'Channel {target_chan} correlations (per-meta overlay)')
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize='small')

                            plt.tight_layout()
                            outpng = os.path.join(outdir, f'channel_{target_chan}_scatter_correlation_combined.png')
                            plt.savefig(outpng, dpi=150)
                            plt.close()
                            print(f'Saved combined scatter correlation plot -> {outpng}')
                        else:
                            print('No ROOT times found for scatter plot')

    # Final summary: if any dump option used the script already created files above.
    print('\nInferred branches:')
    print('  channel branch :', ch_branch)
    print('  channelIdx branch :', idx_branch)
    print('  time branch :', time_branch)
    print('  energy branch :', energy_branch)

    if not (args.dump_csv or mapped_csv_arg or getattr(args, 'dump_json', None) or getattr(args, 'dump_simple_csv', None)):
        print('\nNo dump option provided. Use --dump-csv, --dump-mapped-csv, --dump-simple-csv or --dump-json to export data.')
    else:
        print('\nDump completed. Inspect the generated files.')
    return


if __name__ == '__main__':
    main()
