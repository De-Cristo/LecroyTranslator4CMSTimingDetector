#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Bar-level timing analysis.

Step 1:  Compute  T_L − T_R  for a given lyso bar and plot the 1-D
         distribution with a Gaussian fit.

Usage examples
--------------
  # Single bar, module up, bar 6
  python3 bar_analysis.py "*.root" --module up --lyso-bar 6

  # With energy cuts
  python3 bar_analysis.py "*.root" --module up --lyso-bar 6 \
      --energy-min 50 --energy-max 500

  # Multiple files, 4 worker processes
  python3 bar_analysis.py run1.root run2.root --module up --lyso-bar 6 --workers 4
"""

import argparse
import sys
import os
import math
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

try:
    import uproot
    import awkward as ak
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import moyal
except ImportError as e:
    print("Missing Python dependency:", e)
    print("Install: pip install uproot awkward numpy matplotlib scipy")
    sys.exit(1)

from channel_mapping import (
    lyso_bar_to_channels_lr,
    UP_MODULE_BASE,
    DOWN_MODULE_BASE,
    TRIGGER_CHANNEL,
)


def log(msg):
    print(f"[bar] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def find_data_tree(f):
    """Find the main data TTree inside the ROOT file."""
    data_keys = [k for k in f.keys() if k.startswith("data")]
    if data_keys:
        best, best_cycle = None, -1
        for k in data_keys:
            cycle = int(k.split(";", 1)[1]) if ";" in k else 0
            if cycle > best_cycle:
                best_cycle = cycle
                best = k
        return best
    tnames = [k for k, v in f.items() if hasattr(v, "num_entries")]
    return tnames[0] if tnames else None


def gauss(x, a, mu, sigma):
    """Un-normalised Gaussian."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ──────────────────────────────────────────────────────────────────
# Per-file processing
# ──────────────────────────────────────────────────────────────────

def process_file(path, cfg):
    """Process one ROOT file.  Return dict of accumulated lists."""
    out = {
        "path": path,
        "t_diff": [],        # T_L − T_R  (ps)
        "phi_diff": [],      # phi_L − phi_R  (ps), from t1fine
        "phi_vs_mcp": [],    # (phi_L + phi_R)/2 − phi_peak  (ps)
        "phi_l_vs_mcp": [],  # phi_L - phi_peak (ps)
        "phi_r_vs_mcp": [],  # phi_R - phi_peak (ps)
        "energy_l_mcp": [],  # E_L (when mcp match exists)
        "energy_r_mcp": [],  # E_R (when mcp match exists)
        "energy_avg_mcp": [], # (E_L + E_R)/2 (when mcp match exists)
        "raw_time_diff": [],  # ((t_bar - t_192) - (peak_time - trigger_time))  [ps]
        "raw_phi_diff": [],   # (phi_bar - phi_192) - (phi_peak - phi_trigger)  [ps]
        "phi_diff_down": [],  # phi_L_down - phi_R_down  (ps)
        "raw_phi_diff_down": [],  # (phi_bar_down - phi_192) - (phi_peak - phi_trigger)  [ps]
        "energy": [],        # (E_L + E_R) / 2
        "energy_l": [],      # E_L
        "energy_r": [],      # E_R
        "t_avg_diff": [],    # (T_avg_up - T_avg_down)  (ps)
        "phi_avg_diff": [],  # (phi_avg_up - phi_avg_down) (ps)
        "energy_down": [],   # (E_L + E_R) / 2  for down bar
        "energy_l_down": [], # E_L down
        "energy_r_down": [], # E_R down
        "raw_time_diff_down": [],  # (t_bar_down - t_192) - (peak_time - trigger_time)
        "t_diff_down": [],   # T_L_down - T_R_down  (ps)
        "phi_vs_mcp_down": [],  # (phi_L_down + phi_R_down)/2 - phi_peak
        "phi_l_vs_mcp_down": [],
        "phi_r_vs_mcp_down": [],
        "energy_l_mcp_down": [],
        "energy_r_mcp_down": [],
        "energy_avg_mcp_down": [],
        "phi_avg_diff_sync_e_up": [],   # E_avg_up synced with phi_avg_diff 
        "phi_avg_diff_sync_e_down": [], # E_avg_down synced with phi_avg_diff
        "phi_l_up_minus_avg_down": [],  # (phi_L_up - phi_avg_down) (ps)
        "energy_l_for_cross": [],       # E_L_up synced with phi_l_up_minus_avg_down
        "t_avg_vs_mcp": [],  # (T_L + T_R)/2 - peak_time
        "t_avg_vs_mcp_down": [],  # (T_L_down + T_R_down)/2 - peak_time
        "t_avg": [],         # (T_L + T_R)/2
        "mcp_t": [],         # T_peak
        "t_avg_down": [],    # (T_L_down + T_R_down)/2
        "mcp_t_down": [],    # T_peak (down)
        "t_192": [],         # T_192
        "mcp_t_trig": [],    # T_trigger
        "raw_time_diff_up_down": [], # (t_bar_up - t_192_up) - (t_bar_down - t_192_down)
        "counters": {
            "total": 0,
            "missing_ch": 0,
            "missing_time": 0,
            "missing_bar": 0,
            "energy_cut": 0,
            "missing_mcp": 0,
            "kept": 0,
        },
    }

    f = uproot.open(path)
    tree_name = find_data_tree(f)
    if tree_name is None:
        log(f"{path}: no data tree found")
        return out
    tree = f[tree_name]

    branches = [cfg["branch_channel"], cfg["branch_time"]]
    if cfg["branch_energy"]:
        branches.append(cfg["branch_energy"])
    if cfg["branch_t1coarse"]:
        branches.append(cfg["branch_t1coarse"])
    arrays = tree.arrays(branches, library="ak")

    # ---- build MCP index map (event_number → peak_time) ----
    mcp_events = set()      # event_number
    mcp_map = {}            # event_number → phi_peak (ps)
    mcp_map_phi_trig = {}   # event_number → phi_trigger (ps)
    mcp_map_peak_time = {}  # event_number → peak_time (ps)
    mcp_map_trig_time = {}  # event_number → trigger_time (ps)
    use_mcp = cfg.get("use_mcp", False)
    if use_mcp:
        mcp_tree_name = cfg["mcp_tree"]
        if mcp_tree_name in f:
            mcp_tree = f[mcp_tree_name]
            mcp_idx = mcp_tree[cfg["mcp_index"]].array(library="np")
            mcp_pt = mcp_tree[cfg["mcp_phi_peak"]].array(library="np")
            mcp_phi_trig = mcp_tree[cfg["mcp_phi_trigger"]].array(library="np")
            mcp_peak_time = mcp_tree[cfg["mcp_peak_time"]].array(library="np")
            mcp_trigger_time = mcp_tree[cfg["mcp_trigger_time"]].array(library="np")
            if cfg["mcp_peak_amp"]:
                try:
                    mcp_peak_amp = mcp_tree[cfg["mcp_peak_amp"]].array(library="np")
                except Exception:
                    mcp_peak_amp = np.full(len(mcp_idx), np.nan)
            else:
                mcp_peak_amp = np.full(len(mcp_idx), np.nan)
            for j in range(len(mcp_idx)):
                try:
                    evt = int(mcp_idx[j])
                    pt = float(mcp_pt[j])
                    ptrig = float(mcp_phi_trig[j])
                    peak_time = float(mcp_peak_time[j])
                    trig_time = float(mcp_trigger_time[j])
                    amp = float(mcp_peak_amp[j])
                except Exception:
                    continue

                if cfg["mcp_peak_amp_min"] is not None and (amp != amp or amp < cfg["mcp_peak_amp_min"]):
                    continue
                if cfg["mcp_peak_amp_max"] is not None and (amp != amp or amp > cfg["mcp_peak_amp_max"]):
                    continue

                mcp_events.add(evt)
                if pt == pt:  # not NaN
                    mcp_map[evt] = pt
                if ptrig == ptrig:  # not NaN
                    mcp_map_phi_trig[evt] = ptrig
                if peak_time == peak_time:  # not NaN
                    mcp_map_peak_time[evt] = peak_time
                if trig_time == trig_time:  # not NaN
                    mcp_map_trig_time[evt] = trig_time
            log(
                f"{path}: MCP map built, index={len(mcp_events)}, "
                f"phi_peak={len(mcp_map)}, "
                f"phi_trigger={len(mcp_map_phi_trig)}, "
                f"peak_time={len(mcp_map_peak_time)}, trigger_time={len(mcp_map_trig_time)}"
            )
        else:
            log(f"{path}: MCP tree '{mcp_tree_name}' not found")

    n = tree.num_entries
    max_e = n if cfg["max_entries"] is None else min(cfg["max_entries"], n)
    log(f"{path}: data tree '{tree_name}', entries={n}, processing={max_e}")

    ch_l = cfg["ch_l"]
    ch_r = cfg["ch_r"]

    for i in range(max_e):
        if i > 0 and i % 200000 == 0:
            log(f"{path}: processed {i}/{max_e}, kept={out['counters']['kept']}")
        out["counters"]["total"] += 1

        # ---- read jagged arrays for this event ----
        try:
            ch_list = ak.to_list(arrays[cfg["branch_channel"]][i])
        except Exception:
            out["counters"]["missing_ch"] += 1
            continue
        try:
            time_list = ak.to_list(arrays[cfg["branch_time"]][i])
        except Exception:
            out["counters"]["missing_time"] += 1
            continue

        # ---- require both L and R channels present ----
        if ch_l not in ch_list or ch_r not in ch_list:
            out["counters"]["missing_bar"] += 1
            continue
        if cfg.get("strict_bar_only", False):
            module_base = cfg["module_base"]
            other_base = cfg["other_module_base"]
            rel_max = cfg.get("rel_max", 31)
            allowed = {ch_l, ch_r, TRIGGER_CHANNEL}
            if cfg.get("ch_l_down") is not None:
                allowed.add(cfg["ch_l_down"])
                allowed.add(cfg["ch_r_down"])
                if any(ch not in allowed for ch in ch_list):
                    out["counters"]["missing_bar"] += 1
                    continue
            else:
                in_other_module = lambda ch: other_base <= ch <= other_base + rel_max
                if any((ch not in allowed) and (not in_other_module(ch)) for ch in ch_list):
                    out["counters"]["missing_bar"] += 1
                    continue

        pos_l = ch_list.index(ch_l)
        pos_r = ch_list.index(ch_r)
        if pos_l >= len(time_list) or pos_r >= len(time_list):
            out["counters"]["missing_bar"] += 1
            continue

        tl = float(time_list[pos_l])
        tr = float(time_list[pos_r])
        if not (tl == tl and tr == tr):       # NaN check
            out["counters"]["missing_bar"] += 1
            continue

        # ---- optional energy cut (sum of L + R) ----
        e_sum = math.nan
        e_l = math.nan
        e_r = math.nan
        try:
            energy_list = ak.to_list(arrays[cfg["branch_energy"]][i]) if cfg["branch_energy"] in arrays.fields else []
            if pos_l < len(energy_list) and pos_r < len(energy_list):
                e_l = float(energy_list[pos_l])
                e_r = float(energy_list[pos_r])
                e_sum = (e_l + e_r) / 2.0
        except Exception:
            e_sum = math.nan
            e_l = math.nan
            e_r = math.nan

        if cfg["energy_min"] is not None:
            if not (e_sum == e_sum and e_sum >= cfg["energy_min"]):
                out["counters"]["energy_cut"] += 1
                continue
        if cfg["energy_max"] is not None:
            if not (e_sum == e_sum and e_sum <= cfg["energy_max"]):
                out["counters"]["energy_cut"] += 1
                continue
                
        # Targeted explicit channel cuts from Landau fits
        if cfg.get("energy_l_min_cut") is not None and (e_l != e_l or e_l < cfg["energy_l_min_cut"]):
            out["counters"]["energy_cut"] += 1
            continue
        if cfg.get("energy_l_max_cut") is not None and (e_l != e_l or e_l > cfg["energy_l_max_cut"]):
            out["counters"]["energy_cut"] += 1
            continue
        if cfg.get("energy_r_min_cut") is not None and (e_r != e_r or e_r < cfg["energy_r_min_cut"]):
            out["counters"]["energy_cut"] += 1
            continue
        if cfg.get("energy_r_max_cut") is not None and (e_r != e_r or e_r > cfg["energy_r_max_cut"]):
            out["counters"]["energy_cut"] += 1
            continue

        # ---- require down bar if configured ----
        down_e_down = math.nan
        down_el_d = math.nan
        down_er_d = math.nan
        tld = math.nan
        trd = math.nan
        if cfg.get("ch_l_down") is not None:
            ch_ld = cfg["ch_l_down"]
            ch_rd = cfg["ch_r_down"]
            if ch_ld not in ch_list or ch_rd not in ch_list:
                out["counters"]["missing_bar"] += 1
                continue
            pos_ld = ch_list.index(ch_ld)
            pos_rd = ch_list.index(ch_rd)
            if pos_ld >= len(time_list) or pos_rd >= len(time_list):
                continue
            tld = float(time_list[pos_ld])
            trd = float(time_list[pos_rd])
            if not (tld == tld and trd == trd):
                continue
            try:
                ed = ak.to_list(arrays[cfg["branch_energy"]][i]) if cfg["branch_energy"] in arrays.fields else []
                if pos_ld < len(ed) and pos_rd < len(ed):
                    down_el_d = float(ed[pos_ld])
                    down_er_d = float(ed[pos_rd])
                    down_e_down = (down_el_d + down_er_d) / 2.0
            except Exception:
                pass
            down_ok = True
            if cfg.get("down_energy_min") is not None:
                if not (down_e_down == down_e_down and down_e_down >= cfg["down_energy_min"]):
                    down_ok = False
            if cfg.get("down_energy_max") is not None:
                if not (down_e_down == down_e_down and down_e_down <= cfg["down_energy_max"]):
                    down_ok = False
            if cfg.get("energy_ld_min_cut") is not None and (down_el_d != down_el_d or down_el_d < cfg["energy_ld_min_cut"]):
                down_ok = False
            if cfg.get("energy_ld_max_cut") is not None and (down_el_d != down_el_d or down_el_d > cfg["energy_ld_max_cut"]):
                down_ok = False
            if cfg.get("energy_rd_min_cut") is not None and (down_er_d != down_er_d or down_er_d < cfg["energy_rd_min_cut"]):
                down_ok = False
            if cfg.get("energy_rd_max_cut") is not None and (down_er_d != down_er_d or down_er_d > cfg["energy_rd_max_cut"]):
                down_ok = False
            if not down_ok:
                continue

        # ---- MCP filter: skip events without MCP match ----
        if use_mcp and i not in mcp_events:
            out["counters"]["missing_mcp"] += 1
            continue

        # ---- store results ----
        out["t_diff"].append(tl - tr)
        if e_sum == e_sum:
            out["energy"].append(e_sum)
            out["energy_l"].append(e_l)
            out["energy_r"].append(e_r)

        # ---- phase from t1coarse ----
        phi_l = math.nan
        phi_r = math.nan
        phi_192 = math.nan
        if cfg["branch_t1coarse"] and cfg["branch_t1coarse"] in arrays.fields:
            try:
                t1coarse_list = ak.to_list(arrays[cfg["branch_t1coarse"]][i])
                if pos_l < len(t1coarse_list) and pos_r < len(t1coarse_list):
                    t1c_l = float(t1coarse_list[pos_l])
                    t1c_r = float(t1coarse_list[pos_r])
                    phi_l = (tl - t1c_l * 6250.0) % 6250.0
                    phi_r = (tr - t1c_r * 6250.0) % 6250.0
                    if phi_l == phi_l and phi_r == phi_r:
                        # shift to [-3125, 3125]
                        pd = (phi_l - phi_r + 3125.0) % 6250.0 - 3125.0
                        out["phi_diff"].append(pd)
                try:
                    pos_192 = ch_list.index(TRIGGER_CHANNEL)
                    if pos_192 < len(t1coarse_list) and pos_192 < len(time_list):
                        t1c_192 = float(t1coarse_list[pos_192])
                        t_192 = float(time_list[pos_192])
                        phi_192 = (t_192 - t1c_192 * 6250.0) % 6250.0
                except Exception:
                    pass
            except Exception:
                pass

        # ---- MCP mode: (phi_L + phi_R)/2 − phi_peak ----
        if use_mcp and i in mcp_map:
            phi_peak = mcp_map[i]
            if phi_l == phi_l:
                out["phi_l_vs_mcp"].append((phi_l - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["energy_l_mcp"].append(e_l)
            if phi_r == phi_r:
                out["phi_r_vs_mcp"].append((phi_r - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["energy_r_mcp"].append(e_r)
            if phi_l == phi_l and phi_r == phi_r:
                phi_avg = (phi_l + phi_r) / 2.0
                p_mcp = (phi_avg - phi_peak + 3125.0) % 6250.0 - 3125.0
                out["phi_vs_mcp"].append(p_mcp)
                out["energy_avg_mcp"].append(e_sum)
        if use_mcp and i in mcp_map_peak_time:
            t_bar = 0.5 * (tl + tr)
            mcp_t = mcp_map_peak_time[i]
            out["t_avg_vs_mcp"].append(t_bar - mcp_t)
            out["t_avg"].append(t_bar)
            out["mcp_t"].append(mcp_t)
        if use_mcp and i in mcp_map_peak_time and i in mcp_map_trig_time:
            try:
                pos_192 = ch_list.index(TRIGGER_CHANNEL)
                if pos_192 < len(time_list):
                    t192 = float(time_list[pos_192])
                    if t192 == t192:
                        t_bar = 0.5 * (tl + tr)
                        mcp_dt = mcp_map_peak_time[i] - mcp_map_trig_time[i]
                        out["raw_time_diff"].append((t_bar - t192) - mcp_dt)
                        out["t_192"].append(t192)
                        out["mcp_t_trig"].append(mcp_map_trig_time[i])
            except Exception:
                pass
        if use_mcp and i in mcp_map and i in mcp_map_phi_trig:
            if phi_l == phi_l and phi_r == phi_r and phi_192 == phi_192:
                phi_bar = 0.5 * (phi_l + phi_r)
                mcp_dt_phi = mcp_map[i] - mcp_map_phi_trig[i]
                rpd = ((phi_bar - phi_192) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0
                out["raw_phi_diff"].append(rpd)

        # ---- optional down-bar comparison ----
        if cfg.get("ch_l_down") is not None:
            ch_ld = cfg["ch_l_down"]
            ch_rd = cfg["ch_r_down"]
            pos_ld = ch_list.index(ch_ld)
            pos_rd = ch_list.index(ch_rd)
            
            out["t_avg_diff"].append((tl + tr) / 2.0 - (tld + trd) / 2.0)
            out["t_diff_down"].append(tld - trd)
            if down_e_down == down_e_down:  # not NaN
                out["energy_down"].append(down_e_down)
                out["energy_l_down"].append(down_el_d)
                out["energy_r_down"].append(down_er_d)
                            
            # calculate phi_avg_diff
            if cfg["branch_t1coarse"] in arrays.fields:
                try:
                    t1c_list = ak.to_list(arrays[cfg["branch_t1coarse"]][i])
                    if pos_ld < len(t1c_list) and pos_rd < len(t1c_list):
                        t1c_ld = float(t1c_list[pos_ld])
                        t1c_rd = float(t1c_list[pos_rd])
                        phi_ld = (tld - t1c_ld * 6250.0) % 6250.0
                        phi_rd = (trd - t1c_rd * 6250.0) % 6250.0
                        if phi_l == phi_l and phi_r == phi_r and phi_ld == phi_ld and phi_rd == phi_rd:
                            phi_avg_up = (phi_l + phi_r) / 2.0
                            phi_avg_down = (phi_ld + phi_rd) / 2.0
                            pad_val = (phi_avg_up - phi_avg_down + 3125.0) % 6250.0 - 3125.0
                            out["phi_avg_diff"].append(pad_val)
                            out["phi_avg_diff_sync_e_up"].append(e_sum)
                            out["phi_avg_diff_sync_e_down"].append(down_e_down)
                            
                            # Cross module Phase Up L vs Avg Down
                            phi_l_up_m_avg_down = (phi_l - phi_avg_down + 3125.0) % 6250.0 - 3125.0
                            out["phi_l_up_minus_avg_down"].append(phi_l_up_m_avg_down)
                            out["energy_l_for_cross"].append(e_l)
                except Exception:
                    pass
            # MCP-corrected down-bar time
            if use_mcp and i in mcp_map_peak_time and i in mcp_map_trig_time:
                try:
                    pos_192d = ch_list.index(TRIGGER_CHANNEL)
                    if pos_192d < len(time_list):
                        t192d = float(time_list[pos_192d])
                        if t192d == t192d:
                            t_bar_up = (tl + tr) / 2.0
                            t_bar_d = (tld + trd) / 2.0
                            mcp_dt_d = mcp_map_peak_time[i] - mcp_map_trig_time[i]
                            out["raw_time_diff_down"].append(
                                (t_bar_d - t192d) - mcp_dt_d
                            )
                            try:
                                pos_192up = ch_list.index(TRIGGER_CHANNEL)
                                if pos_192up < len(time_list):
                                    t192up = float(time_list[pos_192up])
                                    if t192up == t192up:
                                        out["raw_time_diff_up_down"].append(
                                            (t_bar_up - t192up) - (t_bar_d - t192d)
                                        )
                            except Exception:
                                pass
                except Exception:
                    pass
            # phi for down bar (from t1coarse)
            if cfg["branch_t1coarse"] and cfg["branch_t1coarse"] in arrays.fields:
                try:
                    t1c_d = ak.to_list(arrays[cfg["branch_t1coarse"]][i])
                    if pos_ld < len(t1c_d) and pos_rd < len(t1c_d):
                        t1c_ld = float(t1c_d[pos_ld])
                        t1c_rd = float(t1c_d[pos_rd])
                        phi_ld = (tld - t1c_ld * 6250.0) % 6250.0
                        phi_rd = (trd - t1c_rd * 6250.0) % 6250.0
                        out["phi_diff_down"].append(phi_ld - phi_rd)
                        phi_avg_d = (phi_ld + phi_rd) / 2.0
                        # phi_vs_mcp for down bar
                        if use_mcp and i in mcp_map:
                            p_mcp_d = (phi_avg_d - mcp_map[i] + 3125.0) % 6250.0 - 3125.0
                            out["phi_vs_mcp_down"].append(p_mcp_d)
                            out["energy_avg_mcp_down"].append(down_e_down)
                            out["phi_l_vs_mcp_down"].append((phi_ld - mcp_map[i] + 3125.0) % 6250.0 - 3125.0)
                            out["phi_r_vs_mcp_down"].append((phi_rd - mcp_map[i] + 3125.0) % 6250.0 - 3125.0)
                            out["energy_l_mcp_down"].append(down_el_d)
                            out["energy_r_mcp_down"].append(down_er_d)
                        # raw_phi_diff for down bar
                        if use_mcp and i in mcp_map and i in mcp_map_phi_trig:
                            if phi_192 == phi_192:  # not NaN
                                mcp_dt_phi_d = mcp_map[i] - mcp_map_phi_trig[i]
                                out["raw_phi_diff_down"].append(
                                    (phi_avg_d - phi_192) - mcp_dt_phi_d
                                )
                except Exception:
                    pass
            # T_avg - T_peak for down bar
            if use_mcp and i in mcp_map_peak_time:
                tb_d = 0.5 * (tld + trd)
                mcp_t_d = mcp_map_peak_time[i]
                out["t_avg_vs_mcp_down"].append(tb_d - mcp_t_d)
                out["t_avg_down"].append(tb_d)
                out["mcp_t_down"].append(mcp_t_d)

        out["counters"]["kept"] += 1

    return out


def process_file_fast(path, cfg):
    """Vectorized, chunked processing for speed."""
    out = {
        "path": path,
        "t_diff": [],
        "phi_diff": [],
        "phi_vs_mcp": [],
        "phi_l_vs_mcp": [],
        "phi_r_vs_mcp": [],
        "energy_l_mcp": [],
        "energy_r_mcp": [],
        "energy_avg_mcp": [],
        "raw_time_diff": [],
        "raw_phi_diff": [],
        "phi_diff_down": [],
        "raw_phi_diff_down": [],
        "energy": [],
        "energy_l": [],
        "energy_r": [],
        "t_avg_diff": [],
        "phi_avg_diff": [],
        "energy_down": [],
        "energy_l_down": [],
        "energy_r_down": [],
        "raw_time_diff_down": [],
        "t_diff_down": [],
        "phi_vs_mcp_down": [],
        "phi_l_vs_mcp_down": [],
        "phi_r_vs_mcp_down": [],
        "energy_l_mcp_down": [],
        "energy_r_mcp_down": [],
        "energy_avg_mcp_down": [],
        "phi_avg_diff_sync_e_up": [],
        "phi_avg_diff_sync_e_down": [],
        "phi_l_up_minus_avg_down": [],
        "energy_l_for_cross": [],
        "t_avg_vs_mcp": [],
        "t_avg_vs_mcp_down": [],
        "t_avg": [],
        "mcp_t": [],
        "t_avg_down": [],
        "mcp_t_down": [],
        "t_192": [],
        "mcp_t_trig": [],
        "raw_time_diff_up_down": [],
        "counters": {
            "total": 0,
            "missing_ch": 0,
            "missing_time": 0,
            "missing_bar": 0,
            "energy_cut": 0,
            "missing_mcp": 0,
            "kept": 0,
        },
    }

    f = uproot.open(path)
    tree_name = find_data_tree(f)
    if tree_name is None:
        log(f"{path}: no data tree found")
        return out
    tree = f[tree_name]

    branches = [cfg["branch_channel"], cfg["branch_time"]]
    if cfg["branch_energy"]:
        branches.append(cfg["branch_energy"])
    if cfg["branch_t1coarse"]:
        branches.append(cfg["branch_t1coarse"])

    use_mcp = cfg.get("use_mcp", False)
    mcp_events = set()
    mcp_map = {}
    mcp_map_phi_trig = {}
    mcp_map_peak_time = {}
    mcp_map_trig_time = {}
    mcp_mask = None

    if use_mcp:
        mcp_tree_name = cfg["mcp_tree"]
        if mcp_tree_name in f:
            mcp_tree = f[mcp_tree_name]
            mcp_idx = mcp_tree[cfg["mcp_index"]].array(library="np")
            mcp_pt = mcp_tree[cfg["mcp_phi_peak"]].array(library="np")
            mcp_phi_trig = mcp_tree[cfg["mcp_phi_trigger"]].array(library="np")
            mcp_peak_time = mcp_tree[cfg["mcp_peak_time"]].array(library="np")
            mcp_trigger_time = mcp_tree[cfg["mcp_trigger_time"]].array(library="np")
            if cfg["mcp_peak_amp"]:
                try:
                    mcp_peak_amp = mcp_tree[cfg["mcp_peak_amp"]].array(library="np")
                except Exception:
                    mcp_peak_amp = np.full(len(mcp_idx), np.nan)
            else:
                mcp_peak_amp = np.full(len(mcp_idx), np.nan)
            for j in range(len(mcp_idx)):
                try:
                    evt = int(mcp_idx[j])
                    pt = float(mcp_pt[j])
                    ptrig = float(mcp_phi_trig[j])
                    peak_time = float(mcp_peak_time[j])
                    trig_time = float(mcp_trigger_time[j])
                    amp = float(mcp_peak_amp[j])
                except Exception:
                    continue

                if cfg["mcp_peak_amp_min"] is not None and (amp != amp or amp < cfg["mcp_peak_amp_min"]):
                    continue
                if cfg["mcp_peak_amp_max"] is not None and (amp != amp or amp > cfg["mcp_peak_amp_max"]):
                    continue

                mcp_events.add(evt)
                if pt == pt:
                    mcp_map[evt] = pt
                if ptrig == ptrig:
                    mcp_map_phi_trig[evt] = ptrig
                if peak_time == peak_time:
                    mcp_map_peak_time[evt] = peak_time
                if trig_time == trig_time:
                    mcp_map_trig_time[evt] = trig_time
            log(
                f"{path}: MCP map built, index={len(mcp_events)}, "
                f"phi_peak={len(mcp_map)}, "
                f"phi_trigger={len(mcp_map_phi_trig)}, "
                f"peak_time={len(mcp_map_peak_time)}, trigger_time={len(mcp_map_trig_time)}"
            )
        else:
            log(f"{path}: MCP tree '{mcp_tree_name}' not found")

    n = tree.num_entries
    max_e = n if cfg["max_entries"] is None else min(cfg["max_entries"], n)
    log(f"{path}: data tree '{tree_name}', entries={n}, processing={max_e}")

    if use_mcp:
        mcp_mask = np.zeros(max_e, dtype=bool)
        for evt in mcp_events:
            if 0 <= evt < max_e:
                mcp_mask[evt] = True

    ch_l = cfg["ch_l"]
    ch_r = cfg["ch_r"]
    require_trigger = cfg.get("require_trigger", False)
    step = cfg.get("step_size", 200000)

    entry_start = 0
    while entry_start < max_e:
        entry_stop = min(entry_start + step, max_e)
        arrays = tree.arrays(
            branches, entry_start=entry_start, entry_stop=entry_stop, library="ak"
        )
        chunk_len = entry_stop - entry_start
        out["counters"]["total"] += chunk_len
        log(f"{path}: processed {entry_stop}/{max_e}, kept={out['counters']['kept']}")

        ch_list = arrays[cfg["branch_channel"]]
        time_list = arrays[cfg["branch_time"]]

        has_l = ak.any(ch_list == ch_l, axis=1)
        has_r = ak.any(ch_list == ch_r, axis=1)
        mask = has_l & has_r
        out["counters"]["missing_bar"] += int(ak.sum(~mask))

        if cfg.get("strict_bar_only", False):
            other_base = cfg["other_module_base"]
            rel_max = cfg.get("rel_max", 31)
            if cfg.get("ch_l_down") is not None:
                allowed = ((ch_list == ch_l) | (ch_list == ch_r)
                           | (ch_list == TRIGGER_CHANNEL)
                           | (ch_list == cfg["ch_l_down"])
                           | (ch_list == cfg["ch_r_down"]))
            else:
                in_other_module = (ch_list >= other_base) & (ch_list <= other_base + rel_max)
                allowed = (ch_list == ch_l) | (ch_list == ch_r) | (ch_list == TRIGGER_CHANNEL) | in_other_module
            mask = mask & ak.all(allowed, axis=1)

        if require_trigger:
            has_192 = ak.any(ch_list == TRIGGER_CHANNEL, axis=1)
            mask = mask & has_192

        if use_mcp and mcp_mask is not None:
            chunk_idx = np.arange(entry_start, entry_stop, dtype=int)
            mask = mask & ak.Array(mcp_mask[chunk_idx])
            out["counters"]["missing_mcp"] += int(ak.sum(~ak.Array(mcp_mask[chunk_idx]) & (has_l & has_r)))

        if not ak.any(mask):
            entry_start = entry_stop
            continue

        ch_sel = ch_list[mask]
        time_sel = time_list[mask]
        tl = ak.firsts(time_sel[ch_sel == ch_l])
        tr = ak.firsts(time_sel[ch_sel == ch_r])

        if cfg["branch_energy"] in arrays.fields:
            energy_list = arrays[cfg["branch_energy"]][mask]
            el = ak.firsts(energy_list[ch_sel == ch_l])
            er = ak.firsts(energy_list[ch_sel == ch_r])
            e_sum = (el + er) / 2.0
        else:
            el = ak.full_like(tl, np.nan, dtype=float)
            er = ak.full_like(tl, np.nan, dtype=float)
            e_sum = ak.full_like(tl, np.nan, dtype=float)

        if cfg["energy_min"] is not None:
            mask_energy = (e_sum == e_sum) & (e_sum >= cfg["energy_min"])
            out["counters"]["energy_cut"] += int(ak.sum(~mask_energy))
        else:
            mask_energy = ak.ones_like(tl, dtype=bool)
        if cfg["energy_max"] is not None:
            mask_energy = mask_energy & (e_sum <= cfg["energy_max"])
            
        # Targeted channel cuts from Pass 1 fits
        if cfg.get("energy_l_min_cut") is not None:
            mask_energy = mask_energy & (el == el) & (el >= cfg["energy_l_min_cut"])
        if cfg.get("energy_l_max_cut") is not None:
            mask_energy = mask_energy & (el == el) & (el <= cfg["energy_l_max_cut"])
        if cfg.get("energy_r_min_cut") is not None:
            mask_energy = mask_energy & (er == er) & (er >= cfg["energy_r_min_cut"])
        if cfg.get("energy_r_max_cut") is not None:
            mask_energy = mask_energy & (er == er) & (er <= cfg["energy_r_max_cut"])

        # ---- require down bar present & valid (if configured) ----
        if cfg.get("ch_l_down") is not None:
            ch_ld = cfg["ch_l_down"]
            ch_rd = cfg["ch_r_down"]
            ch_tmp = ch_sel[mask_energy]
            has_ld = ak.any(ch_tmp == ch_ld, axis=1)
            has_rd = ak.any(ch_tmp == ch_rd, axis=1)
            down_present = has_ld & has_rd
            
            down_e_ok = down_present
            if cfg["branch_energy"] in arrays.fields:
                energy_tmp = arrays[cfg["branch_energy"]][mask][mask_energy]
                el_d_tmp = ak.firsts(energy_tmp[ch_tmp == ch_ld])
                er_d_tmp = ak.firsts(energy_tmp[ch_tmp == ch_rd])
                e_d_tmp = (el_d_tmp + er_d_tmp) / 2.0
                if cfg.get("down_energy_min") is not None:
                    down_e_ok = down_e_ok & (e_d_tmp >= cfg["down_energy_min"])
                if cfg.get("down_energy_max") is not None:
                    down_e_ok = down_e_ok & (e_d_tmp <= cfg["down_energy_max"])
                if cfg.get("energy_ld_min_cut") is not None:
                    down_e_ok = down_e_ok & (el_d_tmp >= cfg["energy_ld_min_cut"])
                if cfg.get("energy_ld_max_cut") is not None:
                    down_e_ok = down_e_ok & (el_d_tmp <= cfg["energy_ld_max_cut"])
                if cfg.get("energy_rd_min_cut") is not None:
                    down_e_ok = down_e_ok & (er_d_tmp >= cfg["energy_rd_min_cut"])
                if cfg.get("energy_rd_max_cut") is not None:
                    down_e_ok = down_e_ok & (er_d_tmp <= cfg["energy_rd_max_cut"])
            down_e_ok = ak.fill_none(down_e_ok, False)
            
            n_before = int(ak.sum(mask_energy))
            out["counters"]["missing_bar"] += n_before - int(ak.sum(down_present))
            
            mask_energy_np = ak.to_numpy(mask_energy)
            mask_energy_np[mask_energy_np] = ak.to_numpy(down_e_ok)
            mask_energy = ak.Array(mask_energy_np)

        if not ak.any(mask_energy):
            entry_start = entry_stop
            continue

        tl = tl[mask_energy]
        tr = tr[mask_energy]
        el = el[mask_energy]
        er = er[mask_energy]
        e_sum = e_sum[mask_energy]

        ch_final = ch_sel[mask_energy]
        time_final = time_sel[mask_energy]

        t_diff = ak.to_numpy(tl - tr)
        out["t_diff"].extend(t_diff.tolist())
        out["energy"].extend(ak.to_numpy(e_sum).tolist())
        out["energy_l"].extend(ak.to_numpy(el).tolist())
        out["energy_r"].extend(ak.to_numpy(er).tolist())

        # ---- down-bar data (both bars guaranteed present) ----
        if cfg.get("ch_l_down") is not None:
            ch_ld = cfg["ch_l_down"]
            ch_rd = cfg["ch_r_down"]
            
            ch_d = ch_final
            time_d = time_final
            tl_d_f = ak.firsts(time_d[ch_d == ch_ld])
            tr_d_f = ak.firsts(time_d[ch_d == ch_rd])
            tl_up_d = tl
            tr_up_d = tr
            
            t_avg_d = ak.to_numpy((tl_up_d + tr_up_d) / 2.0 - (tl_d_f + tr_d_f) / 2.0)
            out["t_avg_diff"].extend(t_avg_d.tolist())
            out["t_diff_down"].extend(ak.to_numpy(tl_d_f - tr_d_f).tolist())
            
            if cfg["branch_energy"] in arrays.fields:
                energy_d = arrays[cfg["branch_energy"]][mask][mask_energy]
                el_d_final = ak.firsts(energy_d[ch_d == ch_ld])
                er_d_final = ak.firsts(energy_d[ch_d == ch_rd])
                e_d_final = (el_d_final + er_d_final) / 2.0
                out["energy_down"].extend(ak.to_numpy(e_d_final).tolist())
                out["energy_l_down"].extend(ak.to_numpy(el_d_final).tolist())
                out["energy_r_down"].extend(ak.to_numpy(er_d_final).tolist())

            # MCP-corrected down-bar time
            if use_mcp:
                chunk_idx_d = np.arange(entry_start, entry_stop, dtype=int)
                ev_d = ak.Array(chunk_idx_d)[mask][mask_energy]
                ev_d_np = ak.to_numpy(ev_d)
                t192_d = ak.firsts(time_final[ch_d == TRIGGER_CHANNEL])
                t192_d_np = ak.to_numpy(ak.fill_none(t192_d, np.nan))
                # Fetch UP bar's 192 time so it exists for calculating raw_time_diff_up_down
                t192_up_tmp = ak.firsts(time_final[ch_sel[mask_energy] == TRIGGER_CHANNEL])
                t192_np = ak.to_numpy(ak.fill_none(t192_up_tmp, np.nan))
                for kk, evt in enumerate(ev_d_np):
                    if evt in mcp_map_peak_time and evt in mcp_map_trig_time:
                        if t192_d_np[kk] == t192_d_np[kk]:
                            tb_up = float((tl_up_d[kk] + tr_up_d[kk]) / 2.0)
                            tb_d = float((tl_d_f[kk] + tr_d_f[kk]) / 2.0)
                            mcp_dt_d = mcp_map_peak_time[evt] - mcp_map_trig_time[evt]
                            out["raw_time_diff_down"].append(
                                (tb_d - t192_d_np[kk]) - mcp_dt_d
                            )
                            if hasattr(t192_np, '__len__') and kk < len(t192_np) and t192_np[kk] == t192_np[kk]:
                                out["raw_time_diff_up_down"].append(
                                    (tb_up - t192_np[kk]) - (tb_d - t192_d_np[kk])
                                )
                        # phi_diff for down bar
                        if cfg["branch_t1coarse"] in arrays.fields:
                            t1c_down = arrays[cfg["branch_t1coarse"]][mask][mask_energy]
                            t1c_ld = ak.firsts(t1c_down[ch_d == ch_ld])
                            t1c_rd = ak.firsts(t1c_down[ch_d == ch_rd])
                            phi_ld = (tl_d_f - t1c_ld * 6250.0) % 6250.0
                            phi_rd = (tr_d_f - t1c_rd * 6250.0) % 6250.0
                            pd_d = (phi_ld - phi_rd + 3125.0) % 6250.0 - 3125.0
                            out["phi_diff_down"].extend(
                                ak.to_numpy(pd_d).tolist()
                            )
                            phi_avg_d = (phi_ld + phi_rd) / 2.0
                            
                            if cfg["branch_t1coarse"] in arrays.fields:
                                t1c_up = arrays[cfg["branch_t1coarse"]][mask][mask_energy]
                                ch_sel_up = ch_final
                                t1c_l_up = ak.firsts(t1c_up[ch_sel_up == ch_l])
                                t1c_r_up = ak.firsts(t1c_up[ch_sel_up == ch_r])
                                phi_l_up = (tl_up_d - t1c_l_up * 6250.0) % 6250.0
                                phi_r_up = (tr_up_d - t1c_r_up * 6250.0) % 6250.0
                                phi_avg_up = (phi_l_up + phi_r_up) / 2.0
                                pad = (phi_avg_up - phi_avg_d + 3125.0) % 6250.0 - 3125.0
                                out["phi_avg_diff"].extend(ak.to_numpy(pad).tolist())
                                out["phi_avg_diff_sync_e_up"].extend(ak.to_numpy(e_sum).tolist())
                                out["phi_avg_diff_sync_e_down"].extend(ak.to_numpy(e_d_final).tolist())
                                
                                # Cross module Phase Up L vs Avg Down
                                p_cross = (phi_l_up - phi_avg_d + 3125.0) % 6250.0 - 3125.0
                                out["phi_l_up_minus_avg_down"].extend(ak.to_numpy(p_cross).tolist())
                                out["energy_l_for_cross"].extend(ak.to_numpy(el).tolist())
                            # phi_vs_mcp for down bar
                            for kk, evt in enumerate(ev_d_np):
                                if evt in mcp_map:
                                    try:
                                        p_mcp_d = (float(phi_avg_d[kk]) - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0
                                        out["phi_vs_mcp_down"].append(p_mcp_d)
                                        out["energy_avg_mcp_down"].append(float(e_d_final[kk]))
                                        out["phi_l_vs_mcp_down"].append((float(phi_ld[kk]) - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0)
                                        out["phi_r_vs_mcp_down"].append((float(phi_rd[kk]) - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0)
                                        out["energy_l_mcp_down"].append(float(el_d_final[kk]))
                                        out["energy_r_mcp_down"].append(float(er_d_final[kk]))
                                    except Exception:
                                        pass
                            # raw_phi_diff for down bar
                            if require_trigger or use_mcp:
                                t1c_192_d = ak.firsts(t1c_down[ch_d == TRIGGER_CHANNEL])
                                t_192_d_t1c = ak.firsts(
                                    time_final[
                                        ch_d == TRIGGER_CHANNEL
                                    ]
                                )
                                phi_192_d = (t_192_d_t1c - t1c_192_d * 6250.0) % 6250.0
                                phi_192_d_np = ak.to_numpy(ak.fill_none(phi_192_d, np.nan))
                                for kk, evt in enumerate(ev_d_np):
                                    if evt in mcp_map and evt in mcp_map_phi_trig:
                                        if phi_192_d_np[kk] == phi_192_d_np[kk]:
                                            try:
                                                mcp_dt_phi_d = mcp_map[evt] - mcp_map_phi_trig[evt]
                                                rpd_d = (float(phi_avg_d[kk]) - float(phi_192_d_np[kk]) - mcp_dt_phi_d + 3125.0) % 6250.0 - 3125.0
                                                out["raw_phi_diff_down"].append(rpd_d)
                                            except Exception:
                                                pass
                        # T_avg - T_peak for down bar
                        for kk, evt in enumerate(ev_d_np):
                            if evt in mcp_map_peak_time:
                                try:
                                    tb_d = float((tl_d_f[kk] + tr_d_f[kk]) / 2.0)
                                    mcp_t_d = mcp_map_peak_time[evt]
                                    out["t_avg_vs_mcp_down"].append(tb_d - mcp_t_d)
                                    out["t_avg_down"].append(tb_d)
                                    out["mcp_t_down"].append(mcp_t_d)
                                except Exception:
                                    pass

        # phase from t1coarse
        phi_l = math.nan
        phi_r = math.nan
        phi_192 = math.nan
        if cfg["branch_t1coarse"] in arrays.fields:
            t1coarse_list = arrays[cfg["branch_t1coarse"]][mask][mask_energy]
            ch_sel2 = ch_final
            t1c_l = ak.firsts(t1coarse_list[ch_sel2 == ch_l])
            t1c_r = ak.firsts(t1coarse_list[ch_sel2 == ch_r])
            phi_l = (tl - t1c_l * 6250.0) % 6250.0
            phi_r = (tr - t1c_r * 6250.0) % 6250.0
            pd = (phi_l - phi_r + 3125.0) % 6250.0 - 3125.0
            phi_diff = ak.to_numpy(pd)
            out["phi_diff"].extend(phi_diff.tolist())

            if require_trigger or use_mcp:
                t1c_192 = ak.firsts(t1coarse_list[ch_sel2 == TRIGGER_CHANNEL])
                t_192 = ak.firsts(time_sel[mask_energy][ch_sel2 == TRIGGER_CHANNEL])
                phi_192 = (t_192 - t1c_192 * 6250.0) % 6250.0

        # MCP mode outputs
        if use_mcp:
            chunk_idx = np.arange(entry_start, entry_stop, dtype=int)
            ev_idx = ak.Array(chunk_idx)[mask][mask_energy]
            ev_idx_np = ak.to_numpy(ev_idx)
            t192 = ak.firsts(time_sel[mask_energy][ch_sel2 == TRIGGER_CHANNEL]) if (require_trigger or use_mcp) else ak.full_like(tl, np.nan, dtype=float)
            t192_np_full = ak.to_numpy(ak.fill_none(t192, np.nan))
            phi_192_np = ak.to_numpy(ak.fill_none(phi_192, np.nan)) if hasattr(phi_192, "layout") else np.full(len(ev_idx_np), np.nan)
            for k, evt in enumerate(ev_idx_np):
                if evt in mcp_map:
                    try:
                        p_mcp_l = (float(phi_l[k]) - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0
                        out["phi_l_vs_mcp"].append(p_mcp_l)
                        out["energy_l_mcp"].append(float(el[k]))
                        p_mcp_r = (float(phi_r[k]) - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0
                        out["phi_r_vs_mcp"].append(p_mcp_r)
                        out["energy_r_mcp"].append(float(er[k]))
                        phi_avg = float((phi_l[k] + phi_r[k]) / 2.0)
                        p_mcp = (phi_avg - mcp_map[evt] + 3125.0) % 6250.0 - 3125.0
                        out["phi_vs_mcp"].append(p_mcp)
                        out["energy_avg_mcp"].append(float(e_sum[k]))
                    except Exception:
                        pass
                if evt in mcp_map_peak_time:
                    try:
                        t_bar = float((tl[k] + tr[k]) / 2.0)
                        mcp_t = mcp_map_peak_time[evt]
                        out["t_avg_vs_mcp"].append(t_bar - mcp_t)
                        out["t_avg"].append(t_bar)
                        out["mcp_t"].append(mcp_t)
                    except Exception:
                        pass
                if evt in mcp_map_peak_time and evt in mcp_map_trig_time:
                    try:
                        t_bar = float((tl[k] + tr[k]) / 2.0)
                        mcp_dt = mcp_map_peak_time[evt] - mcp_map_trig_time[evt]
                        if t192_np_full[k] == t192_np_full[k]:
                            out["raw_time_diff"].append((t_bar - t192_np_full[k]) - mcp_dt)
                            out["t_192"].append(t192_np_full[k])
                            out["mcp_t_trig"].append(mcp_map_trig_time[evt])
                    except Exception:
                        pass
                if evt in mcp_map and evt in mcp_map_phi_trig:
                    try:
                            phi_bar = float((phi_l[k] + phi_r[k]) / 2.0)
                            mcp_dt_phi = mcp_map[evt] - mcp_map_phi_trig[evt]
                            rpd = ((phi_bar - float(phi_192_np[k])) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0
                            out["raw_phi_diff"].append(rpd)
                    except Exception:
                        pass

        out["counters"]["kept"] += len(t_diff)
        entry_start = entry_stop

    return out


# ──────────────────────────────────────────────────────────────────
# Plotting
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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    
    if 'sig_f' in locals() and sig_f is not None:
        return float(sig_f)
    return None


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
        ax.grid(True, alpha=0.3)
        fitted_sigmas.append(sig_f)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved segmented plot: {out_path}")

    valid_sigmas = [s for s in fitted_sigmas if s is not None]
    if valid_sigmas:
        return sum(valid_sigmas) / len(valid_sigmas)
    return None


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
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    
    if fit_landau and mpv is not None and scale is not None:
        return cut_lo, cut_hi
    return None, None

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
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")
    
    return a, b, c


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Bar-level timing analysis (T_L − T_R)")
    p.add_argument("file", nargs="+",
                   help="Input ROOT file(s) or glob pattern(s)")
    p.add_argument("--module", choices=["up", "down"], required=True,
                   help="Detector module")
    p.add_argument("--lyso-bar", type=int, required=True,
                   help="Lyso bar index (0-15)")
    # Branch names
    p.add_argument("--branch-idx", default="channelID")
    p.add_argument("--branch-time", default="time")
    p.add_argument("--branch-energy", default="energy")
    p.add_argument("--branch-t1coarse", default="t1coarse")
    # Cuts
    p.add_argument("--energy-min", type=float, default=None)
    p.add_argument("--energy-max", type=float, default=None)
    p.add_argument("--max-entries", type=int, default=None)
    # Output
    p.add_argument("--nbins", type=int, default=100,
                   help="Number of histogram bins (default: 100)")
    p.add_argument("--out-t-diff", default=None,
                   help="Output plot path (default: t_diff_<module>_bar<N>.png)")
    p.add_argument("--out-energy", default=None,
                   help="Output energy plot path (default: energy_<module>_bar<N>.png)")
    p.add_argument("--t-diff-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="T_L-T_R histogram range, e.g. --t-diff-range -500 500")
    p.add_argument("--energy-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Energy histogram range, e.g. --energy-range 0 1000")
    p.add_argument("--phi-diff-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="phi_diff histogram range, e.g. --phi-diff-range -500 500")
    p.add_argument("--out-phi-diff", default=None,
                   help="Output phi_diff plot path (default: phi_diff_<module>_bar<N>.png)")
    # Processing
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fast", action="store_true",
                   help="Enable fast vectorized processing (recommended)")
    p.add_argument("--step-size", type=int, default=200000,
                   help="Chunk size for --fast mode (default: 200000)")
    p.add_argument("--strict-bar-only", action="store_true",
                   help="Require only the selected bar channels (L/R) "
                        "to be present in the event; allow trigger channel 192")
    # MCP mode
    p.add_argument("--mcp", action="store_true",
                   help="Enable MCP mode: filter events by MCP index, "
                        "plot (phi_L+phi_R)/2 - phi_peak")
    p.add_argument("--mcp-tree", default="MCP",
                   help="Name of the MCP TTree (default: MCP)")
    p.add_argument("--mcp-index", default="index",
                   help="MCP tree branch for event index (default: index)")
    p.add_argument("--mcp-phi-peak", default="phi_peak",
                   help="MCP tree branch for peak phase (default: phi_peak)")
    p.add_argument("--mcp-phi-trigger", default="phi_trigger",
                   help="MCP tree branch for trigger phase (default: phi_trigger)")
    p.add_argument("--mcp-peak-time", default="peak_time",
                   help="MCP tree branch for peak time (default: peak_time)")
    p.add_argument("--mcp-trigger-time", default="trigger_time",
                   help="MCP tree branch for trigger time (default: trigger_time)")
    p.add_argument("--mcp-peak-amp", default="peak_amp",
                   help="MCP tree branch for peak amplitude")
    p.add_argument("--mcp-peak-amp-min", type=float, default=None,
                   help="Minimum MCP peak_amp allowed")
    p.add_argument("--mcp-peak-amp-max", type=float, default=None,
                   help="Maximum MCP peak_amp allowed")
    p.add_argument("--out-phi-vs-mcp", default=None,
                   help="Output (phi_avg - phi_peak) plot path")
    p.add_argument("--out-raw-time-diff", default=None,
                   help="Output (t_bar - t_192) - (peak_time - trigger_time) plot path")
    p.add_argument("--out-raw-phi-diff", default=None,
                   help="Output (phi_bar - phi_192) - (phi_peak - phi_trigger) plot path")
    p.add_argument("--phi-vs-mcp-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="(phi_avg - phi_peak) histogram range")
    p.add_argument("--require-trigger", action="store_true",
                   help="When using MCP, require trigger channel 192 in the event")
    # Down module (dual-bar mode)
    p.add_argument("--down-lyso-bar", type=int, nargs='+', default=None,
                   help="Enable dual-module mode; bar index(es) on the other module")
    p.add_argument("--down-energy-min", type=float, default=None,
                   help="Minimum energy cut for the down-module bar")
    p.add_argument("--down-energy-max", type=float, default=None,
                   help="Maximum energy cut for the down-module bar")
    p.add_argument("--t-avg-diff-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="avg_time_up - avg_time_down histogram range")
    p.add_argument("--out-t-avg-diff", default=None,
                   help="Output path for avg_time_up - avg_time_down plot")
    p.add_argument("--raw-time-diff-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="raw_time_diff histogram range (overrides auto \u00b12\u03c3)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- resolve files ----
    files = []
    for item in args.file:
        if any(c in item for c in ["*", "?", "["]):
            files.extend(sorted(glob.glob(item)))
        else:
            files.append(item)
    if not files:
        print("No files matched:", args.file)
        sys.exit(2)
    log(f"{len(files)} file(s) to process")

    # ---- resolve channels ----
    base = UP_MODULE_BASE if args.module == "up" else DOWN_MODULE_BASE
    other_base = DOWN_MODULE_BASE if args.module == "up" else UP_MODULE_BASE
    if args.lyso_bar not in lyso_bar_to_channels_lr:
        print(f"Invalid lyso bar: {args.lyso_bar}. "
              f"Valid: {sorted(lyso_bar_to_channels_lr.keys())}")
        sys.exit(2)
    rel = lyso_bar_to_channels_lr[args.lyso_bar]
    ch_l = int(base + rel["L"])
    ch_r = int(base + rel["R"])
    label = f"module {args.module} bar {args.lyso_bar}"
    log(f"{label}:  ch_L={ch_l}, ch_R={ch_r}")

    # ---- optional down module(s) ----
    down_bars = []
    if args.down_lyso_bar is not None:
        for db in args.down_lyso_bar:
            if db not in lyso_bar_to_channels_lr:
                print(f"Invalid down lyso bar: {db}. "
                      f"Valid: {sorted(lyso_bar_to_channels_lr.keys())}")
                sys.exit(2)
            down_bars.append(db)
        down_mod_name = "down" if args.module == "up" else "up"
        down_bar_label = "_".join(str(b) for b in down_bars)
        log(f"down bars: {down_bars}")

    # ---- build base config (without down-bar channels) ----
    cfg_base = {
        "branch_channel": args.branch_idx,
        "branch_time": args.branch_time,
        "branch_energy": args.branch_energy,
        "branch_t1coarse": args.branch_t1coarse,
        "ch_l": ch_l,
        "ch_r": ch_r,
        "module_base": base,
        "other_module_base": other_base,
        "rel_max": 31,
        "energy_min": args.energy_min,
        "energy_max": args.energy_max,
        "max_entries": args.max_entries,
        "use_mcp": args.mcp,
        "strict_bar_only": args.strict_bar_only,
        "fast": args.fast,
        "step_size": args.step_size,
        "require_trigger": args.require_trigger or args.mcp,
        "mcp_tree": args.mcp_tree,
        "mcp_index": args.mcp_index,
        "mcp_phi_peak": args.mcp_phi_peak,
        "mcp_phi_trigger": args.mcp_phi_trigger,
        "mcp_peak_time": args.mcp_peak_time,
        "mcp_trigger_time": args.mcp_trigger_time,
        "mcp_peak_amp": args.mcp_peak_amp,
        "mcp_peak_amp_min": args.mcp_peak_amp_min,
        "mcp_peak_amp_max": args.mcp_peak_amp_max,
        "ch_l_down": None,
        "ch_r_down": None,
        "down_energy_min": None,
        "down_energy_max": None,
    }

    # ---- determine iteration list ----
    # Each entry is (down_bar_index_or_None, ch_l_down, ch_r_down)
    if down_bars:
        down_base = DOWN_MODULE_BASE if args.module == "up" else UP_MODULE_BASE
        iterations = []
        for db in down_bars:
            rel_d = lyso_bar_to_channels_lr[db]
            iterations.append((
                db,
                int(down_base + rel_d["L"]),
                int(down_base + rel_d["R"]),
            ))
    else:
        iterations = [(None, None, None)]

    # ---- run + aggregate helper ----
    all_keys = [
        "t_diff", "phi_diff", "phi_vs_mcp", "phi_l_vs_mcp", "phi_r_vs_mcp", 
        "energy_l_mcp", "energy_r_mcp", "energy_avg_mcp", "raw_time_diff", "raw_phi_diff",
        "phi_diff_down", "raw_phi_diff_down", "energy", "energy_l", "energy_r", "t_avg_diff", "phi_avg_diff",
        "energy_down", "energy_l_down", "energy_r_down", "raw_time_diff_down", "t_diff_down",
        "phi_vs_mcp_down", "phi_l_vs_mcp_down", "phi_r_vs_mcp_down", 
        "energy_l_mcp_down", "energy_r_mcp_down", "energy_avg_mcp_down",
        "phi_avg_diff_sync_e_up", "phi_avg_diff_sync_e_down",
        "phi_l_up_minus_avg_down", "energy_l_for_cross",
        "t_avg_vs_mcp", "t_avg_vs_mcp_down",
        "t_avg", "mcp_t", "t_avg_down", "mcp_t_down",
        "t_192", "mcp_t_trig",
        "raw_time_diff_up_down",
    ]
    
    def run_extraction(current_cfg_base):
        local_merged = {k: [] for k in all_keys}
        local_counters = None

        for db, ch_ld, ch_rd in iterations:
            cfg = dict(current_cfg_base)
            cfg["ch_l_down"] = ch_ld
            cfg["ch_r_down"] = ch_rd
            cfg["down_energy_min"] = args.down_energy_min if db is not None else None
            cfg["down_energy_max"] = args.down_energy_max if db is not None else None
            if db is not None:
                log(f"--- iteration: up bar {args.lyso_bar} vs {down_mod_name} bar {db}  "
                    f"ch_L_down={ch_ld}, ch_R_down={ch_rd} ---")

            if args.workers > 1:
                log(f"Running with {args.workers} worker processes")
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    if args.fast:
                        results = list(ex.map(partial(process_file_fast, cfg=cfg), files))
                    else:
                        results = list(ex.map(partial(process_file, cfg=cfg), files))
            else:
                log("Running single-process")
                if args.fast:
                    results = [process_file_fast(p, cfg) for p in files]
                else:
                    results = [process_file(p, cfg) for p in files]

            # aggregate this iteration
            if local_counters is None:
                local_counters = {k: 0 for k in results[0]["counters"]}
            for res in results:
                for k in all_keys:
                    local_merged[k].extend(res[k])
                for k in local_counters:
                    local_counters[k] += res["counters"].get(k, 0)
                    
        return local_merged, local_counters

    # ==============================================================
    # PASS 1: Discovery (Raw Energies to Determine Cuts)
    # ==============================================================
    log("=== PASS 1: Extracting raw energy spectra for Landau cuts ===")
    merged_raw, _ = run_extraction(cfg_base)

    out_energy_l = f"energy_l_{args.module}_bar{args.lyso_bar}.png"
    cut_l_lo, cut_l_hi = plot_energy(merged_raw["energy_l"], out_energy_l,
               f"Energy L  ({label})", nbins=args.nbins,
               hist_range=tuple(args.energy_range) if args.energy_range else None,
               fit_landau=True)

    out_energy_r = f"energy_r_{args.module}_bar{args.lyso_bar}.png"
    cut_r_lo, cut_r_hi = plot_energy(merged_raw["energy_r"], out_energy_r,
               f"Energy R  ({label})", nbins=args.nbins,
               hist_range=tuple(args.energy_range) if args.energy_range else None,
               fit_landau=True)
               
    # Set the discovered cuts into the config for Pass 2
    if cut_l_lo is not None: cfg_base["energy_l_min_cut"] = cut_l_lo
    if cut_l_hi is not None: cfg_base["energy_l_max_cut"] = cut_l_hi
    if cut_r_lo is not None: cfg_base["energy_r_min_cut"] = cut_r_lo
    if cut_r_hi is not None: cfg_base["energy_r_max_cut"] = cut_r_hi
    
    # Process down bar cut limits if applicable
    if down_bars:
        out_energy_l_down = f"energy_l_{down_mod_name}_bar{down_bar_label}.png"
        cut_ld_lo, cut_ld_hi = plot_energy(merged_raw["energy_l_down"], out_energy_l_down,
                    f"Energy L  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.energy_range) if args.energy_range else None,
                    fit_landau=True)

        out_energy_r_down = f"energy_r_{down_mod_name}_bar{down_bar_label}.png"
        cut_rd_lo, cut_rd_hi = plot_energy(merged_raw["energy_r_down"], out_energy_r_down,
                    f"Energy R  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.energy_range) if args.energy_range else None,
                    fit_landau=True)
                    
        if cut_ld_lo is not None: cfg_base["energy_ld_min_cut"] = cut_ld_lo
        if cut_ld_hi is not None: cfg_base["energy_ld_max_cut"] = cut_ld_hi
        if cut_rd_lo is not None: cfg_base["energy_rd_min_cut"] = cut_rd_lo
        if cut_rd_hi is not None: cfg_base["energy_rd_max_cut"] = cut_rd_hi

    # ==============================================================
    # PASS 2: Final Filtering
    # ==============================================================
    log("=== PASS 2: Filtering events strictly within energy cuts ===")
    merged, counters = run_extraction(cfg_base)

    log(f"counters: {counters}")
    log(f"T_L − T_R entries: {len(merged['t_diff'])}")
    log(f"phi_diff entries: {len(merged['phi_diff'])}")

    # ---- variables to store fitted sigmas for absolute resolution calc ----
    sig_phi_up_mcp = None
    sig_phi_down_mcp = None
    sig_phi_up_down = None

    # ---- plot ----
    out_path = args.out_t_diff or f"t_diff_{args.module}_bar{args.lyso_bar}.png"
    plot_t_diff(merged["t_diff"], out_path,
                f"$T_L - T_R$  ({label})", nbins=args.nbins,
                hist_range=tuple(args.t_diff_range) if args.t_diff_range else None)

    # ---- energy plot ----
    out_energy = args.out_energy or f"energy_{args.module}_bar{args.lyso_bar}.png"
    plot_energy(merged["energy"], out_energy,
               f"Energy  ({label})", nbins=args.nbins,
               hist_range=tuple(args.energy_range) if args.energy_range else None,
               fit_landau=False)


    # ---- phi_diff plot ----
    out_phi = args.out_phi_diff or f"phi_diff_{args.module}_bar{args.lyso_bar}.png"
    plot_t_diff(merged["phi_diff"], out_phi,
                f"$\\phi_L - \\phi_R$  ({label})", nbins=args.nbins,
                hist_range=tuple(args.phi_diff_range) if args.phi_diff_range else None,
                xlabel="$\\phi_L - \\phi_R$  (ps)")

    # ---- phi_avg − phi_peak plot (MCP mode) ----
    if args.mcp:
        out_mcp = args.out_phi_vs_mcp or f"phi_vs_mcp_{args.module}_bar{args.lyso_bar}.png"
        log(f"phi_vs_mcp entries: {len(merged['phi_vs_mcp'])}")
        sig_phi_up_mcp = plot_t_diff(merged["phi_vs_mcp"], out_mcp,
                    f"$(\\phi_L + \\phi_R)/2 - \\phi_{{peak}}$  ({label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                    xlabel="$(\\phi_L + \\phi_R)/2 - \\phi_{peak}$  (ps)",
                    color="steelblue")

        # Plot individual L/R channels against MCP
        out_phi_l_mcp = f"phi_l_vs_mcp_{args.module}_bar{args.lyso_bar}.png"
        plot_t_diff(merged["phi_l_vs_mcp"], out_phi_l_mcp,
                    f"$\\phi_L - \\phi_{{peak}}$  ({label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                    xlabel="$\\phi_L - \\phi_{peak}$  (ps)",
                    color="cornflowerblue")

        out_phi_r_mcp = f"phi_r_vs_mcp_{args.module}_bar{args.lyso_bar}.png"
        plot_t_diff(merged["phi_r_vs_mcp"], out_phi_r_mcp,
                    f"$\\phi_R - \\phi_{{peak}}$  ({label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                    xlabel="$\\phi_R - \\phi_{peak}$  (ps)",
                    color="cornflowerblue")

        # Plot 2D Scatters against single-channel energy and capture polynomial fit parameters
        out_phi_l_vs_e = f"phi_l_vs_e_mcp_{args.module}_bar{args.lyso_bar}.png"
        a_l, b_l, c_l = plot_phi_vs_energy(merged["phi_l_vs_mcp"], merged["energy_l_mcp"],
                           out_phi_l_vs_e, f"$\\phi_L - \\phi_{{peak}}$ vs $E_L$  ({label})",
                           ylabel="$\\phi_L - \\phi_{peak}$ (ps)")

        out_phi_r_vs_e = f"phi_r_vs_e_mcp_{args.module}_bar{args.lyso_bar}.png"
        a_r, b_r, c_r = plot_phi_vs_energy(merged["phi_r_vs_mcp"], merged["energy_r_mcp"],
                           out_phi_r_vs_e, f"$\\phi_R - \\phi_{{peak}}$ vs $E_R$  ({label})",
                           ylabel="$\\phi_R - \\phi_{peak}$ (ps)")

        out_phi_avg_vs_e = f"phi_avg_vs_e_mcp_{args.module}_bar{args.lyso_bar}.png"
        a_avg, b_avg, c_avg = plot_phi_vs_energy(merged["phi_vs_mcp"], merged["energy_avg_mcp"],
                           out_phi_avg_vs_e, f"$\\langle \\phi \\rangle - \\phi_{{peak}}$ vs $\\langle E \\rangle$  ({label})",
                           ylabel="$\\langle \\phi \\rangle - \\phi_{peak}$ (ps)")
        
    # ---- Apply Calibrations and Plot Corrected 1D / 2D Histograms ----
        if a_l is not None and b_l is not None and c_l is not None:
            calib_phi_l = []
            calib_e_l = []
            for p, e in zip(merged["phi_l_vs_mcp"], merged["energy_l_mcp"]):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_l.append(p - (a_l * e**2 + b_l * e + c_l))
                    calib_e_l.append(e)
            out_phi_l_calib = f"phi_l_vs_mcp_calibrated_{args.module}_bar{args.lyso_bar}.png"
            plot_t_diff(calib_phi_l, out_phi_l_calib,
                        f"Calibrated $\\phi_L - \\phi_{{peak}}$  ({label})",
                        nbins=args.nbins,
                        hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                        xlabel="Calibrated $\\phi_L - \\phi_{peak}$  (ps)",
                        color="mediumseagreen")
            out_phi_l_vs_e_calib = f"phi_l_vs_e_mcp_calibrated_{args.module}_bar{args.lyso_bar}.png"
            plot_phi_vs_energy(calib_phi_l, calib_e_l,
                               out_phi_l_vs_e_calib, f"Calib $\\phi_L - \\phi_{{peak}}$ vs $E_L$  ({label})",
                               ylabel="Calibrated $\\phi_L - \\phi_{peak}$ (ps)", fit_poly=False)

        if a_r is not None and b_r is not None and c_r is not None:
            calib_phi_r = []
            calib_e_r = []
            for p, e in zip(merged["phi_r_vs_mcp"], merged["energy_r_mcp"]):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_r.append(p - (a_r * e**2 + b_r * e + c_r))
                    calib_e_r.append(e)
            out_phi_r_calib = f"phi_r_vs_mcp_calibrated_{args.module}_bar{args.lyso_bar}.png"
            plot_t_diff(calib_phi_r, out_phi_r_calib,
                        f"Calibrated $\\phi_R - \\phi_{{peak}}$  ({label})",
                        nbins=args.nbins,
                        hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                        xlabel="Calibrated $\\phi_R - \\phi_{peak}$  (ps)",
                        color="mediumseagreen")
            out_phi_r_vs_e_calib = f"phi_r_vs_e_mcp_calibrated_{args.module}_bar{args.lyso_bar}.png"
            plot_phi_vs_energy(calib_phi_r, calib_e_r,
                               out_phi_r_vs_e_calib, f"Calib $\\phi_R - \\phi_{{peak}}$ vs $E_R$  ({label})",
                               ylabel="Calibrated $\\phi_R - \\phi_{peak}$ (ps)", fit_poly=False)

        if a_avg is not None and b_avg is not None and c_avg is not None:
            calib_phi_avg = []
            calib_e_avg = []
            for p, e in zip(merged["phi_vs_mcp"], merged["energy_avg_mcp"]):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_avg.append(p - (a_avg * e**2 + b_avg * e + c_avg))
                    calib_e_avg.append(e)
            out_phi_avg_calib = f"phi_avg_vs_e_mcp_calibrated_{args.module}_bar{args.lyso_bar}.png"
            sig_phi_up_mcp_calib = plot_t_diff(calib_phi_avg, out_phi_avg_calib,
                        f"Calibrated $\\langle \\phi \\rangle - \\phi_{{peak}}$  ({label})",
                        nbins=args.nbins,
                        hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                        xlabel="Calibrated $\\langle \\phi \\rangle - \\phi_{peak}$  (ps)",
                        color="forestgreen")
            out_phi_avg_vs_e_calib = f"phi_avg_vs_e_mcp_calibrated_scatter_{args.module}_bar{args.lyso_bar}.png"
            plot_phi_vs_energy(calib_phi_avg, calib_e_avg,
                               out_phi_avg_vs_e_calib, f"Calib $\\langle \\phi \\rangle - \\phi_{{peak}}$ vs $\\langle E \\rangle$  ({label})",
                               ylabel="Calibrated $\\langle \\phi \\rangle - \\phi_{peak}$ (ps)", fit_poly=False)
            
            # Print a quick terminal line to compare the sigmas
            if sig_phi_up_mcp and sig_phi_up_mcp_calib:
                log(f"Calibration applied. Raw Intrinsic Sigma: {sig_phi_up_mcp:.2f}ps | Calibrated Sigma: {sig_phi_up_mcp_calib:.2f}ps")

    # ---- avg_time_up − avg_time_down plot (dual-module mode) ----
    if down_bars:
        dual_label = (f"module {args.module} bar {args.lyso_bar} vs "
                      f"{down_mod_name} bar {down_bar_label}")

        # ---- down module energy spectrum ----
        out_energy_down = f"energy_{down_mod_name}_bar{down_bar_label}.png"
        plot_energy(merged["energy_down"], out_energy_down,
                    f"Energy  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.energy_range) if args.energy_range else None,
                    fit_landau=False)

        # ---- down module T_L − T_R ----
        log(f"t_diff_down entries: {len(merged['t_diff_down'])}")
        out_tdiff_down = f"t_diff_{down_mod_name}_bar{down_bar_label}.png"
        plot_t_diff(merged["t_diff_down"], out_tdiff_down,
                    f"$T_L - T_R$  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.t_diff_range) if args.t_diff_range else None)

        log(f"phi_avg_diff entries: {len(merged['phi_avg_diff'])}")
        out_phiavg = f"phi_avg_up_minus_down_bar{args.lyso_bar}_bar{down_bar_label}.png"
        sig_phi_up_down = plot_t_diff(merged["phi_avg_diff"], out_phiavg,
                    f"$\\langle \\phi \\rangle_{{up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                    xlabel="$\\langle \\phi \\rangle_{up} - \\langle \\phi \\rangle_{down}$  (ps)",
                    color="cyan")
                    
        # ---- UP L phi - DOWN AVG phi cross-module plot ----
        log(f"phi_l_up_minus_avg_down entries: {len(merged['phi_l_up_minus_avg_down'])}")
        out_phi_l_up_down = f"phi_l_up_minus_avg_down_bar{args.lyso_bar}_bar{down_bar_label}.png"
        sig_phi_l_up_down = plot_t_diff(merged["phi_l_up_minus_avg_down"], out_phi_l_up_down,
                    f"$\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                    xlabel="$\\phi_{L, up} - \\langle \\phi \\rangle_{down}$  (ps)",
                    color="darkcyan")
                    
        out_phi_l_up_down_vs_e = f"phi_l_up_minus_avg_down_vs_e_l_up_bar{args.lyso_bar}_bar{down_bar_label}.png"
        a_cross, b_cross, c_cross = plot_phi_vs_energy(merged["phi_l_up_minus_avg_down"], merged["energy_l_for_cross"],
                            out_phi_l_up_down_vs_e, f"$\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$ vs $E_{{L, up}}$  ({dual_label})",
                            ylabel="$\\phi_{L, up} - \\langle \\phi \\rangle_{down}$ (ps)")
                            
        if a_cross is not None and b_cross is not None and c_cross is not None:
            calib_phi_cross = []
            calib_e_cross = []
            for p, e in zip(merged["phi_l_up_minus_avg_down"], merged["energy_l_for_cross"]):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_cross.append(p - (a_cross * e**2 + b_cross * e + c_cross))
                    calib_e_cross.append(e)
            
            out_phi_cross_calib = f"phi_l_up_minus_avg_down_calibrated_bar{args.lyso_bar}_bar{down_bar_label}.png"
            plot_t_diff(calib_phi_cross, out_phi_cross_calib,
                        f"Calib $\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                        nbins=args.nbins,
                        hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                        xlabel="Calib $\\phi_{L, up} - \\langle \\phi \\rangle_{down}$  (ps)",
                        color="teal")
                        
            out_phi_cross_vs_e_calib = f"phi_l_up_minus_avg_down_vs_e_l_up_calibrated_bar{args.lyso_bar}_bar{down_bar_label}.png"
            plot_phi_vs_energy(calib_phi_cross, calib_e_cross,
                               out_phi_cross_vs_e_calib, f"Calib $\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$ vs $E_{{L, up}}$  ({dual_label})",
                               ylabel="Calib $\\phi_{L, up} - \\langle \\phi \\rangle_{down}$ (ps)", fit_poly=False)

        # ---- down module phi_diff ----
        log(f"phi_diff_down entries: {len(merged['phi_diff_down'])}")
        out_phidiff_down = f"phi_diff_{down_mod_name}_bar{down_bar_label}.png"
        plot_t_diff(merged["phi_diff_down"], out_phidiff_down,
                    f"$\\phi_L - \\phi_R$  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_diff_range) if args.phi_diff_range else None,
                    xlabel="$\\phi_L - \\phi_R$  (ps)")

        sig_phi_down_mcp_calib = None

        # ---- down module raw_time_diff vs MCP ----
        if args.mcp:
            # ---- down module phi_vs_mcp ----
            log(f"phi_vs_mcp_down entries: {len(merged['phi_vs_mcp_down'])}")
            out_phi_mcp_down = f"phi_vs_mcp_{down_mod_name}_bar{down_bar_label}.png"
            phi_mcp_down_range = None
            phi_mcp_down_vals = merged["phi_vs_mcp_down"]
            if merged["phi_vs_mcp_down"]:
                mu = float(np.mean(merged["phi_vs_mcp_down"]))
                sig = float(np.std(merged["phi_vs_mcp_down"], ddof=1))
                if sig > 0:
                    lo = mu - 2.0 * sig
                    hi = mu + 2.0 * sig
                    phi_mcp_down_vals = [x for x in merged["phi_vs_mcp_down"] if lo <= x <= hi]
                    log(f"phi_vs_mcp_down (±2σ) kept: {len(phi_mcp_down_vals)} of {len(merged['phi_vs_mcp_down'])}")
                    phi_mcp_down_range = (lo, hi)
                else:
                    phi_mcp_down_range = (mu - 0.5, mu + 0.5)
            sig_phi_down_mcp = plot_t_diff(phi_mcp_down_vals, out_phi_mcp_down,
                        f"$(\\phi_L + \\phi_R)/2 - \\phi_{{peak}}$  "
                        f"(module {down_mod_name} bar {down_bar_label})",
                        nbins=args.nbins,
                        hist_range=phi_mcp_down_range,
                        xlabel="$(\\phi_L + \\phi_R)/2 - \\phi_{peak}$  (ps)",
                        color="steelblue")

            # Plot individual L/R channels against MCP (Down Bar)
            out_phi_l_mcp_down = f"phi_l_vs_mcp_{down_mod_name}_bar{down_bar_label}.png"
            plot_t_diff(merged["phi_l_vs_mcp_down"], out_phi_l_mcp_down,
                        f"$\\phi_L - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                        nbins=args.nbins,
                        hist_range=phi_mcp_down_range,
                        xlabel="$\\phi_L - \\phi_{peak}$  (ps)",
                        color="cornflowerblue")

            out_phi_r_mcp_down = f"phi_r_vs_mcp_{down_mod_name}_bar{down_bar_label}.png"
            plot_t_diff(merged["phi_r_vs_mcp_down"], out_phi_r_mcp_down,
                        f"$\\phi_R - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                        nbins=args.nbins,
                        hist_range=phi_mcp_down_range,
                        xlabel="$\\phi_R - \\phi_{peak}$  (ps)",
                        color="cornflowerblue")

            # Plot 2D Scatters against single-channel energy and capture polynomial fit parameters (Down Bar)
            out_phi_l_vs_e_down = f"phi_l_vs_e_mcp_{down_mod_name}_bar{down_bar_label}.png"
            a_l_down, b_l_down, c_l_down = plot_phi_vs_energy(merged["phi_l_vs_mcp_down"], merged["energy_l_mcp_down"],
                            out_phi_l_vs_e_down, f"$\\phi_L - \\phi_{{peak}}$ vs $E_L$  (module {down_mod_name} bar {down_bar_label})",
                            ylabel="$\\phi_L - \\phi_{peak}$ (ps)")

            out_phi_r_vs_e_down = f"phi_r_vs_e_mcp_{down_mod_name}_bar{down_bar_label}.png"
            a_r_down, b_r_down, c_r_down = plot_phi_vs_energy(merged["phi_r_vs_mcp_down"], merged["energy_r_mcp_down"],
                            out_phi_r_vs_e_down, f"$\\phi_R - \\phi_{{peak}}$ vs $E_R$  (module {down_mod_name} bar {down_bar_label})",
                            ylabel="$\\phi_R - \\phi_{peak}$ (ps)")

            out_phi_avg_vs_e_down = f"phi_avg_vs_e_mcp_{down_mod_name}_bar{down_bar_label}.png"
            a_avg_down, b_avg_down, c_avg_down = plot_phi_vs_energy(merged["phi_vs_mcp_down"], merged["energy_avg_mcp_down"],
                            out_phi_avg_vs_e_down, f"$\\langle \\phi \\rangle - \\phi_{{peak}}$ vs $\\langle E \\rangle$  (module {down_mod_name} bar {down_bar_label})",
                            ylabel="$\\langle \\phi \\rangle - \\phi_{peak}$ (ps)")
            
            # ---- Apply Calibrations and Plot Corrected 1D Histograms (Down Bar) ----
            if a_l_down is not None and b_l_down is not None and c_l_down is not None:
                calib_phi_l_down = []
                calib_e_l_down = []
                for p, e in zip(merged["phi_l_vs_mcp_down"], merged["energy_l_mcp_down"]):
                    if np.isfinite(p) and np.isfinite(e):
                        calib_phi_l_down.append(p - (a_l_down * e**2 + b_l_down * e + c_l_down))
                        calib_e_l_down.append(e)
                out_phi_l_calib_down = f"phi_l_vs_mcp_calibrated_{down_mod_name}_bar{down_bar_label}.png"
                plot_t_diff(calib_phi_l_down, out_phi_l_calib_down,
                            f"Calibrated $\\phi_L - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                            nbins=args.nbins,
                            hist_range=phi_mcp_down_range,
                            xlabel="Calibrated $\\phi_L - \\phi_{peak}$  (ps)",
                            color="mediumseagreen")
                out_phi_l_vs_e_calib_down = f"phi_l_vs_e_mcp_calibrated_{down_mod_name}_bar{down_bar_label}.png"
                plot_phi_vs_energy(calib_phi_l_down, calib_e_l_down,
                                   out_phi_l_vs_e_calib_down, f"Calib $\\phi_L - \\phi_{{peak}}$ vs $E_L$  (module {down_mod_name} bar {down_bar_label})",
                                   ylabel="Calibrated $\\phi_L - \\phi_{peak}$ (ps)", fit_poly=False)


            if a_r_down is not None and b_r_down is not None and c_r_down is not None:
                calib_phi_r_down = []
                calib_e_r_down = []
                for p, e in zip(merged["phi_r_vs_mcp_down"], merged["energy_r_mcp_down"]):
                    if np.isfinite(p) and np.isfinite(e):
                        calib_phi_r_down.append(p - (a_r_down * e**2 + b_r_down * e + c_r_down))
                        calib_e_r_down.append(e)
                out_phi_r_calib_down = f"phi_r_vs_mcp_calibrated_{down_mod_name}_bar{down_bar_label}.png"
                plot_t_diff(calib_phi_r_down, out_phi_r_calib_down,
                            f"Calibrated $\\phi_R - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                            nbins=args.nbins,
                            hist_range=phi_mcp_down_range,
                            xlabel="Calibrated $\\phi_R - \\phi_{peak}$  (ps)",
                            color="mediumseagreen")
                out_phi_r_vs_e_calib_down = f"phi_r_vs_e_mcp_calibrated_{down_mod_name}_bar{down_bar_label}.png"
                plot_phi_vs_energy(calib_phi_r_down, calib_e_r_down,
                                   out_phi_r_vs_e_calib_down, f"Calib $\\phi_R - \\phi_{{peak}}$ vs $E_R$  (module {down_mod_name} bar {down_bar_label})",
                                   ylabel="Calibrated $\\phi_R - \\phi_{peak}$ (ps)", fit_poly=False)


            if a_avg_down is not None and b_avg_down is not None and c_avg_down is not None:
                calib_phi_avg_down = []
                calib_e_avg_down = []
                for p, e in zip(merged["phi_vs_mcp_down"], merged["energy_avg_mcp_down"]):
                    if np.isfinite(p) and np.isfinite(e):
                        calib_phi_avg_down.append(p - (a_avg_down * e**2 + b_avg_down * e + c_avg_down))
                        calib_e_avg_down.append(e)
                out_phi_avg_calib_down = f"phi_avg_vs_e_mcp_calibrated_{down_mod_name}_bar{down_bar_label}.png"
                sig_phi_down_mcp_calib = plot_t_diff(calib_phi_avg_down, out_phi_avg_calib_down,
                            f"Calibrated $\\langle \\phi \\rangle - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                            nbins=args.nbins,
                            hist_range=phi_mcp_down_range,
                            xlabel="Calibrated $\\langle \\phi \\rangle - \\phi_{peak}$  (ps)",
                            color="forestgreen")
                out_phi_avg_vs_e_calib_down = f"phi_avg_vs_e_mcp_calibrated_scatter_{down_mod_name}_bar{down_bar_label}.png"
                plot_phi_vs_energy(calib_phi_avg_down, calib_e_avg_down,
                                   out_phi_avg_vs_e_calib_down, f"Calib $\\langle \\phi \\rangle - \\phi_{{peak}}$ vs $\\langle E \\rangle$  (module {down_mod_name} bar {down_bar_label})",
                                   ylabel="Calibrated $\\langle \\phi \\rangle - \\phi_{peak}$ (ps)", fit_poly=False)

            if sig_phi_down_mcp and sig_phi_down_mcp_calib:
                log(f"Down Bar Calibrated. Raw Intrinsic Sigma: {sig_phi_down_mcp:.2f}ps | Calibrated Sigma: {sig_phi_down_mcp_calib:.2f}ps")

            # ---- up vs down module raw time difference (t_bar_up - t_192) - (t_bar_down - t_192d) ----

    # ---- Calculate Absolute Timing Resolutions ----
    if sig_phi_up_down and sig_phi_up_mcp and sig_phi_down_mcp:
        print("\n" + "="*50)
        print("  ABSOLUTE TIMING RESOLUTION CALCULATION")
        print("="*50)
        print(f"Measured sigmas:")
        print(f"  sigma(up - down)   = {sig_phi_up_down:.2f} ps")
        print(f"  sigma(up - MCP)    = {sig_phi_up_mcp:.2f} ps")
        print(f"  sigma(down - MCP)  = {sig_phi_down_mcp:.2f} ps")
        
        v1 = sig_phi_up_down**2
        v2 = sig_phi_up_mcp**2
        v3 = sig_phi_down_mcp**2
        
        var_up = 0.5 * (v1 + v2 - v3)
        var_down = 0.5 * (v1 + v3 - v2)
        var_mcp = 0.5 * (v2 + v3 - v1)
        
        print("\nCalculated intrinsic resolutions:")
        if var_up > 0:
            print(f"  sigma(up bar)      = {math.sqrt(var_up):.2f} ps")
        else:
            print("  sigma(up bar)      = [imaginary] (variance < 0)")
            
        if var_down > 0:
            print(f"  sigma(down bar(s)) = {math.sqrt(var_down):.2f} ps")
        else:
            print("  sigma(down bar(s)) = [imaginary] (variance < 0)")
            
        if var_mcp > 0:
            print(f"  sigma(MCP)         = {math.sqrt(var_mcp):.2f} ps")
        else:
            print("  sigma(MCP)         = [imaginary] (variance < 0)")
        print("="*50 + "\n")

    if sig_phi_up_down and sig_phi_up_mcp_calib and sig_phi_down_mcp_calib:
        print("\n" + "="*50)
        print("  CALIBRATED ABSOLUTE TIMING RESOLUTION CALCULATION")
        print("="*50)
        print(f"Post-Calibration Measured sigmas:")
        print(f"  sigma(up - down)_{{raw}}     = {sig_phi_up_down:.2f} ps  <-- (Uncalibrated)")
        print(f"  sigma(up - MCP)_{{calib}}    = {sig_phi_up_mcp_calib:.2f} ps")
        print(f"  sigma(down - MCP)_{{calib}}  = {sig_phi_down_mcp_calib:.2f} ps")
        
        v1_calib = sig_phi_up_down**2
        v2_calib = sig_phi_up_mcp_calib**2
        v3_calib = sig_phi_down_mcp_calib**2
        
        var_up_calib = 0.5 * (v1_calib + v2_calib - v3_calib)
        var_down_calib = 0.5 * (v1_calib + v3_calib - v2_calib)
        var_mcp_calib = 0.5 * (v2_calib + v3_calib - v1_calib)
        
        print("\nCalculated intrinsic resolutions (Using raw up-down):")
        if var_up_calib > 0:
            print(f"  sigma(up bar)_{{calib}}      = {math.sqrt(var_up_calib):.2f} ps")
        else:
            print("  sigma(up bar)_{calib}      = [imaginary] (variance < 0)")
            
        if var_down_calib > 0:
            print(f"  sigma(down bar(s))_{{calib}} = {math.sqrt(var_down_calib):.2f} ps")
        else:
            print("  sigma(down bar(s))_{calib} = [imaginary] (variance < 0)")
            
        if var_mcp_calib > 0:
            print(f"  sigma(MCP)_{{calib}}         = {math.sqrt(var_mcp_calib):.2f} ps")
        else:
            print("  sigma(MCP)_{calib}         = [imaginary] (variance < 0)")
        print("="*50 + "\n")

    log("done")


if __name__ == "__main__":
    main()
