#!/usr/bin/env python3
"""
Per-file ROOT data extraction for bar-level timing analysis.

Contains both the event-loop processor (`process_file`) and the
vectorized chunked processor (`process_file_fast`).
"""

import math
import numpy as np
import uproot
import awkward as ak

from bar_helpers import log, find_data_tree, make_output_dict
from channel_mapping import TRIGGER_CHANNEL


def _mcp_internal_dt_selector(peak_times, trigger_times, scale):
    """Return keep mask and summary for a robust MCP internal timing cut."""
    dt_vals = np.asarray(peak_times, dtype=float) - np.asarray(trigger_times, dtype=float)
    finite = np.isfinite(dt_vals)
    keep_mask = np.ones(len(dt_vals), dtype=bool)
    if not np.any(finite):
        return keep_mask, None

    dt_finite = dt_vals[finite]
    median = float(np.median(dt_finite))
    mad = float(np.median(np.abs(dt_finite - median)))
    method = "MAD"
    width = scale * mad

    if not np.isfinite(width) or width <= 0:
        q25, q75 = np.percentile(dt_finite, [25.0, 75.0])
        iqr = float(q75 - q25)
        if np.isfinite(iqr) and iqr > 0:
            method = "IQR"
            width = scale * 0.7413 * iqr
        else:
            method = "exact"
            width = 0.0

    keep_mask[finite] = np.abs(dt_finite - median) <= width
    keep_mask[~finite] = False
    summary = {
        "median": median,
        "width": float(width),
        "method": method,
        "kept": int(np.sum(keep_mask[finite])),
        "total": int(np.sum(finite)),
    }
    return keep_mask, summary


def _build_mcp_maps(root_file, cfg, path):
    """Load MCP branches, apply MCP selections, and return event maps."""
    mcp_events = set()
    mcp_map = {}
    mcp_map_phi_trig = {}
    mcp_map_peak_time = {}
    mcp_map_trig_time = {}
    use_mcp = cfg.get("use_mcp", False)

    if not use_mcp:
        return mcp_events, mcp_map, mcp_map_phi_trig, mcp_map_peak_time, mcp_map_trig_time

    mcp_tree_name = cfg["mcp_tree"]
    if mcp_tree_name not in root_file:
        log(f"{path}: MCP tree '{mcp_tree_name}' not found")
        return mcp_events, mcp_map, mcp_map_phi_trig, mcp_map_peak_time, mcp_map_trig_time

    mcp_tree = root_file[mcp_tree_name]
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

    base_keep = np.ones(len(mcp_idx), dtype=bool)
    for j in range(len(mcp_idx)):
        try:
            amp = float(mcp_peak_amp[j])
        except Exception:
            base_keep[j] = False
            continue
        if cfg["mcp_peak_amp_min"] is not None and (amp != amp or amp < cfg["mcp_peak_amp_min"]):
            base_keep[j] = False
            continue
        if cfg["mcp_peak_amp_max"] is not None and (amp != amp or amp > cfg["mcp_peak_amp_max"]):
            base_keep[j] = False
            continue

    internal_keep = np.ones(len(mcp_idx), dtype=bool)
    if cfg.get("mcp_internal_dt_cut", False):
        internal_keep, dt_summary = _mcp_internal_dt_selector(
            mcp_peak_time[base_keep], mcp_trigger_time[base_keep], cfg.get("mcp_internal_dt_nmad", 3.0)
        )
        filtered_keep = np.zeros(len(mcp_idx), dtype=bool)
        filtered_keep[base_keep] = internal_keep
        base_keep = filtered_keep
        if dt_summary is not None:
            log(
                f"{path}: MCP internal dt cut [{dt_summary['method']}] "
                f"median={dt_summary['median']:.3f} ps, half-window={dt_summary['width']:.3f} ps, "
                f"kept={dt_summary['kept']}/{dt_summary['total']}"
            )

    for j in range(len(mcp_idx)):
        if not base_keep[j]:
            continue
        try:
            evt = int(mcp_idx[j])
            pt = float(mcp_pt[j])
            ptrig = float(mcp_phi_trig[j])
            peak_time = float(mcp_peak_time[j])
            trig_time = float(mcp_trigger_time[j])
        except Exception:
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
    return mcp_events, mcp_map, mcp_map_phi_trig, mcp_map_peak_time, mcp_map_trig_time


# ──────────────────────────────────────────────────────────────────
# Per-file processing  (event loop)
# ──────────────────────────────────────────────────────────────────

def process_file(path, cfg):
    """Process one ROOT file.  Return dict of accumulated lists."""
    out = make_output_dict(path)

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

    # ---- build MCP index map ----
    mcp_events, mcp_map, mcp_map_phi_trig, mcp_map_peak_time, mcp_map_trig_time = _build_mcp_maps(f, cfg, path)
    use_mcp = cfg.get("use_mcp", False)

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

        # =========================================================
        # 1. EARLY EXTRACTION & STRICT FILTERING ("Golden Event")
        # =========================================================
        
        # --- Up Bar Energy & Cuts ---
        e_sum, e_l, e_r = math.nan, math.nan, math.nan
        try:
            energy_list = ak.to_list(arrays[cfg["branch_energy"]][i]) if cfg["branch_energy"] in arrays.fields else []
            if pos_l < len(energy_list) and pos_r < len(energy_list):
                e_l = float(energy_list[pos_l])
                e_r = float(energy_list[pos_r])
                e_sum = (e_l + e_r) / 2.0
        except Exception:
            pass

        if not (e_sum == e_sum): continue
        if cfg["energy_min"] is not None and e_sum < cfg["energy_min"]: out["counters"]["energy_cut"] += 1; continue
        if cfg["energy_max"] is not None and e_sum > cfg["energy_max"]: out["counters"]["energy_cut"] += 1; continue
        if cfg.get("energy_l_min_cut") is not None and e_l < cfg["energy_l_min_cut"]: out["counters"]["energy_cut"] += 1; continue
        if cfg.get("energy_l_max_cut") is not None and e_l > cfg["energy_l_max_cut"]: out["counters"]["energy_cut"] += 1; continue
        if cfg.get("energy_r_min_cut") is not None and e_r < cfg["energy_r_min_cut"]: out["counters"]["energy_cut"] += 1; continue
        if cfg.get("energy_r_max_cut") is not None and e_r > cfg["energy_r_max_cut"]: out["counters"]["energy_cut"] += 1; continue

        # --- Up Bar Phase ---
        phi_l, phi_r = math.nan, math.nan
        if cfg["branch_t1coarse"] and cfg["branch_t1coarse"] in arrays.fields:
            try:
                t1coarse_list = ak.to_list(arrays[cfg["branch_t1coarse"]][i])
                if pos_l < len(t1coarse_list) and pos_r < len(t1coarse_list):
                    t1c_l = float(t1coarse_list[pos_l])
                    t1c_r = float(t1coarse_list[pos_r])
                    phi_l = (tl - t1c_l * 6250.0) % 6250.0
                    phi_r = (tr - t1c_r * 6250.0) % 6250.0
            except Exception: pass
        if not (phi_l == phi_l and phi_r == phi_r): continue

        # --- Down Bar (if configured) ---
        tld, trd, down_el_d, down_er_d, down_e_down, phi_ld, phi_rd = (math.nan,) * 7
        req_down = cfg.get("ch_l_down") is not None
        if req_down:
            ch_ld, ch_rd = cfg["ch_l_down"], cfg["ch_r_down"]
            if ch_ld not in ch_list or ch_rd not in ch_list:
                out["counters"]["missing_bar"] += 1
                continue
            pos_ld, pos_rd = ch_list.index(ch_ld), ch_list.index(ch_rd)
            if pos_ld >= len(time_list) or pos_rd >= len(time_list): continue
            tld, trd = float(time_list[pos_ld]), float(time_list[pos_rd])
            if not (tld == tld and trd == trd): continue
            
            # Energy
            try:
                if pos_ld < len(energy_list) and pos_rd < len(energy_list):
                    down_el_d = float(energy_list[pos_ld])
                    down_er_d = float(energy_list[pos_rd])
                    down_e_down = (down_el_d + down_er_d) / 2.0
            except Exception: pass
            
            if not (down_e_down == down_e_down): continue
            if cfg.get("down_energy_min") is not None and down_e_down < cfg["down_energy_min"]: continue
            if cfg.get("down_energy_max") is not None and down_e_down > cfg["down_energy_max"]: continue
            if cfg.get("energy_ld_min_cut") is not None and down_el_d < cfg["energy_ld_min_cut"]: continue
            if cfg.get("energy_ld_max_cut") is not None and down_el_d > cfg["energy_ld_max_cut"]: continue
            if cfg.get("energy_rd_min_cut") is not None and down_er_d < cfg["energy_rd_min_cut"]: continue
            if cfg.get("energy_rd_max_cut") is not None and down_er_d > cfg["energy_rd_max_cut"]: continue
            
            # Phase
            try:
                if pos_ld < len(t1coarse_list) and pos_rd < len(t1coarse_list):
                    t1c_ld = float(t1coarse_list[pos_ld])
                    t1c_rd = float(t1coarse_list[pos_rd])
                    phi_ld = (tld - t1c_ld * 6250.0) % 6250.0
                    phi_rd = (trd - t1c_rd * 6250.0) % 6250.0
            except Exception: pass
            if not (phi_ld == phi_ld and phi_rd == phi_rd): continue

        # --- MCP & Trigger (Ch192) ---
        t192, phi_192 = math.nan, math.nan
        mcp_peak_time, mcp_trig_time, phi_peak, phi_trig = (math.nan,) * 4
        if use_mcp:
            if i not in mcp_events:
                out["counters"]["missing_mcp"] += 1
                continue
            if i not in mcp_map or i not in mcp_map_phi_trig or i not in mcp_map_peak_time or i not in mcp_map_trig_time:
                continue
            phi_peak = mcp_map[i]
            phi_trig = mcp_map_phi_trig[i]
            mcp_peak_time = mcp_map_peak_time[i]
            mcp_trig_time = mcp_map_trig_time[i]

            # Need 192 for MCP correlation
            if TRIGGER_CHANNEL not in ch_list: continue
            pos_192 = ch_list.index(TRIGGER_CHANNEL)
            if pos_192 >= len(time_list): continue
            t192 = float(time_list[pos_192])
            if not (t192 == t192): continue
            
            try:
                if pos_192 < len(t1coarse_list):
                    t1c_192 = float(t1coarse_list[pos_192])
                    phi_192 = (t192 - t1c_192 * 6250.0) % 6250.0
            except Exception: pass
            if not (phi_192 == phi_192): continue


        # =========================================================
        # 2. COMMIT: All required variables are valid ("Golden Event")
        # =========================================================
        out["counters"]["kept"] += 1

        # -- Up Bar --
        out["t_diff"].append(tl - tr)
        out["energy"].append(e_sum)
        out["energy_l"].append(e_l)
        out["energy_r"].append(e_r)
        
        pd_up = (phi_l - phi_r + 3125.0) % 6250.0 - 3125.0
        out["phi_diff"].append(pd_up)
        phi_avg_up = (phi_r + 0.5 * pd_up + 3125.0) % 6250.0 - 3125.0

        if use_mcp:
            t_bar_up = (tl + tr) / 2.0
            
            out["phi_l_vs_mcp"].append((phi_l - phi_peak + 3125.0) % 6250.0 - 3125.0)
            out["phi_r_vs_mcp"].append((phi_r - phi_peak + 3125.0) % 6250.0 - 3125.0)
            out["phi_l_vs_mcp_sync"].append((phi_l - phi_peak + 3125.0) % 6250.0 - 3125.0)
            out["phi_r_vs_mcp_sync"].append((phi_r - phi_peak + 3125.0) % 6250.0 - 3125.0)
            out["phi_vs_mcp"].append((phi_avg_up - phi_peak + 3125.0) % 6250.0 - 3125.0)
            out["phi_vs_mcp_trig"].append((phi_avg_up - phi_trig + 3125.0) % 6250.0 - 3125.0)
            # keep raw L/R for wrap diagnostics
            out["phi_l_raw_sync"].append(phi_l)
            out["phi_r_raw_sync"].append(phi_r)
            
            out["energy_l_mcp"].append(e_l)
            out["energy_r_mcp"].append(e_r)
            out["energy_l_mcp_sync"].append(e_l)
            out["energy_r_mcp_sync"].append(e_r)
            out["energy_avg_mcp"].append(e_sum)
            
            out["t_avg_vs_mcp"].append(t_bar_up - mcp_peak_time)
            out["t_avg"].append(t_bar_up)
            out["mcp_t"].append(mcp_peak_time)
            out["t_192"].append(t192)
            out["mcp_t_trig"].append(mcp_trig_time)
            
            mcp_dt_time = mcp_peak_time - mcp_trig_time
            out["raw_time_diff"].append((t_bar_up - t192) - mcp_dt_time)
            
            mcp_dt_phi = phi_peak - phi_trig
            out["raw_phi_diff"].append(((phi_avg_up - phi_192) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0)
            out["phi_trig_diff"].append((phi_192 - phi_trig + 3125.0) % 6250.0 - 3125.0)

        # -- Down Bar & Cross-Module --
        if req_down:
            pd_down = (phi_ld - phi_rd + 3125.0) % 6250.0 - 3125.0
            phi_avg_down = (phi_rd + 0.5 * pd_down + 3125.0) % 6250.0 - 3125.0
            t_bar_down = (tld + trd) / 2.0
            
            out["t_diff_down"].append(tld - trd)
            out["t_avg_diff"].append(t_bar_up - t_bar_down)
            out["energy_down"].append(down_e_down)
            out["energy_l_down"].append(down_el_d)
            out["energy_r_down"].append(down_er_d)
            
            out["phi_diff_down"].append(pd_down)
            out["phi_avg_diff"].append((phi_avg_up - phi_avg_down + 3125.0) % 6250.0 - 3125.0)
            out["phi_avg_diff_sync_e_up"].append(e_sum)
            out["phi_avg_diff_sync_e_down"].append(down_e_down)
            
            out["phi_l_up_minus_avg_down"].append((phi_l - phi_avg_down + 3125.0) % 6250.0 - 3125.0)
            out["energy_l_for_cross"].append(e_l)
            out["phi_r_up_minus_avg_down"].append((phi_r - phi_avg_down + 3125.0) % 6250.0 - 3125.0)
            out["energy_r_for_cross"].append(e_r)

            if use_mcp:
                out["phi_l_vs_mcp_down"].append((phi_ld - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_r_vs_mcp_down"].append((phi_rd - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_l_vs_mcp_down_sync"].append((phi_ld - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_r_vs_mcp_down_sync"].append((phi_rd - phi_peak + 3125.0) % 6250.0 - 3125.0)
                # Cross-module sync arrays populated in the same MCP-coincident branch.
                out["phi_l_up_minus_avg_down_sync"].append((phi_l - phi_avg_down + 3125.0) % 6250.0 - 3125.0)
                out["phi_r_up_minus_avg_down_sync"].append((phi_r - phi_avg_down + 3125.0) % 6250.0 - 3125.0)
                out["energy_l_for_cross_sync"].append(e_l)
                out["energy_r_for_cross_sync"].append(e_r)
                out["phi_vs_mcp_down"].append((phi_avg_down - phi_peak + 3125.0) % 6250.0 - 3125.0)
                
                out["energy_l_mcp_down"].append(down_el_d)
                out["energy_r_mcp_down"].append(down_er_d)
                out["energy_l_mcp_down_sync"].append(down_el_d)
                out["energy_r_mcp_down_sync"].append(down_er_d)
                out["energy_avg_mcp_down"].append(down_e_down)
                
                out["t_avg_vs_mcp_down"].append(t_bar_down - mcp_peak_time)
                out["t_avg_down"].append(t_bar_down)
                out["mcp_t_down"].append(mcp_peak_time)
                
                out["raw_time_diff_down"].append((t_bar_down - t192) - mcp_dt_time)
                out["raw_time_diff_up_down"].append(((t_bar_up - t192) - (t_bar_down - t192)))
                out["raw_phi_diff_down"].append(((phi_avg_down - phi_192) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0)

    return out


# ──────────────────────────────────────────────────────────────────
# Per-file processing  (vectorized / chunked)
# ──────────────────────────────────────────────────────────────────

def process_file_fast(path, cfg):
    """Vectorized, chunked processing for speed."""
    out = make_output_dict(path)

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
    mcp_events, mcp_map, mcp_map_phi_trig, mcp_map_peak_time, mcp_map_trig_time = _build_mcp_maps(f, cfg, path)
    mcp_mask = None

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

        # =========================================================
        # 1. EARLY EXTRACTION & STRICT FILTERING ("Golden Event")
        # =========================================================
        
        # --- Up Bar ---
        ch_sel_up = ch_list[mask]
        time_sel_up = time_list[mask]
        tl = ak.firsts(time_sel_up[ch_sel_up == ch_l])
        tr = ak.firsts(time_sel_up[ch_sel_up == ch_r])
        
        # Up Energy
        if cfg["branch_energy"] in arrays.fields:
            energy_list = arrays[cfg["branch_energy"]][mask]
            el = ak.firsts(energy_list[ch_sel_up == ch_l])
            er = ak.firsts(energy_list[ch_sel_up == ch_r])
            e_sum = (el + er) / 2.0
        else:
            el, er, e_sum = (ak.full_like(tl, np.nan, dtype=float) for _ in range(3))

        mask_energy = (e_sum == e_sum)
        if cfg["energy_min"] is not None: mask_energy = mask_energy & (e_sum >= cfg["energy_min"])
        if cfg["energy_max"] is not None: mask_energy = mask_energy & (e_sum <= cfg["energy_max"])
        if cfg.get("energy_l_min_cut") is not None: mask_energy = mask_energy & (el >= cfg["energy_l_min_cut"])
        if cfg.get("energy_l_max_cut") is not None: mask_energy = mask_energy & (el <= cfg["energy_l_max_cut"])
        if cfg.get("energy_r_min_cut") is not None: mask_energy = mask_energy & (er >= cfg["energy_r_min_cut"])
        if cfg.get("energy_r_max_cut") is not None: mask_energy = mask_energy & (er <= cfg["energy_r_max_cut"])
        
        mask_np = ak.to_numpy(mask)
        mask_np[mask_np] = ak.to_numpy(mask_energy)
        mask = ak.Array(mask_np)

        # Up Phase
        phi_l, phi_r = ak.full_like(tl, np.nan, dtype=float), ak.full_like(tl, np.nan, dtype=float)
        if cfg["branch_t1coarse"] in arrays.fields:
            t1c_list = arrays[cfg["branch_t1coarse"]][mask]
            ch_sel_tmp = ch_list[mask]
            t1c_l = ak.firsts(t1c_list[ch_sel_tmp == ch_l])
            t1c_r = ak.firsts(t1c_list[ch_sel_tmp == ch_r])
            tl_m = ak.firsts(time_list[mask][ch_sel_tmp == ch_l])
            tr_m = ak.firsts(time_list[mask][ch_sel_tmp == ch_r])
            phi_l = (tl_m - t1c_l * 6250.0) % 6250.0
            phi_r = (tr_m - t1c_r * 6250.0) % 6250.0
        
        mask_np = ak.to_numpy(mask)
        mask_np[mask_np] = ak.to_numpy((phi_l == phi_l) & (phi_r == phi_r))
        mask = ak.Array(mask_np)


        # --- Down Bar (if configured) ---
        req_down = cfg.get("ch_l_down") is not None
        if req_down:
            ch_ld, ch_rd = cfg["ch_l_down"], cfg["ch_r_down"]
            ch_tmp = ch_list[mask]
            has_ld = ak.any(ch_tmp == ch_ld, axis=1)
            has_rd = ak.any(ch_tmp == ch_rd, axis=1)
            
            mask_np = ak.to_numpy(mask)
            mask_np[mask_np] = ak.to_numpy(has_ld & has_rd)
            mask = ak.Array(mask_np)

            # Down Energy
            if cfg["branch_energy"] in arrays.fields:
                energy_tmp = arrays[cfg["branch_energy"]][mask]
                ch_tmp = ch_list[mask]
                el_d = ak.firsts(energy_tmp[ch_tmp == ch_ld])
                er_d = ak.firsts(energy_tmp[ch_tmp == ch_rd])
                e_d_tmp = (el_d + er_d) / 2.0
                
                down_e_ok = (e_d_tmp == e_d_tmp)
                if cfg.get("down_energy_min") is not None: down_e_ok = down_e_ok & (e_d_tmp >= cfg["down_energy_min"])
                if cfg.get("down_energy_max") is not None: down_e_ok = down_e_ok & (e_d_tmp <= cfg["down_energy_max"])
                if cfg.get("energy_ld_min_cut") is not None: down_e_ok = down_e_ok & (el_d >= cfg["energy_ld_min_cut"])
                if cfg.get("energy_ld_max_cut") is not None: down_e_ok = down_e_ok & (el_d <= cfg["energy_ld_max_cut"])
                if cfg.get("energy_rd_min_cut") is not None: down_e_ok = down_e_ok & (er_d >= cfg["energy_rd_min_cut"])
                if cfg.get("energy_rd_max_cut") is not None: down_e_ok = down_e_ok & (er_d <= cfg["energy_rd_max_cut"])
                
                mask_np = ak.to_numpy(mask)
                mask_np[mask_np] = ak.to_numpy(ak.fill_none(down_e_ok, False))
                mask = ak.Array(mask_np)

            # Down Phase
            if cfg["branch_t1coarse"] in arrays.fields:
                t1c_list = arrays[cfg["branch_t1coarse"]][mask]
                ch_tmp = ch_list[mask]
                t1c_ld = ak.firsts(t1c_list[ch_tmp == ch_ld])
                t1c_rd = ak.firsts(t1c_list[ch_tmp == ch_rd])
                tl_m_d = ak.firsts(time_list[mask][ch_tmp == ch_ld])
                tr_m_d = ak.firsts(time_list[mask][ch_tmp == ch_rd])
                phi_ld = (tl_m_d - t1c_ld * 6250.0) % 6250.0
                phi_rd = (tr_m_d - t1c_rd * 6250.0) % 6250.0
                
                mask_np = ak.to_numpy(mask)
                mask_np[mask_np] = ak.to_numpy((phi_ld == phi_ld) & (phi_rd == phi_rd))
                mask = ak.Array(mask_np)

        # --- MCP & Trigger (Ch192) ---
        if use_mcp:
            # MCP requires Ch192
            has_192 = ak.any(ch_list[mask] == TRIGGER_CHANNEL, axis=1)
            mask_np = ak.to_numpy(mask)
            mask_np[mask_np] = ak.to_numpy(has_192)
            mask = ak.Array(mask_np)

            t1c_192, phi_192, t192 = None, None, None
            if cfg["branch_t1coarse"] in arrays.fields:
                t1c_list = arrays[cfg["branch_t1coarse"]][mask]
                ch_tmp = ch_list[mask]
                t1c_192 = ak.firsts(t1c_list[ch_tmp == TRIGGER_CHANNEL])
                t192 = ak.firsts(time_list[mask][ch_tmp == TRIGGER_CHANNEL])
                phi_192 = (t192 - t1c_192 * 6250.0) % 6250.0
                
                mask_np = ak.to_numpy(mask)
                mask_np[mask_np] = ak.to_numpy(phi_192 == phi_192)
                mask = ak.Array(mask_np)

            chunk_idx = np.arange(entry_start, entry_stop, dtype=int)
            ev_idx = ak.Array(chunk_idx)[mask]
            
            # Map valid MCP properties
            mcp_ok = np.zeros(len(ev_idx), dtype=bool)
            for i, evt in enumerate(ak.to_numpy(ev_idx)):
                if evt in mcp_events and evt in mcp_map and evt in mcp_map_phi_trig and evt in mcp_map_peak_time and evt in mcp_map_trig_time:
                    mcp_ok[i] = True
            
            mask_np = ak.to_numpy(mask)
            mask_np[mask_np] = mcp_ok
            mask = ak.Array(mask_np)

        if not ak.any(mask):
            entry_start = entry_stop
            continue

        # =========================================================
        # 2. COMMIT: All required variables are valid ("Golden Event")
        # =========================================================
        
        # Sliced Arrays
        ch_final = ch_list[mask]
        time_final = time_list[mask]
        
        tl = ak.firsts(time_final[ch_final == ch_l])
        tr = ak.firsts(time_final[ch_final == ch_r])
        el = ak.firsts(arrays[cfg["branch_energy"]][mask][ch_final == ch_l]) if cfg["branch_energy"] in arrays.fields else ak.full_like(tl, np.nan)
        er = ak.firsts(arrays[cfg["branch_energy"]][mask][ch_final == ch_r]) if cfg["branch_energy"] in arrays.fields else ak.full_like(tl, np.nan)
        e_sum = (el + er) / 2.0

        t1c_final = arrays[cfg["branch_t1coarse"]][mask] if cfg["branch_t1coarse"] in arrays.fields else None
        
        # Up Phase
        if t1c_final is not None:
            t1c_l = ak.firsts(t1c_final[ch_final == ch_l])
            t1c_r = ak.firsts(t1c_final[ch_final == ch_r])
            phi_l = (tl - t1c_l * 6250.0) % 6250.0
            phi_r = (tr - t1c_r * 6250.0) % 6250.0

        n_kept = len(tl)
        out["counters"]["kept"] += n_kept

        # -- Up Bar --
        t_diff = ak.to_numpy(tl - tr)
        out["t_diff"].extend(t_diff.tolist())
        out["energy"].extend(ak.to_numpy(e_sum).tolist())
        out["energy_l"].extend(ak.to_numpy(el).tolist())
        out["energy_r"].extend(ak.to_numpy(er).tolist())
        
        if t1c_final is not None:
            pd_up = (phi_l - phi_r + 3125.0) % 6250.0 - 3125.0
            out["phi_diff"].extend(ak.to_numpy(pd_up).tolist())

        # -- Down Bar & Cross-Module --
        if req_down:
            tl_d = ak.firsts(time_final[ch_final == ch_ld])
            tr_d = ak.firsts(time_final[ch_final == ch_rd])
            el_d = ak.firsts(arrays[cfg["branch_energy"]][mask][ch_final == ch_ld]) if cfg["branch_energy"] in arrays.fields else ak.full_like(tl_d, np.nan)
            er_d = ak.firsts(arrays[cfg["branch_energy"]][mask][ch_final == ch_rd]) if cfg["branch_energy"] in arrays.fields else ak.full_like(tl_d, np.nan)
            e_sum_d = (el_d + er_d) / 2.0
            
            t_bar_up = (tl + tr) / 2.0
            t_bar_d = (tl_d + tr_d) / 2.0
            
            out["t_avg_diff"].extend(ak.to_numpy(t_bar_up - t_bar_d).tolist())
            out["t_diff_down"].extend(ak.to_numpy(tl_d - tr_d).tolist())
            
            out["energy_down"].extend(ak.to_numpy(e_sum_d).tolist())
            out["energy_l_down"].extend(ak.to_numpy(el_d).tolist())
            out["energy_r_down"].extend(ak.to_numpy(er_d).tolist())

            if t1c_final is not None:
                t1c_ld = ak.firsts(t1c_final[ch_final == ch_ld])
                t1c_rd = ak.firsts(t1c_final[ch_final == ch_rd])
                phi_ld = (tl_d - t1c_ld * 6250.0) % 6250.0
                phi_rd = (tr_d - t1c_rd * 6250.0) % 6250.0
                pd_up = (phi_l - phi_r + 3125.0) % 6250.0 - 3125.0
                pd_down = (phi_ld - phi_rd + 3125.0) % 6250.0 - 3125.0
                phi_avg_up = (phi_r + 0.5 * pd_up + 3125.0) % 6250.0 - 3125.0
                phi_avg_down = (phi_rd + 0.5 * pd_down + 3125.0) % 6250.0 - 3125.0
                
                out["phi_diff_down"].extend(ak.to_numpy(pd_down).tolist())
                out["phi_avg_diff"].extend(ak.to_numpy((phi_avg_up - phi_avg_down + 3125.0) % 6250.0 - 3125.0).tolist())
                out["phi_avg_diff_sync_e_up"].extend(ak.to_numpy(e_sum).tolist())
                out["phi_avg_diff_sync_e_down"].extend(ak.to_numpy(e_sum_d).tolist())
                out["phi_l_up_minus_avg_down"].extend(ak.to_numpy((phi_l - phi_avg_down + 3125.0) % 6250.0 - 3125.0).tolist())
                out["energy_l_for_cross"].extend(ak.to_numpy(el).tolist())
                out["phi_r_up_minus_avg_down"].extend(ak.to_numpy((phi_r - phi_avg_down + 3125.0) % 6250.0 - 3125.0).tolist())
                out["energy_r_for_cross"].extend(ak.to_numpy(er).tolist())

        # -- MCP outputs --
        if use_mcp:
            chunk_idx = np.arange(entry_start, entry_stop, dtype=int)
            ev_idx_np = ak.to_numpy(ak.Array(chunk_idx)[mask])
            t192 = ak.firsts(time_final[ch_final == TRIGGER_CHANNEL])
            t1c_192 = ak.firsts(t1c_final[ch_final == TRIGGER_CHANNEL])
            phi_192 = (t192 - t1c_192 * 6250.0) % 6250.0
            
            phi_l_np = ak.to_numpy(phi_l)
            phi_r_np = ak.to_numpy(phi_r)
            d_wrap_up = (phi_l_np - phi_r_np + 3125.0) % 6250.0 - 3125.0
            phi_avg_up_np = (phi_r_np + 0.5 * d_wrap_up + 3125.0) % 6250.0 - 3125.0
            t_bar_up_np = ak.to_numpy((tl + tr) / 2.0)
            el_np = ak.to_numpy(el)
            er_np = ak.to_numpy(er)
            e_sum_np = ak.to_numpy(e_sum)
            t192_np = ak.to_numpy(t192)
            phi_192_np = ak.to_numpy(phi_192)

            if req_down:
                phi_ld_np = ak.to_numpy(phi_ld)
                phi_rd_np = ak.to_numpy(phi_rd)
                d_wrap_down = (phi_ld_np - phi_rd_np + 3125.0) % 6250.0 - 3125.0
                phi_avg_down_np = (phi_rd_np + 0.5 * d_wrap_down + 3125.0) % 6250.0 - 3125.0
                t_bar_d_np = ak.to_numpy((tl_d + tr_d) / 2.0)
                el_d_np = ak.to_numpy(el_d)
                er_d_np = ak.to_numpy(er_d)
                e_sum_d_np = ak.to_numpy(e_sum_d)

            for k, evt in enumerate(ev_idx_np):
                phi_peak = mcp_map[evt]
                phi_trig = mcp_map_phi_trig[evt]
                mcp_peak_time = mcp_map_peak_time[evt]
                mcp_trig_time = mcp_map_trig_time[evt]
                
                # Up Bar
                out["phi_l_vs_mcp"].append((phi_l_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_r_vs_mcp"].append((phi_r_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_l_vs_mcp_sync"].append((phi_l_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_r_vs_mcp_sync"].append((phi_r_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                phi_avg_wrap = phi_avg_up_np[k]
                out["phi_vs_mcp"].append((phi_avg_wrap - phi_peak + 3125.0) % 6250.0 - 3125.0)
                out["phi_vs_mcp_trig"].append((phi_avg_wrap - phi_trig + 3125.0) % 6250.0 - 3125.0)
                out["phi_l_raw_sync"].append(phi_l_np[k])
                out["phi_r_raw_sync"].append(phi_r_np[k])
                
                out["energy_l_mcp"].append(el_np[k])
                out["energy_r_mcp"].append(er_np[k])
                out["energy_l_mcp_sync"].append(el_np[k])
                out["energy_r_mcp_sync"].append(er_np[k])
                out["energy_avg_mcp"].append(e_sum_np[k])
                
                out["t_avg_vs_mcp"].append(t_bar_up_np[k] - mcp_peak_time)
                out["t_avg"].append(t_bar_up_np[k])
                out["mcp_t"].append(mcp_peak_time)
                out["t_192"].append(t192_np[k])
                out["mcp_t_trig"].append(mcp_trig_time)
                
                mcp_dt_time = mcp_peak_time - mcp_trig_time
                out["raw_time_diff"].append((t_bar_up_np[k] - t192_np[k]) - mcp_dt_time)
                
                mcp_dt_phi = phi_peak - phi_trig
                out["raw_phi_diff"].append(((phi_avg_wrap - phi_192_np[k]) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0)
                out["phi_trig_diff"].append((phi_192_np[k] - phi_trig + 3125.0) % 6250.0 - 3125.0)

                # Down Bar
                if req_down:
                    out["phi_l_vs_mcp_down"].append((phi_ld_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                    out["phi_r_vs_mcp_down"].append((phi_rd_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                    out["phi_l_vs_mcp_down_sync"].append((phi_ld_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                    out["phi_r_vs_mcp_down_sync"].append((phi_rd_np[k] - phi_peak + 3125.0) % 6250.0 - 3125.0)
                    d_wrap_down = (phi_ld_np[k] - phi_rd_np[k] + 3125.0) % 6250.0 - 3125.0
                    phi_avg_down_wrap = (phi_rd_np[k] + 0.5 * d_wrap_down + 3125.0) % 6250.0 - 3125.0
                    out["phi_vs_mcp_down"].append((phi_avg_down_wrap - phi_peak + 3125.0) % 6250.0 - 3125.0)
                    
                    out["energy_l_mcp_down"].append(el_d_np[k])
                    out["energy_r_mcp_down"].append(er_d_np[k])
                    out["energy_l_mcp_down_sync"].append(el_d_np[k])
                    out["energy_r_mcp_down_sync"].append(er_d_np[k])
                    out["energy_avg_mcp_down"].append(e_sum_d_np[k])
                    # Cross-module sync arrays populated in the same MCP-coincident branch.
                    out["phi_l_up_minus_avg_down_sync"].append((phi_l_np[k] - phi_avg_down_wrap + 3125.0) % 6250.0 - 3125.0)
                    out["phi_r_up_minus_avg_down_sync"].append((phi_r_np[k] - phi_avg_down_wrap + 3125.0) % 6250.0 - 3125.0)
                    out["energy_l_for_cross_sync"].append(el_np[k])
                    out["energy_r_for_cross_sync"].append(er_np[k])
                    
                    out["t_avg_vs_mcp_down"].append(t_bar_d_np[k] - mcp_peak_time)
                    out["t_avg_down"].append(t_bar_d_np[k])
                    out["mcp_t_down"].append(mcp_peak_time)
                    
                    out["raw_time_diff_down"].append((t_bar_d_np[k] - t192_np[k]) - mcp_dt_time)
                    out["raw_time_diff_up_down"].append(((t_bar_up_np[k] - t192_np[k]) - (t_bar_d_np[k] - t192_np[k])))
                    out["raw_phi_diff_down"].append(((phi_avg_down_wrap - phi_192_np[k]) - mcp_dt_phi + 3125.0) % 6250.0 - 3125.0)

        out["counters"]["kept"] += len(t_diff)
        entry_start = entry_stop

    return out
