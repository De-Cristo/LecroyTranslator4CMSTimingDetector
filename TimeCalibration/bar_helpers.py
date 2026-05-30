#!/usr/bin/env python3
"""
Shared helper utilities for bar-level timing analysis.

Extracted from bar_analysis.py to avoid duplication across modules.
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[bar] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────
# ROOT helpers
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


# ──────────────────────────────────────────────────────────────────
# Fit functions
# ──────────────────────────────────────────────────────────────────

def gauss(x, a, mu, sigma):
    """Un-normalised Gaussian."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ──────────────────────────────────────────────────────────────────
# Output-dict factory  (avoids duplicating the template in
#                        process_file and process_file_fast)
# ──────────────────────────────────────────────────────────────────

def make_output_dict(path):
    """Return a fresh accumulator dict for per-file processing."""
    return {
        "path": path,
        "t_diff": [],        # T_L − T_R  (ps)
        "phi_diff": [],      # phi_L − phi_R  (ps), from t1fine
        "phi_vs_mcp": [],    # wrap-aware avg vs phi_peak  (ps)
        "phi_vs_mcp_trig": [], # wrap-aware avg vs phi_trigger (ps)
        "phi_l_vs_mcp": [],  # phi_L - phi_peak (ps)
        "phi_r_vs_mcp": [],  # phi_R - phi_peak (ps)
        "energy_l_mcp": [],  # E_L (when mcp match exists)
        "energy_r_mcp": [],  # E_R (when mcp match exists)
        "energy_avg_mcp": [], # (E_L + E_R)/2 (when mcp match exists)
        "phi_l_raw_sync": [],  # raw phi_L synced with phi_vs_mcp
        "phi_r_raw_sync": [],  # raw phi_R synced with phi_vs_mcp
        # ---- synchronized per-channel arrays (same events as phi_vs_mcp) ----
        "phi_l_vs_mcp_sync": [],  # phi_L - phi_peak, synced with phi_vs_mcp
        "phi_r_vs_mcp_sync": [],  # phi_R - phi_peak, synced with phi_vs_mcp
        "energy_l_mcp_sync": [],  # E_L, synced with phi_vs_mcp
        "energy_r_mcp_sync": [],  # E_R, synced with phi_vs_mcp
        # ---- synchronized per-channel arrays for down bar ----
        "phi_l_vs_mcp_down_sync": [],
        "phi_r_vs_mcp_down_sync": [],
        "energy_l_mcp_down_sync": [],
        "energy_r_mcp_down_sync": [],
        "raw_time_diff": [],  # ((t_bar - t_192) - (peak_time - trigger_time))  [ps]
        "raw_phi_diff": [],   # (phi_bar - phi_192) - (phi_peak - phi_trigger)  [ps]
        "phi_trig_diff": [],  # (phi_192 - phi_trigger) [ps]
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
        "phi_r_up_minus_avg_down": [],  # (phi_R_up - phi_avg_down) (ps)
        "energy_l_for_cross": [],       # E_L_up synced with phi_l_up_minus_avg_down
        "energy_r_for_cross": [],       # E_R_up synced with phi_r_up_minus_avg_down
        # ---- strict sync with MCP-coincident loop (up+down+192+MCP complete) ----
        "phi_l_up_minus_avg_down_sync": [],
        "phi_r_up_minus_avg_down_sync": [],
        "energy_l_for_cross_sync": [],
        "energy_r_for_cross_sync": [],
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
