#!/usr/bin/env python3
# Author: Licheng Zhang (licheng.zhang@cern.ch)
# Time: Feb-2026
"""
Bar-level timing analysis  (modular version).

Step 1:  Compute  T_L − T_R  for a given lyso bar and plot the 1-D
         distribution with a Gaussian fit.

Usage examples
--------------
  # Single bar, module up, bar 6
  python3 bar_analysis_main.py "*.root" --module up --lyso-bar 6

  # With energy cuts
  python3 bar_analysis_main.py "*.root" --module up --lyso-bar 6 \
      --energy-min 50 --energy-max 500

  # Multiple files, 4 worker processes
  python3 bar_analysis_main.py run1.root run2.root --module up --lyso-bar 6 --workers 4
"""

import argparse
import sys
import os
import math
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
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

from bar_helpers import log
from bar_processing import process_file, process_file_fast
from bar_plotting import (
    plot_t_diff,
    plot_t_diff_segmented,
    plot_t_diff_aligned,
    plot_energy,
    plot_phi_vs_energy,
    plot_phi_l_vs_phi_r,
    plot_correlation_2d,
)


# ──────────────────────────────────────────────────────────────────
# Argument parsing
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
    p.add_argument("--mcp-internal-dt-cut", action="store_true",
                   help="Enable robust MCP internal timing cut using peak_time - trigger_time")
    p.add_argument("--mcp-internal-dt-nmad", type=float, default=3.0,
                   help="Half-window scale for MCP internal timing cut in robust-width units (default: 3.0)")
    p.add_argument("--out-phi-vs-mcp", default=None,
                   help="Output (phi_avg - phi_peak) plot path")
    p.add_argument("--out-phi-vs-mcp-trig", default=None,
                   help="Output (phi_avg - phi_trigger) plot path")
    p.add_argument("--out-raw-time-diff", default=None,
                   help="Output (t_bar - t_192) - (peak_time - trigger_time) plot path")
    p.add_argument("--out-raw-phi-diff", default=None,
                   help="Output (phi_bar - phi_192) - (phi_peak - phi_trigger) plot path")
    p.add_argument("--out-phi-trig-diff", default=None,
                   help="Output (phi_192 - phi_trigger) plot path")
    p.add_argument("--out-phi-vs-trig-corr", default=None,
                   help="Output event-by-event correlation plot: (phi_bar - phi_peak) vs (phi_192 - phi_trigger)")
    p.add_argument("--out-rawphi-vs-trig-corr", default=None,
                   help="Output event-by-event correlation plot: raw_phi_diff vs (phi_192 - phi_trigger)")
    p.add_argument("--phi-vs-mcp-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="(phi_avg - phi_peak) histogram range")
    p.add_argument("--phi-vs-mcp-trig-range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="(phi_avg - phi_trigger) histogram range")
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
                   help="raw_time_diff histogram range (overrides auto ±2σ)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

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
        "mcp_internal_dt_cut": args.mcp_internal_dt_cut,
        "mcp_internal_dt_nmad": args.mcp_internal_dt_nmad,
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
        "t_diff", "phi_diff", "phi_vs_mcp", "phi_vs_mcp_trig", "phi_l_vs_mcp", "phi_r_vs_mcp",
        "phi_l_raw_sync", "phi_r_raw_sync",
        "energy_l_mcp", "energy_r_mcp", "energy_avg_mcp", "raw_time_diff", "raw_phi_diff", "phi_trig_diff",
        "phi_l_vs_mcp_sync", "phi_r_vs_mcp_sync", "energy_l_mcp_sync", "energy_r_mcp_sync",
        "phi_l_vs_mcp_down_sync", "phi_r_vs_mcp_down_sync", "energy_l_mcp_down_sync", "energy_r_mcp_down_sync",
        "phi_diff_down", "raw_phi_diff_down", "energy", "energy_l", "energy_r", "t_avg_diff", "phi_avg_diff",
        "energy_down", "energy_l_down", "energy_r_down", "raw_time_diff_down", "t_diff_down",
        "phi_vs_mcp_down", "phi_l_vs_mcp_down", "phi_r_vs_mcp_down", 
        "energy_l_mcp_down", "energy_r_mcp_down", "energy_avg_mcp_down",
        "phi_avg_diff_sync_e_up", "phi_avg_diff_sync_e_down",
        "phi_l_up_minus_avg_down", "phi_r_up_minus_avg_down", "energy_l_for_cross", "energy_r_for_cross",
        "phi_l_up_minus_avg_down_sync", "phi_r_up_minus_avg_down_sync",
        "energy_l_for_cross_sync", "energy_r_for_cross_sync",
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
    sig_phi_up_down_calib = None
    sig_phi_trig = None
    sig_phi_up_mcp_perchan = None
    sig_phi_down_mcp_perchan = None

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
                    f"$\\phi_{{avg}}^\\star - \\phi_{{peak}}$  ({label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                    xlabel="$\\phi_{avg}^\\star - \\phi_{peak}$  (ps)",
                    color="steelblue")

        out_mcp_trig = args.out_phi_vs_mcp_trig or f"phi_vs_mcp_trig_{args.module}_bar{args.lyso_bar}.png"
        log(f"phi_vs_mcp_trig entries: {len(merged['phi_vs_mcp_trig'])}")
        sig_phi_up_mcp_trig = plot_t_diff(merged["phi_vs_mcp_trig"], out_mcp_trig,
                    f"$\\phi_{{avg}}^\\star - \\phi_{{trigger}}$  ({label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_vs_mcp_trig_range) if args.phi_vs_mcp_trig_range else None,
                    xlabel="$\\phi_{avg}^\\star - \\phi_{trigger}$  (ps)",
                    color="steelblue")

        out_phi_lr = f"phi_l_vs_phi_r_{args.module}_bar{args.lyso_bar}.png"
        plot_phi_l_vs_phi_r(
            merged["phi_l_raw_sync"], merged["phi_r_raw_sync"],
            out_phi_lr,
            f"$\\phi_L$ vs $\\phi_R$  ({label})")

        # Δ2: double-difference (trigger/common-mode cancelation)
        out_raw_phi = args.out_raw_phi_diff or f"raw_phi_diff_{args.module}_bar{args.lyso_bar}.png"
        sig_phi_raw_diff = plot_t_diff(
            merged["raw_phi_diff"], out_raw_phi,
            f"$\\Delta_2 = (\\phi_{{bar}} - \\phi_{{192}}) - (\\phi_{{peak}} - \\phi_{{trigger}})$  ({label})",
            nbins=args.nbins,
            hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
            xlabel="$\\Delta_2$  (ps)",
            color="slategray")

        # Δ_trig: trigger vs MCP trigger phase
        out_phi_trig = args.out_phi_trig_diff or f"phi_trig_diff_{args.module}_bar{args.lyso_bar}.png"
        sig_phi_trig = plot_t_diff(
            merged["phi_trig_diff"], out_phi_trig,
            f"$\\Delta_{{trig}} = \\phi_{{192}} - \\phi_{{trigger}}$  ({label})",
            nbins=args.nbins,
            hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
            xlabel="$\\Delta_{trig}$  (ps)",
            color="slateblue")

        out_phi_vs_trig_corr = args.out_phi_vs_trig_corr or f"phi_vs_trig_corr_{args.module}_bar{args.lyso_bar}.png"
        plot_correlation_2d(
            merged["phi_trig_diff"], merged["phi_vs_mcp"],
            out_phi_vs_trig_corr,
            f"$\\phi_{{avg}}^\\star - \\phi_{{peak}}$ vs $\\Delta_{{trig}}$  ({label})",
            xlabel="$\\Delta_{trig} = \\phi_{192} - \\phi_{trigger}$  (ps)",
            ylabel="$\\phi_{avg}^\\star - \\phi_{peak}$  (ps)")

        out_phi_vs_trig_corr_zoom = f"phi_vs_trig_corr_zoom_{args.module}_bar{args.lyso_bar}.png"
        plot_correlation_2d(
            merged["phi_trig_diff"], merged["phi_vs_mcp"],
            out_phi_vs_trig_corr_zoom,
            f"$\\phi_{{avg}}^\\star - \\phi_{{peak}}$ vs $\\Delta_{{trig}}$ (Zoom)  ({label})",
            xlabel="$\\Delta_{trig} = \\phi_{192} - \\phi_{trigger}$  (ps)",
            ylabel="$\\phi_{avg}^\\star - \\phi_{peak}$  (ps)",
            hist_range=[[200, 800], [-600, 0]])

        out_rawphi_vs_trig_corr = args.out_rawphi_vs_trig_corr or f"rawphi_vs_trig_corr_{args.module}_bar{args.lyso_bar}.png"
        plot_correlation_2d(
            merged["phi_trig_diff"], merged["raw_phi_diff"],
            out_rawphi_vs_trig_corr,
            f"$\\Delta_2$ vs $\\Delta_{{trig}}$  ({label})",
            xlabel="$\\Delta_{trig} = \\phi_{192} - \\phi_{trigger}$  (ps)",
            ylabel="$\\Delta_2$  (ps)")

        if sig_phi_up_mcp and sig_phi_raw_diff:
            ratio = sig_phi_raw_diff / sig_phi_up_mcp if sig_phi_up_mcp else float("nan")
            log(f"Width check: sigma(Δ1)={sig_phi_up_mcp:.2f} ps, sigma(Δ2)={sig_phi_raw_diff:.2f} ps, ratio={ratio:.3f}")
        if sig_phi_trig:
            log(f"Trigger phase width: sigma(Δ_trig)={sig_phi_trig:.2f} ps")

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

        # ---- Per-channel calibrated bar average ----
        if a_l is not None and b_l is not None and c_l is not None \
           and a_r is not None and b_r is not None and c_r is not None:
            phi_bar_perchan = []
            for pl, pr, el_s, er_s in zip(merged["phi_l_vs_mcp_sync"], merged["phi_r_vs_mcp_sync"],
                                           merged["energy_l_mcp_sync"], merged["energy_r_mcp_sync"]):
                if np.isfinite(pl) and np.isfinite(pr) and np.isfinite(el_s) and np.isfinite(er_s):
                    pl_c = pl - (a_l * el_s**2 + b_l * el_s + c_l)
                    pr_c = pr - (a_r * er_s**2 + b_r * er_s + c_r)
                    phi_bar_perchan.append((pl_c + pr_c) / 2.0)
            out_perchan = f"phi_bar_perchan_calib_{args.module}_bar{args.lyso_bar}.png"
            sig_phi_up_mcp_perchan = plot_t_diff(
                phi_bar_perchan, out_perchan,
                f"Per-ch calib $\\langle \\phi \\rangle - \\phi_{{peak}}$  ({label})",
                nbins=args.nbins,
                hist_range=tuple(args.phi_vs_mcp_range) if args.phi_vs_mcp_range else None,
                xlabel="Per-ch calibrated $\\langle \\phi \\rangle - \\phi_{peak}$  (ps)",
                color="darkorange")
            log(f"Per-channel calibrated bar sigma: {sig_phi_up_mcp_perchan} ps")
            if sig_phi_up_mcp and sig_phi_up_mcp_perchan:
                log(f"Comparison — Raw: {sig_phi_up_mcp:.2f}ps | Per-channel Calibrated: {sig_phi_up_mcp_perchan:.2f}ps")

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
                    
        # ---- UP channel(s) - DOWN AVG cross-module plot ----
        # In MCP mode, use strict sync arrays populated in the same MCP-coincident loop.
        if args.mcp and len(merged["phi_l_up_minus_avg_down_sync"]) > 0:
            cross_phi_l = merged["phi_l_up_minus_avg_down_sync"]
            cross_phi_r = merged["phi_r_up_minus_avg_down_sync"]
            cross_e_l = merged["energy_l_for_cross_sync"]
            cross_e_r = merged["energy_r_for_cross_sync"]
        else:
            cross_phi_l = merged["phi_l_up_minus_avg_down"]
            cross_phi_r = merged["phi_r_up_minus_avg_down"]
            cross_e_l = merged["energy_l_for_cross"]
            cross_e_r = merged["energy_r_for_cross"]

        log(f"phi_l_up_minus_avg_down entries: {len(cross_phi_l)}")
        out_phi_l_up_down = f"phi_l_up_minus_avg_down_bar{args.lyso_bar}_bar{down_bar_label}.png"
        sig_phi_l_up_down = plot_t_diff(cross_phi_l, out_phi_l_up_down,
                    f"$\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                    xlabel="$\\phi_{L, up} - \\langle \\phi \\rangle_{down}$  (ps)",
                    color="darkcyan")
                    
        out_phi_l_up_down_vs_e = f"phi_l_up_minus_avg_down_vs_e_l_up_bar{args.lyso_bar}_bar{down_bar_label}.png"
        a_cross_l, b_cross_l, c_cross_l = plot_phi_vs_energy(cross_phi_l, cross_e_l,
                            out_phi_l_up_down_vs_e, f"$\\phi_{{L, up}} - \\langle \\phi \\rangle_{{down}}$ vs $E_{{L, up}}$  ({dual_label})",
                            ylabel="$\\phi_{L, up} - \\langle \\phi \\rangle_{down}$ (ps)")

        out_phi_r_up_down_vs_e = f"phi_r_up_minus_avg_down_vs_e_r_up_bar{args.lyso_bar}_bar{down_bar_label}.png"
        a_cross_r, b_cross_r, c_cross_r = plot_phi_vs_energy(cross_phi_r, cross_e_r,
                            out_phi_r_up_down_vs_e, f"$\\phi_{{R, up}} - \\langle \\phi \\rangle_{{down}}$ vs $E_{{R, up}}$  ({dual_label})",
                            ylabel="$\\phi_{R, up} - \\langle \\phi \\rangle_{down}$ (ps)")
                            
        if a_cross_l is not None and b_cross_l is not None and c_cross_l is not None:
            calib_phi_cross = []
            calib_e_cross = []
            for p, e in zip(cross_phi_l, cross_e_l):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_cross.append(p - (a_cross_l * e**2 + b_cross_l * e + c_cross_l))
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

        if a_cross_r is not None and b_cross_r is not None and c_cross_r is not None:
            calib_phi_cross_r = []
            calib_e_cross_r = []
            for p, e in zip(cross_phi_r, cross_e_r):
                if np.isfinite(p) and np.isfinite(e):
                    calib_phi_cross_r.append(p - (a_cross_r * e**2 + b_cross_r * e + c_cross_r))
                    calib_e_cross_r.append(e)
            
            out_phi_cross_calib_r = f"phi_r_up_minus_avg_down_calibrated_bar{args.lyso_bar}_bar{down_bar_label}.png"
            plot_t_diff(calib_phi_cross_r, out_phi_cross_calib_r,
                        f"Calib $\\phi_{{R, up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                        nbins=args.nbins,
                        hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                        xlabel="Calib $\\phi_{R, up} - \\langle \\phi \\rangle_{down}$  (ps)",
                        color="teal")
            
            out_phi_cross_vs_e_calib_r = f"phi_r_up_minus_avg_down_vs_e_r_up_calibrated_bar{args.lyso_bar}_bar{down_bar_label}.png"
            plot_phi_vs_energy(calib_phi_cross_r, calib_e_cross_r,
                               out_phi_cross_vs_e_calib_r, f"Calib $\\phi_{{R, up}} - \\langle \\phi \\rangle_{{down}}$ vs $E_{{R, up}}$  ({dual_label})",
                               ylabel="Calib $\\phi_{R, up} - \\langle \\phi \\rangle_{down}$ (ps)", fit_poly=False)

        # ---- combined L/R calibrated up-down difference ----
        if (a_cross_l is not None and b_cross_l is not None and c_cross_l is not None and
            a_cross_r is not None and b_cross_r is not None and c_cross_r is not None):
            calib_up_down = []
            for pl, pr, el, er in zip(
                cross_phi_l, cross_phi_r, cross_e_l, cross_e_r
            ):
                if np.isfinite(pl) and np.isfinite(pr) and np.isfinite(el) and np.isfinite(er):
                    pl_c = pl - (a_cross_l * el**2 + b_cross_l * el + c_cross_l)
                    pr_c = pr - (a_cross_r * er**2 + b_cross_r * er + c_cross_r)
                    calib_up_down.append((pl_c + pr_c) / 2.0)
            out_phiavg_calib = f"phi_avg_up_minus_down_calibrated_bar{args.lyso_bar}_bar{down_bar_label}.png"
            sig_phi_up_down_calib = plot_t_diff(
                calib_up_down, out_phiavg_calib,
                f"Calib $\\langle \\phi \\rangle_{{up}} - \\langle \\phi \\rangle_{{down}}$  ({dual_label})",
                nbins=args.nbins,
                hist_range=tuple(args.t_avg_diff_range) if args.t_avg_diff_range else None,
                xlabel="Calib $\\langle \\phi \\rangle_{up} - \\langle \\phi \\rangle_{down}$  (ps)",
                color="cadetblue")

        # ---- down module phi_diff ----
        log(f"phi_diff_down entries: {len(merged['phi_diff_down'])}")
        out_phidiff_down = f"phi_diff_{down_mod_name}_bar{down_bar_label}.png"
        plot_t_diff(merged["phi_diff_down"], out_phidiff_down,
                    f"$\\phi_L - \\phi_R$  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=tuple(args.phi_diff_range) if args.phi_diff_range else None,
                    xlabel="$\\phi_L - \\phi_R$  (ps)")

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

            # ---- Per-channel calibrated bar average (down bar) ----
            if a_l_down is not None and b_l_down is not None and c_l_down is not None \
               and a_r_down is not None and b_r_down is not None and c_r_down is not None:
                phi_bar_perchan_down = []
                for pl, pr, el_s, er_s in zip(merged["phi_l_vs_mcp_down_sync"], merged["phi_r_vs_mcp_down_sync"],
                                               merged["energy_l_mcp_down_sync"], merged["energy_r_mcp_down_sync"]):
                    if np.isfinite(pl) and np.isfinite(pr) and np.isfinite(el_s) and np.isfinite(er_s):
                        pl_c = pl - (a_l_down * el_s**2 + b_l_down * el_s + c_l_down)
                        pr_c = pr - (a_r_down * er_s**2 + b_r_down * er_s + c_r_down)
                        phi_bar_perchan_down.append((pl_c + pr_c) / 2.0)
                out_perchan_down = f"phi_bar_perchan_calib_{down_mod_name}_bar{down_bar_label}.png"
                sig_phi_down_mcp_perchan = plot_t_diff(
                    phi_bar_perchan_down, out_perchan_down,
                    f"Per-ch calib $\\langle \\phi \\rangle - \\phi_{{peak}}$  (module {down_mod_name} bar {down_bar_label})",
                    nbins=args.nbins,
                    hist_range=phi_mcp_down_range,
                    xlabel="Per-ch calibrated $\\langle \\phi \\rangle - \\phi_{peak}$  (ps)",
                    color="darkorange")
                log(f"Down bar per-channel calibrated sigma: {sig_phi_down_mcp_perchan} ps")

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

    if sig_phi_up_down_calib and sig_phi_up_mcp_perchan and sig_phi_down_mcp_perchan:
        print("\n" + "="*50)
        print("  CALIBRATED ABSOLUTE TIMING RESOLUTION CALCULATION")
        print("="*50)
        print(f"Post-Calibration Measured sigmas (Per-Channel):")
        print(f"  sigma(up - down)_{{calib}}   = {sig_phi_up_down_calib:.2f} ps")
        print(f"  sigma(up - MCP)_{{calib}}    = {sig_phi_up_mcp_perchan:.2f} ps")
        print(f"  sigma(down - MCP)_{{calib}}  = {sig_phi_down_mcp_perchan:.2f} ps")
        
        v1_calib = sig_phi_up_down_calib**2
        v2_calib = sig_phi_up_mcp_perchan**2
        v3_calib = sig_phi_down_mcp_perchan**2
        
        var_up_calib = 0.5 * (v1_calib + v2_calib - v3_calib)
        var_down_calib = 0.5 * (v1_calib + v3_calib - v2_calib)
        var_mcp_calib = 0.5 * (v2_calib + v3_calib - v1_calib)
        
        print("\nCalculated intrinsic resolutions:")
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

    if sig_phi_up_down and sig_phi_up_mcp and sig_phi_down_mcp and sig_phi_trig:
        print("\n" + "="*50)
        print("  TRIGGER-SUBTRACTED ABSOLUTE TIMING RESOLUTION")
        print("="*50)
        print("Measured sigmas:")
        print(f"  sigma(up - down)      = {sig_phi_up_down:.2f} ps   [unchanged]")
        print(f"  sigma(up - MCP)       = {sig_phi_up_mcp:.2f} ps")
        print(f"  sigma(down - MCP)     = {sig_phi_down_mcp:.2f} ps")
        print(f"  sigma(Delta_trig)     = {sig_phi_trig:.2f} ps")

        v_up_mcp_sub = sig_phi_up_mcp**2 - sig_phi_trig**2
        v_down_mcp_sub = sig_phi_down_mcp**2 - sig_phi_trig**2

        if v_up_mcp_sub > 0:
            print(f"  sigma(up - MCP)_sub   = sqrt({sig_phi_up_mcp:.2f}^2 - {sig_phi_trig:.2f}^2) = {math.sqrt(v_up_mcp_sub):.2f} ps")
        else:
            print("  sigma(up - MCP)_sub   = [imaginary] (variance < 0)")

        if v_down_mcp_sub > 0:
            print(f"  sigma(down - MCP)_sub = sqrt({sig_phi_down_mcp:.2f}^2 - {sig_phi_trig:.2f}^2) = {math.sqrt(v_down_mcp_sub):.2f} ps")
        else:
            print("  sigma(down - MCP)_sub = [imaginary] (variance < 0)")

        if v_up_mcp_sub > 0 and v_down_mcp_sub > 0:
            v1_sub = sig_phi_up_down**2
            v2_sub = v_up_mcp_sub
            v3_sub = v_down_mcp_sub

            var_up_sub = 0.5 * (v1_sub + v2_sub - v3_sub)
            var_down_sub = 0.5 * (v1_sub + v3_sub - v2_sub)
            var_mcp_sub = 0.5 * (v2_sub + v3_sub - v1_sub)

            print("\nCalculated intrinsic resolutions:")
            if var_up_sub > 0:
                print(f"  sigma(up bar)_sub      = {math.sqrt(var_up_sub):.2f} ps")
            else:
                print("  sigma(up bar)_sub      = [imaginary] (variance < 0)")

            if var_down_sub > 0:
                print(f"  sigma(down bar)_sub    = {math.sqrt(var_down_sub):.2f} ps")
            else:
                print("  sigma(down bar)_sub    = [imaginary] (variance < 0)")

            if var_mcp_sub > 0:
                print(f"  sigma(MCP)_sub         = {math.sqrt(var_mcp_sub):.2f} ps")
            else:
                print("  sigma(MCP)_sub         = [imaginary] (variance < 0)")
        print("="*50 + "\n")

    if sig_phi_up_down_calib and sig_phi_up_mcp_perchan and sig_phi_down_mcp_perchan and sig_phi_trig:
        print("\n" + "="*50)
        print("  CALIBRATED TRIGGER-SUBTRACTED ABSOLUTE RESOLUTION")
        print("="*50)
        print("Measured sigmas:")
        print(f"  sigma(up - down)_{{calib}}   = {sig_phi_up_down_calib:.2f} ps   [unchanged]")
        print(f"  sigma(up - MCP)_{{calib}}    = {sig_phi_up_mcp_perchan:.2f} ps")
        print(f"  sigma(down - MCP)_{{calib}}  = {sig_phi_down_mcp_perchan:.2f} ps")
        print(f"  sigma(Delta_trig)           = {sig_phi_trig:.2f} ps")

        v_up_mcp_cal_sub = sig_phi_up_mcp_perchan**2 - sig_phi_trig**2
        v_down_mcp_cal_sub = sig_phi_down_mcp_perchan**2 - sig_phi_trig**2

        if v_up_mcp_cal_sub > 0:
            print(f"  sigma(up - MCP)_{{sub}}     = sqrt({sig_phi_up_mcp_perchan:.2f}^2 - {sig_phi_trig:.2f}^2) = {math.sqrt(v_up_mcp_cal_sub):.2f} ps")
        else:
            print("  sigma(up - MCP)_sub      = [imaginary] (variance < 0)")

        if v_down_mcp_cal_sub > 0:
            print(f"  sigma(down - MCP)_{{sub}}   = sqrt({sig_phi_down_mcp_perchan:.2f}^2 - {sig_phi_trig:.2f}^2) = {math.sqrt(v_down_mcp_cal_sub):.2f} ps")
        else:
            print("  sigma(down - MCP)_sub    = [imaginary] (variance < 0)")

        if v_up_mcp_cal_sub > 0 and v_down_mcp_cal_sub > 0:
            v1_cal_sub = sig_phi_up_down_calib**2
            v2_cal_sub = v_up_mcp_cal_sub
            v3_cal_sub = v_down_mcp_cal_sub

            var_up_cal_sub = 0.5 * (v1_cal_sub + v2_cal_sub - v3_cal_sub)
            var_down_cal_sub = 0.5 * (v1_cal_sub + v3_cal_sub - v2_cal_sub)
            var_mcp_cal_sub = 0.5 * (v2_cal_sub + v3_cal_sub - v1_cal_sub)

            print("\nCalculated intrinsic resolutions:")
            if var_up_cal_sub > 0:
                print(f"  sigma(up bar)_sub        = {math.sqrt(var_up_cal_sub):.2f} ps")
            else:
                print("  sigma(up bar)_sub        = [imaginary] (variance < 0)")

            if var_down_cal_sub > 0:
                print(f"  sigma(down bar)_sub      = {math.sqrt(var_down_cal_sub):.2f} ps")
            else:
                print("  sigma(down bar)_sub      = [imaginary] (variance < 0)")

            if var_mcp_cal_sub > 0:
                print(f"  sigma(MCP)_sub           = {math.sqrt(var_mcp_cal_sub):.2f} ps")
            else:
                print("  sigma(MCP)_sub           = [imaginary] (variance < 0)")
        print("="*50 + "\n")

    log("done")


if __name__ == "__main__":
    main()
