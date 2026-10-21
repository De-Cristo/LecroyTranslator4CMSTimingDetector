"""Build Run-4399 direct MCP-t0 quadratic references for extended Q-walk channels."""

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_channels import (
    DOWNSTREAM_BAR_CHANNELS,
    TRIGGER_CHANNEL,
    UPSTREAM_BAR_CHANNELS,
    bar_pair_for_channel,
    exact_four_channel_topology,
    reference_topology,
)
from Q_Walk.qwalk_downstream_t0_reference import _quality_mask
from Q_Walk.qwalk_run4076_safe_time2_scatter import _histogram_peak_region, _limits
from Q_Walk.qwalk_t0_data import (
    _read_mcp_events,
    _read_tofhir_hits,
    match_mcp_events,
    reconstruct_reference_delta_t,
)
from Q_Walk.qwalk_t0_models import build_comparison_profile, fit_reference_quadratic


DEFAULT_PATTERN = "/eos/user/l/lichengz/MTD/TB2025/sync_out_4399_example/*.root"
DEFAULT_OUTPUT = Path("Q_Walk/results/run4399_extended_t0_reference_v1")
DEFAULT_REFERENCE = Path("Q_Walk/reference_quadratic_t0_extended_v1")
UPSTREAM_CHANNELS = (130, 133, 134, 135, 136, 137, 139, 149, 150, 151, 154, 155, 156, 159)
DOWNSTREAM_CHANNELS = (239, 240, 238, 242, 236, 244, 235, 245)
UPSTREAM_COINCIDENCE_PAIR = (234, 243)
DOWNSTREAM_COINCIDENCE_PAIR = (137, 150)


def _natural_key(path):
    parts = re.split(r"(\d+)", Path(path).name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts), str(path)


def _native(value):
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_topology_specs():
    """Return one exact five-hit topology specification per requested bar."""
    specs = {}
    for module, channels in (
        ("upstream", UPSTREAM_CHANNELS),
        ("downstream", DOWNSTREAM_CHANNELS),
    ):
        requested_bars = sorted({
            bar for bar, pair in (UPSTREAM_BAR_CHANNELS if module == "upstream" else DOWNSTREAM_BAR_CHANNELS).items()
            if any(channel in pair for channel in channels)
        })
        for bar in requested_bars:
            pair = (UPSTREAM_BAR_CHANNELS if module == "upstream" else DOWNSTREAM_BAR_CHANNELS)[bar]
            selected_channels = tuple(channel for channel in pair if channel in channels)
            if not selected_channels:
                continue
            if module == "upstream":
                topology = reference_topology(bar)
                coincidence_pair = UPSTREAM_COINCIDENCE_PAIR
            else:
                topology = exact_four_channel_topology(8, bar)
                coincidence_pair = DOWNSTREAM_COINCIDENCE_PAIR
            key = f"{module}_bar{bar}"
            specs[key] = {
                "module": module,
                "bar": int(bar),
                "pair": tuple(map(int, pair)),
                "channels": selected_channels,
                "coincidence_pair": coincidence_pair,
                "topology": frozenset(topology),
            }
    return specs


def _matched_frames_from_file(path, specs):
    with uproot.open(path) as root_file:
        hits = _read_tofhir_hits(root_file["data"])
        mcp = _read_mcp_events(root_file["MCP"])
    topologies = hits.groupby("entry")["channelID"].agg(
        hit_count="size", channel_set=lambda values: frozenset(map(int, values))
    )
    frames = {}
    file_id = Path(path).stem
    for key, spec in specs.items():
        topology = spec["topology"]
        entries = topologies.index[
            topologies["hit_count"].eq(len(topology))
            & topologies["channel_set"].eq(topology)
        ]
        if len(entries) == 0:
            continue
        events = hits.loc[hits["entry"].isin(entries)]
        ch192 = events.loc[
            events["channelID"].eq(TRIGGER_CHANNEL), ["entry", "time"]
        ].rename(columns={"time": "ch192_time"})
        signal = events.loc[events["channelID"].ne(TRIGGER_CHANNEL)].merge(
            ch192, on="entry", how="inner", validate="many_to_one"
        )
        matched = match_mcp_events(signal, mcp)
        if matched.empty:
            continue
        matched["file_id"] = file_id
        matched["event_id"] = matched["entry"].map(lambda entry: f"{file_id}:{entry}")
        matched["bar"] = spec["bar"]
        frames[key] = matched
    return frames


def _plot_reference(profile, fit, channel, module, bar, q_cut, prefix):
    q = profile["energy_mean"].to_numpy(dtype=float)
    y_ns = profile["delta_t_ref_mean_ps"].to_numpy(dtype=float) / 1000.0
    yerr_ns = profile["delta_t_ref_sem_ps"].to_numpy(dtype=float) / 1000.0
    order = np.argsort(q)
    grid_q = np.linspace(1.0 / fit.x_max, 1.0 / fit.x_min, 400)
    grid_ns = fit.evaluate(1.0 / grid_q) / 1000.0
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.errorbar(q[order], y_ns[order], yerr=yerr_ns[order], fmt="o", color="black", capsize=3, label="20-bin t0 profile mean +/- SEM")
    axis.plot(grid_q, grid_ns, color="#d62728", linewidth=2.2, label="Weighted quadratic t0 fit")
    axis.axvline(q_cut, color="#f58518", linewidth=1.5, linestyle="--", label="Peak-region left boundary")
    axis.set_xlabel("Q [ADC]")
    axis.set_ylabel("Direct t0 residual [ns]")
    axis.set_title(f"Run 4399, {module} Bar {bar}, channel {channel}: direct MCP-t0 walk")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    figure.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def selected_inv_q_edges(energy, n_bins):
    """Return uniform inverse-Q bins spanning the complete selected Q range."""
    values = np.asarray(energy, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("selected Q range is empty")
    q_min = float(np.min(values))
    q_max = float(np.max(values))
    if not q_min < q_max:
        raise ValueError("selected Q range must be nonzero")
    return np.linspace(1.0 / q_max, 1.0 / q_min, int(n_bins) + 1)


def run(root_pattern=DEFAULT_PATTERN, output_dir=DEFAULT_OUTPUT, reference_dir=DEFAULT_REFERENCE, profile_bins=20,
        workers=1, cache_dir=None):
    paths = sorted((Path(path) for path in glob.glob(str(root_pattern))), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"no files match {root_pattern!r}")
    output_dir = Path(output_dir)
    reference_dir = Path(reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    specs = build_topology_specs()
    collected = defaultdict(list)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames = pool.map(partial(_matched_frames_from_file, specs=specs), paths)
        for index, (path, result) in enumerate(zip(paths, frames)):
            for key, frame in result.items():
                collected[key].append(frame)
            print(f"Read {index + 1}/{len(paths)}: {path.name}", flush=True)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input_pattern": str(root_pattern),
        "input_file_count": len(paths),
        "input_files": [{"path": str(p), "size_bytes": p.stat().st_size,
                         "mtime_ns": p.stat().st_mtime_ns} for p in paths],
        "clock_quality": "finite-trigger segmentation; whole-group pre-LOWESS scaled MAD <= 1000 ps; no row shifts",
        "time2_mode": "safe",
        "event_selection": "exact five-hit topology with channel 192, matched MCP event",
        "quality_selection": "finite positive Q, slope, Delta I, delta_t12, and direct t0 residual",
        "q_selection": "Q >= lower half-maximum boundary of the prompt Q spectrum",
        "profile": {"bins": int(profile_bins), "timing_clip": "per-bin 5 MAD", "error_bar": "SEM"},
        "channels": {},
    }
    for key, spec in specs.items():
        if key not in collected:
            for channel in spec["channels"]:
                report["channels"][str(channel)] = {"status": "failed", "reason": "no matched topology events"}
            continue
        matched = pd.concat(collected[key], ignore_index=True)
        aligned = reconstruct_reference_delta_t(matched)
        if cache_dir is not None:
            aligned.to_csv(cache_dir / f"{key}_aligned.csv.gz", index=False)
        audit = aligned.drop_duplicates("event_id").groupby(["file_id", "segment"]).agg(
            events=("event_id", "size"), scaled_mad_ps=("clock_line_scaled_mad_ps", "first"),
            line_slope=("ch192_line_slope", "first"), status=("reference_alignment_status", "first"))
        audit.to_csv(output_dir / f"{key}_clock_groups.csv")
        for channel in spec["channels"]:
            try:
                frame = aligned.loc[aligned["channelID"].eq(channel)].copy()
                prompt = frame.loc[_quality_mask(frame)].copy()
                if len(prompt) < profile_bins * 5:
                    raise ValueError(f"too few quality hits: {len(prompt)}")
                prompt_q = prompt["energy"].to_numpy(dtype=float)
                peak = _histogram_peak_region(prompt_q, _limits(prompt_q, 0.5, 99.5, 20.0))
                q_cut = float(peak["half_max_q_low_adc"])
                selected = prompt.loc[prompt["energy"].ge(q_cut)].copy()
                inv_q_edges = selected_inv_q_edges(selected["energy"], profile_bins)
                profile = build_comparison_profile(
                    selected,
                    n_bins=profile_bins,
                    min_entries=5,
                    inv_q_edges=inv_q_edges,
                )
                if len(profile) < 3:
                    raise ValueError(f"fewer than three t0 profile bins: {len(profile)}")
                fit = fit_reference_quadratic(profile)
                profile_q_min = float(1.0 / profile["inv_q_bin_high"].max())
                profile_q_max = float(1.0 / profile["inv_q_bin_low"].min())
                stem = f"run4399_{spec['module']}_bar{spec['bar']}_ch{channel}_t0_quadratic"
                profile_path = reference_dir / f"{stem}_profile.csv"
                metadata_path = reference_dir / f"{stem}.json"
                prefix = output_dir / stem
                profile.to_csv(profile_path, index=False)
                metadata = {
                    "channel": int(channel), "module": spec["module"], "bar": spec["bar"],
                    "pair": list(spec["pair"]), "coincidence_pair": list(spec["coincidence_pair"]),
                    "formula": "walk_ns(Q) = a + b / Q + c / Q^2",
                    "coefficients_ns": {"a": float(fit.coefficients[2] / 1000.0), "b": float(fit.coefficients[1] / 1000.0), "c": float(fit.coefficients[0] / 1000.0)},
                    "covariance_ns2_abc": (fit.covariance / 1_000_000.0)[::-1, ::-1].tolist(),
                    "fit_method": "weighted quadratic fit to 20 full-selected-range inverse-Q profile means with weights 1/SEM^2",
                    "fit_domain": {"q_min_adc": float(1.0 / fit.x_max), "q_max_adc": float(1.0 / fit.x_min)},
                    "profile_domain": {"q_min_adc": profile_q_min, "q_max_adc": profile_q_max},
                    "selection": {"topology_matched_events": int(matched["event_id"].nunique()), "channel_hits": int(len(frame)), "quality_hits": int(len(prompt)), "q_cut_adc": q_cut, "selected_hits": int(len(selected)), "profile_bins_retained": int(len(profile))},
                    "profile_csv": str(profile_path),
                    "clock_quality": report["clock_quality"],
                }
                with metadata_path.open("w", encoding="utf-8") as stream:
                    json.dump(_native(metadata), stream, indent=2)
                    stream.write("\n")
                _plot_reference(profile, fit, channel, spec["module"], spec["bar"], q_cut, prefix)
                report["channels"][str(channel)] = {"status": "ok", "metadata": str(metadata_path), "profile": str(profile_path), "figures": [str(prefix.with_suffix(ext)) for ext in (".png", ".pdf")]}
            except (ValueError, np.linalg.LinAlgError) as error:
                report["channels"][str(channel)] = {"status": "failed", "reason": str(error)}
    with (output_dir / "run4399_extended_t0_reference_report.json").open("w", encoding="utf-8") as stream:
        json.dump(_native(report), stream, indent=2, allow_nan=True)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_pattern", nargs="?", default=DEFAULT_PATTERN)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE)
    parser.add_argument("--profile-bins", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    arguments = parser.parse_args()
    print(json.dumps(_native(run(arguments.root_pattern, arguments.out_dir, arguments.reference_dir, arguments.profile_bins,
                                arguments.workers, arguments.cache_dir)), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
