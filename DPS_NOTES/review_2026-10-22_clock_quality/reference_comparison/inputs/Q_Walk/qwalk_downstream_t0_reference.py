"""Build direct Run-4399 MCP-t0 quadratic references for downstream bars."""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_channels import DOWNSTREAM_BAR_CHANNELS
from Q_Walk.qwalk_run4076_safe_time2_scatter import (
    _fit_landau_gauss,
    _histogram_peak_region,
    _limits,
)
from Q_Walk.qwalk_t0_data import (
    load_synchronized_topology_files,
    reconstruct_reference_delta_t,
)
from Q_Walk.qwalk_t0_models import build_comparison_profile, fit_reference_quadratic


DEFAULT_PATTERN = "/eos/user/l/lichengz/MTD/TB2025/sync_out_4399_example/*.root"
DEFAULT_OUTPUT = Path("Q_Walk/results/run4399_downstream_t0_bar8")
DEFAULT_REFERENCE = Path("Q_Walk/reference_quadratic_t0")
UPSTREAM_BAR = 8
DOWNSTREAM_BARS = (2, 3)


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


def _quality_mask(frame):
    q = frame["energy"].to_numpy(dtype=float)
    slope = frame["slope_ua_per_ns"].to_numpy(dtype=float)
    delta_i = frame["delta_i_ua"].to_numpy(dtype=float)
    delta_t12 = frame["delta_t12_ps"].to_numpy(dtype=float)
    timing = frame["delta_t_ref_ps"].to_numpy(dtype=float)
    return (
        np.isfinite(q)
        & (q > 0.0)
        & np.isfinite(slope)
        & (slope > 0.0)
        & np.isfinite(delta_i)
        & (delta_i > 0.0)
        & np.isfinite(delta_t12)
        & (delta_t12 > 0.0)
        & np.isfinite(timing)
    )


def _plot_reference(profile, fit, channel, bar, q_cut, prefix):
    q = profile["energy_mean"].to_numpy(dtype=float)
    y_ns = profile["delta_t_ref_mean_ps"].to_numpy(dtype=float) / 1000.0
    yerr_ns = profile["delta_t_ref_sem_ps"].to_numpy(dtype=float) / 1000.0
    order = np.argsort(q)
    q_low = 1.0 / fit.x_max
    q_high = 1.0 / fit.x_min
    grid_q = np.linspace(q_low, q_high, 400)
    grid_ns = fit.evaluate(1.0 / grid_q) / 1000.0

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.errorbar(
        q[order], y_ns[order], yerr=yerr_ns[order], fmt="o", color="black",
        capsize=3, label="20-bin t0 profile mean +/- SEM",
    )
    axis.plot(grid_q, grid_ns, color="#d62728", linewidth=2.2, label="Weighted quadratic t0 fit")
    axis.axvline(q_cut, color="#f58518", linewidth=1.5, linestyle="--", label="Peak-region left boundary")
    axis.set_xlabel("Q [ADC]")
    axis.set_ylabel("Direct t0 residual [ns]")
    axis.set_title(f"Run 4399, downstream Bar {bar}, channel {channel}: direct MCP-t0 walk")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    figure.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def run(root_pattern=DEFAULT_PATTERN, output_dir=DEFAULT_OUTPUT, reference_dir=DEFAULT_REFERENCE, profile_bins=20):
    paths = sorted((Path(path) for path in glob.glob(str(root_pattern))), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"no files match {root_pattern!r}")
    output_dir = Path(output_dir)
    reference_dir = Path(reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input_pattern": str(root_pattern),
        "input_file_count": len(paths),
        "time2_mode": "safe",
        "upstream_coincidence_bar": UPSTREAM_BAR,
        "upstream_coincidence_channels": [137, 150],
        "event_selection": "exact five-hit topology: Bar 8 {137,150}, target downstream bar pair, and channel 192; matched MCP event",
        "quality_selection": "finite positive Q, slope, Delta I, delta_t12, and direct t0 residual",
        "q_selection": "Q >= lower half-maximum boundary of the prompt Q spectrum",
        "profile": {"bins": int(profile_bins), "timing_clip": "per-bin 5 MAD", "error_bar": "SEM"},
        "channels": {},
    }

    for bar in DOWNSTREAM_BARS:
        target_channels = DOWNSTREAM_BAR_CHANNELS[bar]
        matched = load_synchronized_topology_files(paths, UPSTREAM_BAR, bar)
        aligned = reconstruct_reference_delta_t(matched)
        for channel in target_channels:
            channel_frame = aligned.loc[aligned["channelID"].eq(channel)].copy()
            quality = _quality_mask(channel_frame)
            prompt = channel_frame.loc[quality].copy()
            if len(prompt) < profile_bins * 5:
                raise ValueError(f"channel {channel} has too few prompt hits: {len(prompt)}")
            prompt_q = prompt["energy"].to_numpy(dtype=float)
            peak_limits = _limits(prompt_q, 0.5, 99.5, 20.0)
            peak = _histogram_peak_region(prompt_q, peak_limits)
            landau_gauss = _fit_landau_gauss(prompt_q, peak)
            q_cut = float(peak["half_max_q_low_adc"])
            selected = prompt.loc[prompt["energy"].ge(q_cut)].copy()
            profile = build_comparison_profile(selected, n_bins=profile_bins, min_entries=5)
            if len(profile) < 3:
                raise ValueError(f"channel {channel} has fewer than three t0 profile bins")
            fit = fit_reference_quadratic(profile)
            coefficients_ns = {
                "a": float(fit.coefficients[2] / 1000.0),
                "b": float(fit.coefficients[1] / 1000.0),
                "c": float(fit.coefficients[0] / 1000.0),
            }
            covariance_ns2 = (fit.covariance / 1_000_000.0)[::-1, ::-1]
            stem = f"run4399_bar8_downbar{bar}_ch{channel}_t0_quadratic"
            profile_path = reference_dir / f"{stem}_profile.csv"
            metadata_path = reference_dir / f"{stem}.json"
            profile.to_csv(profile_path, index=False)
            metadata = {
                "channel": int(channel),
                "downstream_bar": int(bar),
                "upstream_bar": UPSTREAM_BAR,
                "formula": "walk_ns(Q) = a + b / Q + c / Q^2",
                "coefficients_ns": coefficients_ns,
                "covariance_ns2_abc": covariance_ns2.tolist(),
                "fit_method": "weighted quadratic fit to 20 inverse-Q profile means with weights 1/SEM^2",
                "fit_domain": {
                    "inv_q_min_per_adc": float(fit.x_min),
                    "inv_q_max_per_adc": float(fit.x_max),
                    "q_min_adc": float(1.0 / fit.x_max),
                    "q_max_adc": float(1.0 / fit.x_min),
                },
                "selection": {
                    "topology_matched_events": int(matched["event_id"].nunique()),
                    "channel_hits": int(len(channel_frame)),
                    "quality_hits": int(len(prompt)),
                    "q_cut_adc": q_cut,
                    "selected_hits": int(len(selected)),
                    "profile_bins_retained": int(len(profile)),
                },
                "landau_gauss_fit": {
                    key: value for key, value in landau_gauss.items()
                    if key not in {"grid", "conditional_density", "signal_conditional_density"}
                },
                "profile_csv": str(profile_path),
            }
            with metadata_path.open("w", encoding="utf-8") as stream:
                json.dump(_native(metadata), stream, indent=2)
                stream.write("\n")
            figure_prefix = output_dir / stem
            _plot_reference(profile, fit, channel, bar, q_cut, figure_prefix)
            report["channels"][str(channel)] = {
                "metadata": str(metadata_path),
                "profile": str(profile_path),
                "figures": [str(figure_prefix.with_suffix(ext)) for ext in (".png", ".pdf")],
            }

    with (output_dir / "run4399_downstream_t0_reference_report.json").open("w", encoding="utf-8") as stream:
        json.dump(_native(report), stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_pattern", nargs="?", default=DEFAULT_PATTERN)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE)
    parser.add_argument("--profile-bins", type=int, default=20)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.root_pattern, arguments.out_dir, arguments.reference_dir, arguments.profile_bins), indent=2))


if __name__ == "__main__":
    main()
