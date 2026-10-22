"""Run-4076 channel-137 peak-right slope profile using safe time2."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_core import (
    calculate_slope,
    decode_thresholds,
    read_root_hits,
    select_upstream_hits_with_fixed_downstream_bar,
)
from Q_Walk.qwalk_run4076_safe_time2_scatter import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    _fit_landau_gauss,
    _histogram_peak_region,
    _limits,
)


def _build_profile(q, slope, bins=10, edges=None):
    if edges is None:
        edges = np.linspace(q.min(), q.max(), bins + 1)
    else:
        edges = np.asarray(edges, dtype=float)
        bins = len(edges) - 1
    assignments = np.digitize(q, edges) - 1
    assignments[assignments == bins] = bins - 1
    rows = []
    for index in range(bins):
        in_bin = assignments == index
        count = int(in_bin.sum())
        if count == 0:
            continue
        values = slope[in_bin]
        sd = float(np.std(values, ddof=1)) if count > 1 else 0.0
        rows.append(
            {
                "q_low_adc": float(edges[index]),
                "q_high_adc": float(edges[index + 1]),
                "q_mean_adc": float(np.mean(q[in_bin])),
                "slope_mean_ua_per_ns": float(np.mean(values)),
                "slope_sd_ua_per_ns": sd,
                "slope_sem_ua_per_ns": float(sd / np.sqrt(count)) if count > 1 else np.nan,
                "count": count,
            }
        )
    return rows


def _fit_linear_profile(profile):
    q = np.asarray([row["q_mean_adc"] for row in profile], dtype=float)
    slope = np.asarray([row["slope_mean_ua_per_ns"] for row in profile], dtype=float)
    sem = np.asarray([row["slope_sem_ua_per_ns"] for row in profile], dtype=float)
    sem_floor = np.sqrt(np.finfo(float).eps) * max(1.0, float(np.nanmax(np.abs(slope))))
    valid = np.isfinite(q) & np.isfinite(slope) & np.isfinite(sem) & (sem > sem_floor)
    if valid.sum() < 2:
        raise ValueError("fewer than two profile bins with finite slope SEM")
    design = np.column_stack((np.ones(valid.sum()), q[valid]))
    weights = 1.0 / sem[valid] ** 2
    normal = design.T @ (weights[:, None] * design)
    covariance = np.linalg.inv(normal)
    coefficients = covariance @ (design.T @ (weights * slope[valid]))
    return {
        "intercept_ua_per_ns": float(coefficients[0]),
        "coefficient_ua_per_ns_per_adc": float(coefficients[1]),
        "covariance": covariance.tolist(),
        "profile_bins_used": int(valid.sum()),
    }


def run(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_hits = read_root_hits(input_path, time2_mode="safe")
    selected = select_upstream_hits_with_fixed_downstream_bar(raw_hits)
    channel_hits = selected.loc[selected["channelID"].eq(137)].copy()

    _, _, delta_i = decode_thresholds(channel_hits["step2"].to_numpy())
    q = channel_hits["energy"].to_numpy(dtype=float)
    delta_t12_ps = (
        channel_hits["time2"].to_numpy(dtype=float)
        - channel_hits["time"].to_numpy(dtype=float)
    )
    slope = calculate_slope(
        channel_hits["time"].to_numpy(dtype=float),
        channel_hits["time2"].to_numpy(dtype=float),
        delta_i,
    )
    coarse_delta = (
        channel_hits["t2coarse"].to_numpy(dtype=np.int64)
        - channel_hits["t1coarse"].to_numpy(dtype=np.int64)
    ) % 1024
    prompt = (
        np.isfinite(q)
        & np.isfinite(slope)
        & (delta_i > 0.0)
        & (delta_t12_ps > 0.0)
        & (coarse_delta == 0)
    )
    prompt_q = q[prompt]
    prompt_slope = slope[prompt]
    prompt_x_limits = _limits(prompt_q, 0.5, 99.5, 20.0)
    peak = _histogram_peak_region(prompt_q, prompt_x_limits)
    landau_gauss_fit = _fit_landau_gauss(prompt_q, peak)
    q_cut = peak["half_max_q_low_adc"]
    peak_right = prompt & (q >= q_cut)
    fit_q = q[peak_right]
    fit_slope = slope[peak_right]
    profile = _build_profile(fit_q, fit_slope, bins=10)
    linear_fit = _fit_linear_profile(profile)

    scatter_x_limits = _limits(fit_q, 0.5, 99.5, 20.0)
    scatter_y_limits = _limits(fit_slope, 0.5, 99.5, 1.0)
    fig, (scatter_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    scatter_axis.scatter(
        fit_q,
        fit_slope,
        s=9,
        alpha=0.36,
        color="#1f77b4",
        edgecolors="none",
        rasterized=True,
        label="Peak-right prompt hits",
    )
    q_mean = np.asarray([row["q_mean_adc"] for row in profile])
    slope_mean = np.asarray([row["slope_mean_ua_per_ns"] for row in profile])
    slope_sd = np.asarray([row["slope_sd_ua_per_ns"] for row in profile])
    scatter_axis.errorbar(
        q_mean,
        slope_mean,
        yerr=slope_sd,
        fmt="o",
        color="black",
        markersize=5,
        capsize=3,
        label="10-bin profile mean +/- SD",
    )
    q_line = np.linspace(fit_q.min(), fit_q.max(), 300)
    slope_line = (
        linear_fit["intercept_ua_per_ns"]
        + linear_fit["coefficient_ua_per_ns_per_adc"] * q_line
    )
    scatter_axis.plot(q_line, slope_line, color="#d62728", linewidth=2.0, label="Weighted linear profile fit")
    scatter_axis.set_xlim(*scatter_x_limits)
    scatter_axis.set_ylim(6.0, 25.0)
    scatter_axis.set_ylabel("Slope [uA/ns]")
    scatter_axis.set_title("Run 4076, channel 137: prompt peak-right slope profile with safe time2")
    scatter_axis.text(
        0.02,
        0.98,
        "Q >= {q_cut:.1f} ADC\nSelected: {count:,}\n"
        "S(Q) = {intercept:.3f} {sign} {magnitude:.5f} Q".format(
            q_cut=q_cut,
            count=len(fit_q),
            intercept=linear_fit["intercept_ua_per_ns"],
            sign="+" if linear_fit["coefficient_ua_per_ns_per_adc"] >= 0.0 else "-",
            magnitude=abs(linear_fit["coefficient_ua_per_ns_per_adc"]),
        ),
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    scatter_axis.grid(True, alpha=0.25)
    scatter_axis.legend(loc="lower right", fontsize=8)

    spectrum_axis.hist(
        prompt_q,
        bins=peak["edges"],
        histtype="stepfilled",
        color="#4c78a8",
        alpha=0.82,
    )
    bin_width = peak["edges"][1] - peak["edges"][0]
    expected_counts = (
        landau_gauss_fit["conditional_density"]
        * landau_gauss_fit["fit_event_count"]
        * bin_width
    )
    spectrum_axis.plot(
        landau_gauss_fit["grid"],
        expected_counts,
        color="#2ca02c",
        linewidth=2.0,
        label="Landau-Gaussian fit",
    )
    spectrum_axis.axvline(q_cut, color="#f58518", linewidth=1.8, label="Peak-region left boundary")
    spectrum_axis.axvline(
        landau_gauss_fit["mpv_q_adc"], color="#2ca02c", linewidth=1.5, linestyle="--", label="Fitted MPV"
    )
    spectrum_axis.set_xlabel("Q [ADC]")
    spectrum_axis.set_ylabel("Hits")
    spectrum_axis.grid(True, alpha=0.25)
    spectrum_axis.legend(loc="upper right", fontsize=8)

    prefix = output_dir / "channel_137_safe_time2_peak_right_profile_linear"
    fig.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    summary = {
        "input": str(input_path),
        "time2_mode": "safe",
        "event_selection": "exact 4-hit: two upstream plus downstream 234 and 243",
        "prompt_selection": "finite Q/slope, positive Delta I, positive Delta t12, coarse delta 0",
        "q_selection": {
            "definition": "Q >= lower half-maximum boundary of the prompt Q peak",
            "q_cut_adc": float(q_cut),
            "selected_hits": int(len(fit_q)),
        },
        "profile": {"bins_requested": 10, "points": profile},
        "linear_profile_fit": linear_fit,
        "landau_gauss_fit": {
            key: value
            for key, value in landau_gauss_fit.items()
            if key not in {"grid", "conditional_density", "signal_conditional_density"}
        },
    }
    with (output_dir / "channel_137_safe_time2_peak_right_profile_linear_summary.json").open("w") as output:
        json.dump(summary, output, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
