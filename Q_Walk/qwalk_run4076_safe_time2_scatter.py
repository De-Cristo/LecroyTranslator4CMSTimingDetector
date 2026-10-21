"""Quick Run-4076 safe-time2 slope-versus-Q diagnostic for channel 137."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import levy_stable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_core import (
    calculate_slope,
    decode_thresholds,
    read_root_hits,
    select_upstream_hits_with_fixed_downstream_bar,
)


DEFAULT_INPUT = (
    "/eos/cms/store/group/dpg_mtd/comm_mtd/TB/MTDTB_H8_Sep2025/"
    "TOFHIR/reco/4076/23_e.root"
)
DEFAULT_OUTPUT = "Q_Walk/results/run4076_23_e_safe_time2"


def _limits(values, low_percentile, high_percentile, minimum_span):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, minimum_span)
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    span = max(high - low, minimum_span)
    return (float(low - 0.05 * span), float(high + 0.05 * span))


def _histogram_peak_region(values, value_range, bins=80):
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    smoothed = np.convolve(counts, np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0, mode="same")
    peak_index = int(np.argmax(smoothed))
    half_maximum = 0.5 * smoothed[peak_index]
    left = peak_index
    while left > 0 and smoothed[left - 1] >= half_maximum:
        left -= 1
    right = peak_index
    while right < len(smoothed) - 1 and smoothed[right + 1] >= half_maximum:
        right += 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "counts": counts,
        "edges": edges,
        "peak_q_adc": float(centers[peak_index]),
        "half_max_q_low_adc": float(edges[left]),
        "half_max_q_high_adc": float(edges[right + 1]),
    }


STANDARD_LANDAU_AXIS = np.linspace(-8.0, 80.0, 3001)
STANDARD_LANDAU_PDF = levy_stable.pdf(STANDARD_LANDAU_AXIS, 1.0, 1.0)
STANDARD_LANDAU_MODE = float(STANDARD_LANDAU_AXIS[np.argmax(STANDARD_LANDAU_PDF)])


def _trapezoid(values, coordinates):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, coordinates)
    return np.trapz(values, coordinates)


def _landau_gauss_density(grid, mpv, landau_width, gaussian_sigma):
    spacing = float(grid[1] - grid[0])
    standardized = (grid - mpv) / landau_width + STANDARD_LANDAU_MODE
    landau_density = np.interp(
        standardized,
        STANDARD_LANDAU_AXIS,
        STANDARD_LANDAU_PDF,
        left=0.0,
        right=0.0,
    ) / landau_width
    return gaussian_filter1d(
        landau_density,
        gaussian_sigma / spacing,
        mode="constant",
        truncate=6.0,
    )


def _fit_landau_gauss(values, peak):
    peak_q = peak["peak_q_adc"]
    left_half_width = peak_q - peak["half_max_q_low_adc"]
    right_half_width = peak["half_max_q_high_adc"] - peak_q
    fit_low = peak_q - max(2.0 * left_half_width, 20.0)
    fit_high = peak_q + max(5.0 * right_half_width, 60.0)
    fit_values = np.asarray(values, dtype=float)
    fit_values = fit_values[(fit_values >= fit_low) & (fit_values <= fit_high)]
    if len(fit_values) < 100:
        raise ValueError("fewer than 100 events in the automatically chosen fit domain")

    width_limit = 0.5 * (fit_high - fit_low)
    padding = max(8.0 * width_limit, 100.0)
    grid = np.linspace(fit_low - padding, fit_high + padding, 1400)
    in_fit_domain = (grid >= fit_low) & (grid <= fit_high)

    fit_width = float(fit_high - fit_low)

    def mixture_density(parameters):
        mpv, landau_width, gaussian_sigma, signal_fraction = parameters
        signal_density = _landau_gauss_density(
            grid, mpv, landau_width, gaussian_sigma
        )
        signal_normalization = _trapezoid(
            signal_density[in_fit_domain], grid[in_fit_domain]
        )
        if not np.isfinite(signal_normalization) or signal_normalization <= 0.0:
            return None
        signal_conditional_density = signal_density / signal_normalization
        background_density = (1.0 - signal_fraction) / fit_width
        conditional_density = (
            signal_fraction * signal_conditional_density
            + np.where(in_fit_domain, background_density, 0.0)
        )
        return signal_conditional_density, conditional_density, background_density

    def objective(parameters):
        mixture = mixture_density(parameters)
        if mixture is None:
            return np.inf
        _, conditional_density, _ = mixture
        values_density = np.interp(
            fit_values, grid, conditional_density, left=0.0, right=0.0
        )
        if np.any(values_density <= 0.0) or not np.all(np.isfinite(values_density)):
            return np.inf
        return float(-np.log(values_density).sum())

    initial = np.array(
        [
            peak_q,
            max(left_half_width / 2.0, 3.0),
            max(left_half_width / 3.0, 1.0),
            0.90,
        ]
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=(
            (fit_low, fit_high),
            (0.5, width_limit),
            (0.2, width_limit),
            (0.0, 1.0),
        ),
        options={"maxls": 100},
    )
    if not result.success:
        raise RuntimeError(result.message)

    signal_conditional_density, conditional_density, background_density = mixture_density(
        result.x
    )
    covariance = None
    if hasattr(result.hess_inv, "todense"):
        covariance = np.asarray(result.hess_inv.todense(), dtype=float).tolist()
    return {
        "mpv_q_adc": float(result.x[0]),
        "landau_width_adc": float(result.x[1]),
        "gaussian_sigma_adc": float(result.x[2]),
        "signal_fraction": float(result.x[3]),
        "background_fraction": float(1.0 - result.x[3]),
        "background_density_per_adc": float(background_density),
        "fit_q_low_adc": float(fit_low),
        "fit_q_high_adc": float(fit_high),
        "fit_event_count": int(len(fit_values)),
        "negative_log_likelihood": float(result.fun),
        "covariance_approx": covariance,
        "grid": grid,
        "conditional_density": conditional_density,
        "signal_conditional_density": signal_conditional_density,
    }


def run(input_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_hits = read_root_hits(input_path, time2_mode="safe")
    selected = select_upstream_hits_with_fixed_downstream_bar(raw_hits)
    channel_hits = selected.loc[selected["channelID"].eq(137)].copy()

    _, _, delta_i = decode_thresholds(channel_hits["step2"].to_numpy())
    delta_t12_ps = (
        channel_hits["time2"].to_numpy(dtype=float)
        - channel_hits["time"].to_numpy(dtype=float)
    )
    slope = calculate_slope(
        channel_hits["time"].to_numpy(dtype=float),
        channel_hits["time2"].to_numpy(dtype=float),
        delta_i,
    )
    q = channel_hits["energy"].to_numpy(dtype=float)

    finite = np.isfinite(q) & np.isfinite(slope)
    positive_dt = np.isfinite(delta_t12_ps) & (delta_t12_ps > 0.0)
    nonpositive_dt = np.isfinite(delta_t12_ps) & (delta_t12_ps <= 0.0)
    low_slope = finite & (np.abs(slope) <= 3.0)
    coarse_delta = (
        channel_hits["t2coarse"].to_numpy(dtype=np.int64)
        - channel_hits["t1coarse"].to_numpy(dtype=np.int64)
    ) % 1024
    prompt = finite & (delta_i > 0.0) & positive_dt & (coarse_delta == 0)
    low_slope_groups = (
        pd.DataFrame(
            {
                "delta_i_ua": delta_i[low_slope],
                "coarse_delta": coarse_delta[low_slope],
                "tac_id": channel_hits.loc[low_slope, "tacID"].to_numpy(),
                "delta_t12_ps": delta_t12_ps[low_slope],
                "slope_ua_per_ns": slope[low_slope],
            }
        )
        .groupby(["delta_i_ua", "coarse_delta", "tac_id"], as_index=False)
        .agg(
            count=("slope_ua_per_ns", "size"),
            median_delta_t12_ps=("delta_t12_ps", "median"),
            median_slope_ua_per_ns=("slope_ua_per_ns", "median"),
        )
        .sort_values("count", ascending=False)
        .head(12)
    )
    summary = {
        "input": str(input_path),
        "time2_mode": "safe",
        "selection": "exact 4-hit: two upstream plus downstream 234 and 243",
        "channel": 137,
        "selected_channel_hits": int(len(channel_hits)),
        "finite_q_slope": int(finite.sum()),
        "positive_delta_t12": int(positive_dt.sum()),
        "nonpositive_delta_t12": int(nonpositive_dt.sum()),
        "nonfinite_delta_t12": int((~np.isfinite(delta_t12_ps)).sum()),
        "q_percentiles_adc": np.nanpercentile(q, [1.0, 50.0, 99.0]).tolist(),
        "slope_percentiles_ua_per_ns": np.nanpercentile(
            slope[finite], [1.0, 50.0, 99.0]
        ).tolist(),
        "abs_slope_le_3_count": int(low_slope.sum()),
        "abs_slope_le_3_top_signatures": low_slope_groups.to_dict("records"),
        "prompt_selection": {
            "definition": "finite Q/slope, positive Delta I, positive Delta t12, coarse delta 0",
            "retained": int(prompt.sum()),
            "rejected_nonpositive_delta_i": int((finite & (delta_i <= 0.0)).sum()),
            "rejected_nonpositive_delta_t12": int((finite & (delta_i > 0.0) & ~positive_dt).sum()),
            "rejected_nonzero_coarse_delta": int((finite & (delta_i > 0.0) & positive_dt & (coarse_delta != 0)).sum()),
        },
    }

    x_limits = _limits(q[finite], 0.5, 99.5, 20.0)
    y_limits = _limits(slope[finite], 0.5, 99.5, 1.0)
    plot_hits = finite & (q >= x_limits[0]) & (q <= x_limits[1])
    plot_hits &= (slope >= y_limits[0]) & (slope <= y_limits[1])

    fig, (scatter_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    scatter_axis.scatter(
        q[plot_hits],
        slope[plot_hits],
        s=3,
        alpha=0.12,
        color="#1f77b4",
        edgecolors="none",
        rasterized=True,
    )
    scatter_axis.set_xlim(*x_limits)
    scatter_axis.set_ylim(*y_limits)
    scatter_axis.set_ylabel("Slope [uA/ns]")
    scatter_axis.set_title("Run 4076, channel 137: safe time2, exact 4-hit selection")
    scatter_axis.text(
        0.02,
        0.98,
        "Selected: {selected:,}\nPositive dt12: {positive:,}\n"
        "Non-positive dt12: {nonpositive:,}".format(
            selected=summary["selected_channel_hits"],
            positive=summary["positive_delta_t12"],
            nonpositive=summary["nonpositive_delta_t12"],
        ),
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    scatter_axis.grid(True, alpha=0.25)

    spectrum_axis.hist(
        q[np.isfinite(q)],
        bins=80,
        range=x_limits,
        histtype="stepfilled",
        color="#4c78a8",
        alpha=0.85,
    )
    spectrum_axis.set_xlabel("Q [ADC]")
    spectrum_axis.set_ylabel("Hits")
    spectrum_axis.grid(True, alpha=0.25)

    prefix = output_dir / "channel_137_safe_time2_slope_vs_q"
    fig.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    prompt_q = q[prompt]
    prompt_slope = slope[prompt]
    prompt_x_limits = _limits(prompt_q, 0.5, 99.5, 20.0)
    prompt_y_limits = _limits(prompt_slope, 0.5, 99.5, 1.0)
    peak = _histogram_peak_region(prompt_q, prompt_x_limits)
    summary["prompt_selection"]["q_peak_visualization"] = {
        "method": "five-bin smoothed histogram with contiguous half-maximum interval",
        "peak_q_adc": peak["peak_q_adc"],
        "half_max_q_low_adc": peak["half_max_q_low_adc"],
        "half_max_q_high_adc": peak["half_max_q_high_adc"],
        "used_for_selection": False,
    }
    try:
        landau_gauss_fit = _fit_landau_gauss(prompt_q, peak)
        summary["prompt_selection"]["landau_gauss_fit"] = {
            key: value
            for key, value in landau_gauss_fit.items()
            if key not in {"grid", "conditional_density", "signal_conditional_density"}
        }
        summary["prompt_selection"]["landau_gauss_fit"]["status"] = "success"
    except Exception as error:
        landau_gauss_fit = None
        summary["prompt_selection"]["landau_gauss_fit"] = {
            "status": "failed",
            "message": str(error),
        }

    fig, (scatter_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    scatter_axis.scatter(
        prompt_q,
        prompt_slope,
        s=9,
        alpha=0.38,
        color="#1f77b4",
        edgecolors="none",
        rasterized=True,
    )
    scatter_axis.set_xlim(*prompt_x_limits)
    scatter_axis.set_ylim(*prompt_y_limits)
    scatter_axis.set_ylabel("Slope [uA/ns]")
    scatter_axis.set_title("Run 4076, channel 137: prompt slope population with safe time2")
    scatter_axis.text(
        0.02,
        0.98,
        "Retained: {retained:,}\nRejected dt12 <= 0: {nonpositive:,}\n"
        "Rejected coarse delta != 0: {slow:,}".format(
            retained=summary["prompt_selection"]["retained"],
            nonpositive=summary["prompt_selection"]["rejected_nonpositive_delta_t12"],
            slow=summary["prompt_selection"]["rejected_nonzero_coarse_delta"],
        ),
        transform=scatter_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    scatter_axis.grid(True, alpha=0.25)

    spectrum_axis.hist(
        prompt_q,
        bins=peak["edges"],
        histtype="stepfilled",
        color="#4c78a8",
        alpha=0.82,
    )
    spectrum_axis.axvspan(
        peak["half_max_q_low_adc"],
        peak["half_max_q_high_adc"],
        color="#f58518",
        alpha=0.25,
        label="Half-maximum peak region",
    )
    spectrum_axis.axvline(
        peak["peak_q_adc"], color="#e45756", linewidth=1.5, linestyle="--", label="Histogram mode (seed)"
    )
    if landau_gauss_fit is not None:
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
        spectrum_axis.axvline(
            landau_gauss_fit["mpv_q_adc"],
            color="#2ca02c",
            linewidth=1.8,
            label="Fitted MPV",
        )
        spectrum_axis.text(
            0.02,
            0.95,
            "Landau-Gaussian fit\n"
            f"MPV = {landau_gauss_fit['mpv_q_adc']:.1f} ADC\n"
            f"xi = {landau_gauss_fit['landau_width_adc']:.1f} ADC\n"
            f"sigma = {landau_gauss_fit['gaussian_sigma_adc']:.1f} ADC",
            transform=spectrum_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
        )
    spectrum_axis.set_xlabel("Q [ADC]")
    spectrum_axis.set_ylabel("Hits")
    spectrum_axis.legend(loc="upper right", fontsize=8)
    spectrum_axis.grid(True, alpha=0.25)

    prompt_prefix = output_dir / "channel_137_safe_time2_prompt_slope_vs_q"
    fig.savefig(prompt_prefix.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(prompt_prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    with (output_dir / "summary.json").open("w") as output:
        json.dump(summary, output, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.input, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
