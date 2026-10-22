"""Safe-time2 Q-walk diagnostic using exact two-hit same-bar events."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_channels import (
    DOWNSTREAM_BAR_CHANNELS,
    UPSTREAM_BAR_CHANNELS,
    bar_pair_for_channel,
)
from Q_Walk.qwalk_core import calculate_slope, decode_thresholds, read_root_hits
from Q_Walk.qwalk_run4076_peak_right_profile import _build_profile, _fit_linear_profile
from Q_Walk.qwalk_run4076_safe_time2_scatter import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    _fit_landau_gauss,
    _histogram_peak_region,
    _limits,
)

BOOTSTRAP_BAND_VISUAL_SCALE = 10.0
DEFAULT_MIN_DELTA_T12_PS = 100.0
UPSTREAM_EXTENDED_CHANNELS = (130, 133, 134, 135, 136, 137, 139, 149, 150, 151, 154, 155, 156, 159)
DOWNSTREAM_EXTENDED_CHANNELS = (235, 236, 238, 239, 240, 242, 244, 245)
LANDAU_GAUSS_ARRAY_FIELDS = {
    "grid",
    "conditional_density",
    "signal_conditional_density",
}


def _dynamic_ylim(
    primary_values=None,
    extra_mins=(),
    extra_maxs=(),
    low_percentile=0.5,
    high_percentile=99.5,
    bottom_padding=0.08,
    top_padding=0.20,
    min_span=1.0,
):
    """Compute dynamic Y-axis limits ensuring all data points, fits, and error bands fit."""
    y_mins, y_maxs = [], []
    if primary_values is not None:
        finite = np.asarray(primary_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) > 0:
            p_low, p_high = np.percentile(finite, [low_percentile, high_percentile])
            y_mins.append(p_low)
            y_maxs.append(p_high)
    for v in extra_mins:
        if v is not None:
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                y_mins.append(float(np.min(arr)))
    for v in extra_maxs:
        if v is not None:
            arr = np.asarray(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                y_maxs.append(float(np.max(arr)))
    if not y_mins or not y_maxs:
        return (0.0, min_span)
    y_min, y_max = min(y_mins), max(y_maxs)
    span = max(y_max - y_min, min_span)
    return (float(y_min - bottom_padding * span), float(y_max + top_padding * span))


def rise_time_mask(delta_t12_ps, minimum_delta_t12_ps):
    """Keep finite reconstructed rises at or above the configured threshold."""
    if not np.isfinite(minimum_delta_t12_ps) or minimum_delta_t12_ps < 0.0:
        raise ValueError("minimum_delta_t12_ps must be finite and non-negative")
    values = np.asarray(delta_t12_ps, dtype=float)
    return np.isfinite(values) & (values >= float(minimum_delta_t12_ps))


def _bootstrap_linear_profile(q, slope, bins=10, replicas=1000, seed=4076):
    """Bootstrap the full profile-and-fit procedure with fixed Q-bin edges."""
    q = np.asarray(q, dtype=float)
    slope = np.asarray(slope, dtype=float)
    edges = np.linspace(q.min(), q.max(), bins + 1)
    assignments = np.digitize(q, edges) - 1
    assignments[assignments == bins] = bins - 1
    generator = np.random.default_rng(seed)
    coefficients = np.empty((replicas, 2), dtype=float)
    attempts = 0
    successful = 0
    max_attempts = 10 * replicas

    while successful < replicas and attempts < max_attempts:
        attempts += 1
        indices = generator.integers(0, len(q), size=len(q))
        profile = _build_profile(q[indices], slope[indices], edges=edges)
        try:
            fit = _fit_linear_profile(profile)
        except (ValueError, np.linalg.LinAlgError):
            continue
        coefficients[successful] = (
            fit["intercept_ua_per_ns"],
            fit["coefficient_ua_per_ns_per_adc"],
        )
        successful += 1
    if successful != replicas:
        raise RuntimeError(
            f"only {successful} valid bootstrap replicas after {attempts} attempts"
        )

    return {
        "replicas": int(replicas),
        "seed": int(seed),
        "attempts": int(attempts),
        "discarded_replicas": int(attempts - successful),
        "q_bin_edges_adc": edges.tolist(),
        "coefficients": coefficients,
    }


def select_pair_with_exactly_two_module_hits(hits, pair, module_channels):
    """Return one bar pair when it is the event's only pair from that module."""
    pair = tuple(map(int, pair))
    module_channels = {
        channel for channels in module_channels.values() for channel in channels
    }
    is_module = hits["channelID"].isin(module_channels)
    is_pair = hits["channelID"].isin(pair)
    module_count = is_module.groupby(hits["entry"]).transform("sum")
    pair_count = is_pair.groupby(hits["entry"]).transform("sum")
    pair_unique = hits.loc[is_pair].groupby("entry")["channelID"].nunique()
    pair_unique_count = hits["entry"].map(pair_unique).fillna(0)
    return hits.loc[
        (module_count == 2)
        & (pair_count == 2)
        & (pair_unique_count == 2)
        & is_pair
    ].copy()


def select_pair_with_exactly_two_upstream_hits(hits, pair):
    """Return an upstream bar pair when it is the event's only upstream pair."""
    return select_pair_with_exactly_two_module_hits(hits, pair, UPSTREAM_BAR_CHANNELS)


def run(
    input_path,
    output_dir,
    channel=137,
    module="upstream",
    raw_hits=None,
    minimum_delta_t12_ps=DEFAULT_MIN_DELTA_T12_PS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_maps = {
        "upstream": UPSTREAM_BAR_CHANNELS,
        "downstream": DOWNSTREAM_BAR_CHANNELS,
    }
    if module not in module_maps:
        raise ValueError("module must be 'upstream' or 'downstream'")
    pair = bar_pair_for_channel(channel)
    if not any(channel in endpoints for endpoints in module_maps[module].values()):
        raise ValueError(f"channel {channel} is not a {module} channel")
    if raw_hits is None:
        raw_hits = read_root_hits(input_path, time2_mode="safe")
    pair_hits = select_pair_with_exactly_two_module_hits(
        raw_hits, pair, module_maps[module]
    )
    channel_hits = pair_hits.loc[pair_hits["channelID"].eq(channel)].copy()

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
    prompt_before_rise_time_cut = (
        np.isfinite(q)
        & np.isfinite(slope)
        & (delta_i > 0.0)
        & (delta_t12_ps > 0.0)
        & (coarse_delta == 0)
    )
    prompt = prompt_before_rise_time_cut & rise_time_mask(
        delta_t12_ps, minimum_delta_t12_ps
    )
    prompt_q = q[prompt]
    prompt_slope = slope[prompt]
    prompt_x_limits = _limits(prompt_q, 0.5, 99.5, 20.0)
    peak = _histogram_peak_region(prompt_q, prompt_x_limits)
    landau_gauss_fit = _fit_landau_gauss(prompt_q, peak)
    q_cut = peak["half_max_q_low_adc"]
    peak_right = prompt & (q >= q_cut)
    selected_q = q[peak_right]
    selected_slope = slope[peak_right]
    profile = _build_profile(selected_q, selected_slope, bins=10)
    linear_fit = _fit_linear_profile(profile)
    bootstrap = _bootstrap_linear_profile(selected_q, selected_slope)

    fig, (scatter_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    scatter_axis.scatter(
        selected_q,
        selected_slope,
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
    q_line = np.linspace(selected_q.min(), selected_q.max(), 300)
    slope_line = (
        linear_fit["intercept_ua_per_ns"]
        + linear_fit["coefficient_ua_per_ns_per_adc"] * q_line
    )
    bootstrap_curves = (
        bootstrap["coefficients"][:, [0]]
        + bootstrap["coefficients"][:, [1]] * q_line[None, :]
    )
    bootstrap_low, bootstrap_high = np.percentile(bootstrap_curves, [16.0, 84.0], axis=0)
    display_bootstrap_low = slope_line + BOOTSTRAP_BAND_VISUAL_SCALE * (bootstrap_low - slope_line)
    display_bootstrap_high = slope_line + BOOTSTRAP_BAND_VISUAL_SCALE * (bootstrap_high - slope_line)
    scatter_axis.fill_between(
        q_line,
        display_bootstrap_low,
        display_bootstrap_high,
        color="#d62728",
        alpha=0.18,
        label=f"68% bootstrap confidence band x{BOOTSTRAP_BAND_VISUAL_SCALE:g} (display)",
    )
    scatter_axis.plot(q_line, slope_line, color="#d62728", linewidth=2.0, label="Weighted linear profile fit")
    scatter_axis.set_xlim(*_limits(selected_q, 0.5, 99.5, 20.0))
    scatter_y_lim = _dynamic_ylim(
        primary_values=selected_slope,
        extra_mins=[slope_mean - slope_sd, slope_line, display_bootstrap_low],
        extra_maxs=[slope_mean + slope_sd, slope_line, display_bootstrap_high],
        bottom_padding=0.08,
        top_padding=0.20,
    )
    scatter_axis.set_ylim(*scatter_y_lim)
    scatter_axis.set_ylabel("Slope [uA/ns]")
    scatter_axis.set_title(
        f"Run 4076, channel {channel}: exact two-{module} bar-pair slope profile with safe time2"
    )
    scatter_axis.text(
        0.02,
        0.98,
        "Required pair: {{{left}, {right}}}\nDelta t12 >= {minimum_delta_t12_ps:g} ps\n"
        "Q >= {q_cut:.1f} ADC\nSelected: {count:,}\n"
        "S(Q) = {intercept:.3f} {sign} {magnitude:.5f} Q\n"
        "1,000-replica bootstrap band x{band_scale:g} (display)".format(
            left=pair[0],
            right=pair[1],
            minimum_delta_t12_ps=minimum_delta_t12_ps,
            q_cut=q_cut,
            count=len(selected_q),
            intercept=linear_fit["intercept_ua_per_ns"],
            sign="+" if linear_fit["coefficient_ua_per_ns_per_adc"] >= 0.0 else "-",
            magnitude=abs(linear_fit["coefficient_ua_per_ns_per_adc"]),
            band_scale=BOOTSTRAP_BAND_VISUAL_SCALE,
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
    landau_gauss_fit_domain = (
        (landau_gauss_fit["grid"] >= landau_gauss_fit["fit_q_low_adc"])
        & (landau_gauss_fit["grid"] <= landau_gauss_fit["fit_q_high_adc"])
    )
    spectrum_axis.plot(
        landau_gauss_fit["grid"][landau_gauss_fit_domain],
        landau_gauss_fit["conditional_density"][landau_gauss_fit_domain]
        * landau_gauss_fit["fit_event_count"]
        * bin_width,
        color="#2ca02c",
        linewidth=2.0,
        label="Landau-Gaussian + uniform background",
    )
    spectrum_axis.hlines(
        landau_gauss_fit["background_density_per_adc"]
        * landau_gauss_fit["fit_event_count"]
        * bin_width,
        landau_gauss_fit["fit_q_low_adc"],
        landau_gauss_fit["fit_q_high_adc"],
        color="#9467bd",
        linewidth=1.7,
        linestyle=":",
        label="Fitted uniform background",
    )
    spectrum_axis.axvline(q_cut, color="#f58518", linewidth=1.8, label="Peak-region left boundary")
    spectrum_axis.axvline(
        landau_gauss_fit["mpv_q_adc"],
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label="Fitted MPV",
    )
    spectrum_axis.set_xlabel("Q [ADC]")
    spectrum_axis.set_ylabel("Hits")
    spectrum_axis.grid(True, alpha=0.25)
    spectrum_axis.legend(loc="upper right", fontsize=8)

    prefix = output_dir / f"channel_{channel}_safe_time2_two_{module}_peak_right_profile_linear"
    fig.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    bootstrap_path = prefix.with_name(f"{prefix.name}_bootstrap_coefficients.npz")
    np.savez_compressed(
        bootstrap_path,
        coefficients=bootstrap["coefficients"],
        q_bin_edges_adc=np.asarray(bootstrap["q_bin_edges_adc"], dtype=float),
    )

    profile_fig, (profile_axis, profile_spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        height_ratios=(3, 1),
        constrained_layout=True,
    )
    profile_axis.errorbar(
        q_mean,
        slope_mean,
        yerr=slope_sd,
        fmt="o",
        color="black",
        markersize=6,
        capsize=3,
        label="10-bin profile mean +/- SD",
    )
    profile_axis.fill_between(
        q_line,
        display_bootstrap_low,
        display_bootstrap_high,
        color="#d62728",
        alpha=0.18,
        label=f"68% bootstrap confidence band x{BOOTSTRAP_BAND_VISUAL_SCALE:g} (display)",
    )
    profile_axis.plot(
        q_line,
        slope_line,
        color="#d62728",
        linewidth=2.0,
        label="Weighted linear profile fit",
    )
    profile_axis.set_xlim(*_limits(selected_q, 0.5, 99.5, 20.0))
    profile_y_lim = _dynamic_ylim(
        extra_mins=[slope_mean - slope_sd, slope_line, display_bootstrap_low],
        extra_maxs=[slope_mean + slope_sd, slope_line, display_bootstrap_high],
        bottom_padding=0.08,
        top_padding=0.20,
    )
    profile_axis.set_ylim(*profile_y_lim)
    profile_axis.set_ylabel("Slope [uA/ns]")
    profile_axis.set_title(f"Run 4076, channel {channel}: slope profile with safe time2")
    profile_axis.text(
        0.02,
        0.98,
        "Required pair: {{{left}, {right}}}\nDelta t12 >= {minimum_delta_t12_ps:g} ps\n"
        "Q >= {q_cut:.1f} ADC\nSelected: {count:,}\n"
        "S(Q) = {intercept:.3f} {sign} {magnitude:.5f} Q".format(
            left=pair[0],
            right=pair[1],
            minimum_delta_t12_ps=minimum_delta_t12_ps,
            q_cut=q_cut,
            count=len(selected_q),
            intercept=linear_fit["intercept_ua_per_ns"],
            sign="+" if linear_fit["coefficient_ua_per_ns_per_adc"] >= 0.0 else "-",
            magnitude=abs(linear_fit["coefficient_ua_per_ns_per_adc"]),
        ),
        transform=profile_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92},
    )
    profile_axis.grid(True, alpha=0.25)
    profile_axis.legend(loc="lower right", fontsize=9)

    profile_spectrum_axis.hist(
        prompt_q,
        bins=peak["edges"],
        histtype="stepfilled",
        color="#4c78a8",
        alpha=0.82,
    )
    profile_spectrum_axis.plot(
        landau_gauss_fit["grid"][landau_gauss_fit_domain],
        landau_gauss_fit["conditional_density"][landau_gauss_fit_domain]
        * landau_gauss_fit["fit_event_count"]
        * bin_width,
        color="#2ca02c",
        linewidth=2.0,
        label="Landau-Gaussian + uniform background",
    )
    profile_spectrum_axis.hlines(
        landau_gauss_fit["background_density_per_adc"]
        * landau_gauss_fit["fit_event_count"]
        * bin_width,
        landau_gauss_fit["fit_q_low_adc"],
        landau_gauss_fit["fit_q_high_adc"],
        color="#9467bd",
        linewidth=1.7,
        linestyle=":",
        label="Fitted uniform background",
    )
    profile_spectrum_axis.axvline(
        q_cut,
        color="#f58518",
        linewidth=1.8,
        label="Peak-region left boundary",
    )
    profile_spectrum_axis.axvline(
        landau_gauss_fit["mpv_q_adc"],
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label="Fitted MPV",
    )
    profile_spectrum_axis.set_xlabel("Q [ADC]")
    profile_spectrum_axis.set_ylabel("Hits")
    profile_spectrum_axis.grid(True, alpha=0.25)
    profile_spectrum_axis.legend(loc="upper right", fontsize=8)
    profile_only_prefix = prefix.with_name(f"{prefix.name}_only")
    profile_fig.savefig(profile_only_prefix.with_suffix(".png"), dpi=180, facecolor="white")
    profile_fig.savefig(profile_only_prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(profile_fig)

    summary = {
        "input": str(input_path),
        "time2_mode": "safe",
        "target_channel": int(channel),
        "required_exact_pair": list(pair),
        "event_selection": (
            f"exactly two {module} hits, equal to the required pair; "
            f"non-{module} hits are ignored"
        ),
        "selection_module": module,
        "pair_channel_hits": int(len(channel_hits)),
        "prompt_selection": {
            "definition": "finite Q/slope, positive Delta I, positive delta t12, coarse delta 0, and delta t12 >= minimum",
            "minimum_delta_t12_ps": float(minimum_delta_t12_ps),
            "before_rise_time_cut_hits": int(prompt_before_rise_time_cut.sum()),
            "rejected_short_rise_hits": int(
                (prompt_before_rise_time_cut & ~rise_time_mask(delta_t12_ps, minimum_delta_t12_ps)).sum()
            ),
        },
        "prompt_channel_hits": int(prompt.sum()),
        "q_selection": {
            "definition": "Q >= lower half-maximum boundary of the prompt Q peak",
            "q_cut_adc": float(q_cut),
            "selected_hits": int(len(selected_q)),
        },
        "profile": {"bins_requested": 10, "points": profile},
        "linear_profile_fit": linear_fit,
        "linear_profile_bootstrap": {
            "replicas": bootstrap["replicas"],
            "seed": bootstrap["seed"],
            "attempts": bootstrap["attempts"],
            "discarded_replicas": bootstrap["discarded_replicas"],
            "confidence_level": "central 68 percent",
            "coefficients_file": str(bootstrap_path.name),
            "fixed_q_bin_edges_adc": bootstrap["q_bin_edges_adc"],
            "intercept_ua_per_ns_percentiles": np.percentile(
                bootstrap["coefficients"][:, 0], [16.0, 50.0, 84.0]
            ).tolist(),
            "coefficient_ua_per_ns_per_adc_percentiles": np.percentile(
                bootstrap["coefficients"][:, 1], [16.0, 50.0, 84.0]
            ).tolist(),
        },
        "landau_gauss_fit": {
            key: value
            for key, value in landau_gauss_fit.items()
            if key not in LANDAU_GAUSS_ARRAY_FIELDS
        },
    }
    with prefix.with_name(f"{prefix.name}_summary.json").open("w") as output:
        json.dump(summary, output, indent=2)
    return summary


def run_all(
    input_path=DEFAULT_INPUT,
    output_dir="Q_Walk/results/run4076_extended_profiles_v1_remake",
    minimum_delta_t12_ps=DEFAULT_MIN_DELTA_T12_PS,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_hits = read_root_hits(input_path, time2_mode="safe")
    results = {}
    for channel in UPSTREAM_EXTENDED_CHANNELS:
        mod_dir = output_dir / "upstream"
        results[f"upstream_{channel}"] = run(
            input_path,
            mod_dir,
            channel=channel,
            module="upstream",
            raw_hits=raw_hits,
            minimum_delta_t12_ps=minimum_delta_t12_ps,
        )
    for channel in DOWNSTREAM_EXTENDED_CHANNELS:
        mod_dir = output_dir / "downstream"
        results[f"downstream_{channel}"] = run(
            input_path,
            mod_dir,
            channel=channel,
            module="downstream",
            raw_hits=raw_hits,
            minimum_delta_t12_ps=minimum_delta_t12_ps,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--channel", type=int, default=137)
    parser.add_argument("--module", choices=("upstream", "downstream"), default="upstream")
    parser.add_argument(
        "--min-delta-t12-ps",
        type=float,
        default=DEFAULT_MIN_DELTA_T12_PS,
        help="Reject reconstructed rises below this value before the Q selection",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Process all 22 extended upstream and downstream channels",
    )
    args = parser.parse_args()
    if args.all_channels:
        print(json.dumps(run_all(args.input, args.out_dir, args.min_delta_t12_ps), indent=2))
    else:
        print(json.dumps(
            run(
                args.input,
                args.out_dir,
                args.channel,
                args.module,
                minimum_delta_t12_ps=args.min_delta_t12_ps,
            ),
            indent=2,
        ))


if __name__ == "__main__":
    main()
