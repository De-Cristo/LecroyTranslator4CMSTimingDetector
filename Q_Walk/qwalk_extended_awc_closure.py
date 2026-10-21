"""Close extended Run-4076 slope profiles against Run-4399 direct-t0 quadratics."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Q_Walk.qwalk_extended_t0_reference import DOWNSTREAM_CHANNELS, UPSTREAM_CHANNELS
from Q_Walk.qwalk_channels import DOWNSTREAM_BAR_CHANNELS, UPSTREAM_BAR_CHANNELS


PROFILE_ROOT = Path("Q_Walk/results/run4076_extended_profiles_v1")
REFERENCE_ROOT = Path("Q_Walk/reference_quadratic_t0_extended_v1")
DEFAULT_OUTPUT = Path("Q_Walk/results/run4076_4399_extended_individual_awc_closure_v1")
AVAILABILITY_TABLE = Path("Q_Walk/results/run4076_4399_channel_availability_scan_v1/channel_availability.csv")
CHANNELS = (*UPSTREAM_CHANNELS, *DOWNSTREAM_CHANNELS)


def common_q_domain(first, second):
    """Return the closed Q intersection or reject disjoint intervals."""
    low = max(map(float, (first[0], second[0])))
    high = min(map(float, (first[1], second[1])))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Run-4076 and Run-4399 Q domains do not overlap")
    return low, high


def _module(channel):
    return "upstream" if channel in UPSTREAM_CHANNELS else "downstream"


def _bar(channel):
    bars = UPSTREAM_BAR_CHANNELS if _module(channel) == "upstream" else DOWNSTREAM_BAR_CHANNELS
    for bar, pair in bars.items():
        if channel in pair:
            return bar
    raise ValueError(f"unmapped channel {channel}")


def _paths(channel, profile_root=PROFILE_ROOT, reference_root=REFERENCE_ROOT):
    module = _module(channel)
    prefix = Path(profile_root) / module / f"channel_{channel}_safe_time2_two_{module}_peak_right_profile_linear"
    reference = Path(reference_root) / f"run4399_{module}_bar{_bar(channel)}_ch{channel}_t0_quadratic.json"
    return prefix.with_name(f"{prefix.name}_summary.json"), prefix.with_name(f"{prefix.name}_bootstrap_coefficients.npz"), reference


def _walk(q, coefficients):
    a, b, c = coefficients
    q = np.asarray(q, dtype=float)
    return a + b / q + c / q**2


def _optimal_awc(inverse_slope_shape, reference_shape):
    denominator = float(np.dot(inverse_slope_shape, inverse_slope_shape))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("inverse-slope shape is flat or non-finite")
    return float(np.dot(inverse_slope_shape, reference_shape) / denominator)


def _channel_terms(channel, grid_points, profile_root=PROFILE_ROOT, reference_root=REFERENCE_ROOT):
    summary_path, bootstrap_path, reference_path = _paths(channel, profile_root, reference_root)
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    with reference_path.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    with np.load(bootstrap_path) as archive:
        bootstrap = np.asarray(archive["coefficients"], dtype=float)
        q_edges = np.asarray(archive["q_bin_edges_adc"], dtype=float)
    coefficients = np.array([
        summary["linear_profile_fit"]["intercept_ua_per_ns"],
        summary["linear_profile_fit"]["coefficient_ua_per_ns_per_adc"],
    ])
    reference_coefficients = tuple(float(reference["coefficients_ns"][key]) for key in ("a", "b", "c"))
    q_profile_domain = (float(q_edges[0]), float(q_edges[-1]))
    reference_support = reference.get("profile_domain", reference["fit_domain"])
    q_reference_domain = (
        float(reference_support["q_min_adc"]),
        float(reference_support["q_max_adc"]),
    )
    availability = np.genfromtxt(
        AVAILABILITY_TABLE, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    scan_row = next(row for row in np.atleast_1d(availability) if int(row["channel"]) == channel)
    q_selected_domain = (float(scan_row["run4399_q_low_adc"]), float(scan_row["run4399_q_high_adc"]))
    q_low, q_high = common_q_domain(
        common_q_domain(q_profile_domain, q_selected_domain), q_reference_domain
    )
    quadratic_extrapolated = False
    q = np.linspace(q_low, q_high, grid_points)
    pivot_q = 0.5 * (q_low + q_high)
    slope = coefficients[0] + coefficients[1] * q
    pivot_slope = coefficients[0] + coefficients[1] * pivot_q
    if np.any(slope <= 0.0) or pivot_slope <= 0.0:
        raise ValueError("non-positive central S(Q) over the common Q domain")
    inverse_shape = 1.0 / slope - 1.0 / pivot_slope
    reference_shape = _walk(q, reference_coefficients) - _walk(pivot_q, reference_coefficients)
    awc = _optimal_awc(inverse_shape, reference_shape)
    bootstrap_slope = bootstrap[:, [0]] + bootstrap[:, [1]] * q[None, :]
    bootstrap_pivot = bootstrap[:, 0] + bootstrap[:, 1] * pivot_q
    valid = np.all(bootstrap_slope > 0.0, axis=1) & (bootstrap_pivot > 0.0)
    bootstrap_shapes = 1.0 / bootstrap_slope[valid] - 1.0 / bootstrap_pivot[valid, None]
    bootstrap_awc = np.asarray([_optimal_awc(shape, reference_shape) for shape in bootstrap_shapes])
    residual_ps = 1000.0 * (awc * inverse_shape - reference_shape)
    return {
        "channel": channel, "module": _module(channel), "bar": _bar(channel),
        "q": q, "q_profile_domain": q_profile_domain, "q_reference_domain": q_reference_domain,
        "q_selected_domain": q_selected_domain, "quadratic_extrapolated": quadratic_extrapolated,
        "q_common_domain": (q_low, q_high), "pivot_q": pivot_q,
        "coefficients": coefficients, "reference_coefficients": reference_coefficients,
        "inverse_shape": inverse_shape, "reference_shape": reference_shape,
        "awc": awc, "bootstrap_awc": bootstrap_awc,
        "shape_rms_ps": float(np.sqrt(np.mean(residual_ps**2))),
        "summary_path": str(summary_path), "reference_path": str(reference_path),
    }


def _plot_channel(item, output_dir):
    q = item["q"]
    awc = item["awc"]
    p16, p50, p84 = np.percentile(item["bootstrap_awc"], [16.0, 50.0, 84.0])
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.plot(q, item["reference_shape"], color="black", linewidth=2.2, label="Run-4399 direct-t0 quadratic")
    axis.plot(q, awc * item["inverse_shape"], color="#d62728", linewidth=2.2, label="Run-4076 slope mapping")
    axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    axis.axvline(item["pivot_q"], color="0.45", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Q [ADC]")
    axis.set_ylabel("Relative walk term W(Q) [ns]")
    axis.set_title(f"Channel {item['channel']}: Run-4076 slope vs Run-4399 direct-t0 shape")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right")
    axis.text(0.02, 0.98, (
        f"Common Q: {q[0]:.1f} to {q[-1]:.1f} ADC\n"
        f"AWC = {awc:.3f} uA (68%: {p16:.3f} to {p84:.3f})\n"
        f"Shape RMS = {item['shape_rms_ps']:.1f} ps"
        + ("\nRun-4399 quadratic extrapolated" if item["quadratic_extrapolated"] else "")
    ), transform=axis.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.55", "alpha": 0.92})
    prefix = Path(output_dir) / f"channel_{item['channel']}_individual_awc_shape_closure"
    figure.savefig(prefix.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return prefix


def _plot_awc_summary(results, output_dir, stem, title, excluded_channels=(), log_x=True):
    styles = {
        "upstream": {"color": "#1f77b4", "marker": "o", "label": "Upstream"},
        "downstream": {"color": "#d62728", "marker": "s", "label": "Downstream"},
    }
    excluded_channels = {int(channel) for channel in excluded_channels}
    valid = [
        item for item in results
        if item["coefficients"][1] > 0.0 and item["channel"] not in excluded_channels
    ]
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for module, style in styles.items():
        members = sorted(
            (item for item in valid if item["module"] == module),
            key=lambda item: 1.0 / item["coefficients"][1],
        )
        for item in members:
            low, _, high = np.percentile(item["bootstrap_awc"], [16.0, 50.0, 84.0])
            axis.vlines(1.0 / item["coefficients"][1], low, high, color=style["color"], alpha=0.7)
        axis.scatter([1.0 / item["coefficients"][1] for item in members], [item["awc"] for item in members], color=style["color"], marker=style["marker"], s=58, label=style["label"], zorder=3)
        offsets = ((5, 7), (5, -13), (5, 20), (5, -26), (5, 33))
        for index, item in enumerate(members):
            axis.annotate(
                f"Ch{item['channel']}",
                (1.0 / item["coefficients"][1], item["awc"]),
                xytext=offsets[index % len(offsets)],
                textcoords="offset points",
                fontsize=8,
            )
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(r"Inverse slope-to-Q coefficient $1/c_1$ [ADC ns/uA]")
    axis.set_ylabel("Individually optimized AWC [uA]")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left")
    prefix_awc = Path(output_dir) / stem
    figure.savefig(prefix_awc.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(prefix_awc.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return prefix_awc


def _plot_summaries(results, output_dir):
    styles = {
        "upstream": {"color": "#1f77b4", "marker": "o", "label": "Upstream"},
        "downstream": {"color": "#d62728", "marker": "s", "label": "Downstream"},
    }
    prefix_awc = _plot_awc_summary(
        results,
        output_dir,
        "awc_vs_inverse_slope_to_q_coefficient",
        "Extended channel AWC vs inverse slope-to-Q coefficient",
    )
    prefix_awc_filtered = _plot_awc_summary(
        results,
        output_dir,
        "awc_vs_inverse_slope_to_q_coefficient_no_ch130_ch240",
        "AWC vs inverse slope-to-Q coefficient (Ch130 and Ch240 omitted)",
        excluded_channels=(130, 240),
    )
    prefix_awc_linear = _plot_awc_summary(
        results,
        output_dir,
        "awc_vs_inverse_slope_to_q_coefficient_no_ch130_ch139_ch240_linear",
        "AWC vs inverse slope-to-Q coefficient (Ch130, Ch139, and Ch240 omitted)",
        excluded_channels=(130, 139, 240),
        log_x=False,
    )
    prefix_awc_linear_reduced = _plot_awc_summary(
        results,
        output_dir,
        "awc_vs_inverse_slope_to_q_coefficient_no_ch130_ch139_ch240_ch242_ch244_linear",
        "AWC vs inverse slope-to-Q coefficient (Ch130, Ch139, Ch240, Ch242, and Ch244 omitted)",
        excluded_channels=(130, 139, 240, 242, 244),
        log_x=False,
    )

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positions = np.arange(len(results))
    colors = [styles[item["module"]]["color"] for item in results]
    axis.scatter(positions, [item["shape_rms_ps"] for item in results], color=colors, s=58)
    axis.set_xticks(positions, [str(item["channel"]) for item in results], rotation=45)
    axis.set_xlabel("Channel")
    axis.set_ylabel("Common-domain shape RMS [ps]")
    axis.set_title("Extended direct-t0 versus slope-shape closure")
    axis.grid(True, alpha=0.25)
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="", color=style["color"], label=style["label"])
            for style in styles.values()
        ],
        loc="upper right",
    )
    prefix_rms = Path(output_dir) / "shape_closure_rms_by_channel"
    figure.savefig(prefix_rms.with_suffix(".png"), dpi=180, facecolor="white")
    figure.savefig(prefix_rms.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return [prefix_awc, prefix_awc_filtered, prefix_awc_linear, prefix_awc_linear_reduced, prefix_rms]


def _serializable(item, prefix):
    p16, p50, p84 = np.percentile(item["bootstrap_awc"], [16.0, 50.0, 84.0])
    c1 = float(item["coefficients"][1])
    return {
        "channel": item["channel"], "module": item["module"], "bar": item["bar"],
        "optimal_awc_ua": item["awc"],
        "bootstrap_percentiles_ua": [float(p16), float(p50), float(p84)],
        "shape_rms_ps": item["shape_rms_ps"],
        "slope_to_q_coefficient_ua_per_ns_per_adc": c1,
        "inverse_slope_to_q_coefficient_adc_ns_per_ua": float(1.0 / c1) if c1 != 0.0 else None,
        "q_profile_domain_adc": list(item["q_profile_domain"]),
        "q_reference_domain_adc": list(item["q_reference_domain"]),
        "q_selected_domain_adc": list(item["q_selected_domain"]),
        "q_common_domain_adc": list(item["q_common_domain"]),
        "quadratic_extrapolated": item["quadratic_extrapolated"],
        "run4076_summary": item["summary_path"], "run4399_reference": item["reference_path"],
        "figures": [str(prefix.with_suffix(ext)) for ext in (".png", ".pdf")],
    }


def run(output_dir=DEFAULT_OUTPUT, grid_points=3000, profile_root=PROFILE_ROOT, reference_root=REFERENCE_ROOT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, failures = [], {}
    for channel in CHANNELS:
        try:
            item = _channel_terms(channel, grid_points, profile_root, reference_root)
            prefix = _plot_channel(item, output_dir)
            result = _serializable(item, prefix)
            results.append(item)
            with (output_dir / f"channel_{channel}_individual_awc_closure.json").open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
        except (FileNotFoundError, KeyError, ValueError, np.linalg.LinAlgError) as error:
            failures[str(channel)] = str(error)
    summaries = _plot_summaries(results, output_dir) if results else []
    report = {
        "method": "common-Q-domain least-squares closure: W_t0(Q) = AWC * [1/S(Q) - 1/S(Q_pivot)]",
        "channel_count_requested": len(CHANNELS), "channel_count_completed": len(results),
        "channels": [_serializable(item, output_dir / f"channel_{item['channel']}_individual_awc_shape_closure") for item in results],
        "failures": failures,
        "summary_figures": [str(prefix.with_suffix(ext)) for prefix in summaries for ext in (".png", ".pdf")],
    }
    with (output_dir / "extended_individual_awc_closure_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-points", type=int, default=3000)
    parser.add_argument("--profile-root", default=PROFILE_ROOT)
    parser.add_argument("--reference-root", default=REFERENCE_ROOT)
    arguments = parser.parse_args()
    print(json.dumps(
        run(arguments.out_dir, arguments.grid_points, arguments.profile_root, arguments.reference_root),
        indent=2,
    ))


if __name__ == "__main__":
    main()
