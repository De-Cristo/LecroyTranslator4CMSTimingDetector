"""Plot consecutive matched-trigger interval ratios minus one; no clock fit."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "assets/note1_matching_ns/matched_timestamps.csv"
OUT = ROOT / "assets/note1_matching_ratio"


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-.5 * ((x - mean) / sigma) ** 2)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    groups = []
    for subset, group in data.groupby("subset", sort=True):
        group = group.sort_values("trigger_ps_from_meta")
        scope = np.diff(group.trigger_ps_from_meta.to_numpy())
        febd = np.diff(group.root_time_ps.to_numpy())
        assert np.all(scope > 0) and np.all(np.isfinite(febd))
        groups.append(pd.DataFrame(dict(subset=subset,
            first_entry=group.entry.to_numpy()[:-1], second_entry=group.entry.to_numpy()[1:],
            scope_interval_ps=scope, febd_interval_ps=febd, ratio_minus_one=febd / scope - 1)))
    points = pd.concat(groups, ignore_index=True)
    points["interval_index"] = np.arange(len(points))
    values = points.ratio_minus_one.to_numpy() * 1e6
    median = float(np.median(values))
    scale = float(1.4826 * np.median(abs(values - median)))
    # Keep every interval in the scatter and CSV; zoom the histogram as in
    # the original ratio figure so the narrow central distribution is visible.
    low, high = np.percentile(values, [1, 99])
    counts, edges = np.histogram(values, bins=400, range=(low, high))
    centers = (edges[1:] + edges[:-1]) / 2
    core = abs(centers - median) <= 1.5 * scale
    assert core.sum() >= 6
    parameters, covariance = curve_fit(gaussian, centers[core], counts[core],
        p0=[float(counts.max()), median, scale],
        sigma=np.sqrt(np.maximum(counts[core], 1)), absolute_sigma=True,
        bounds=([0, median - 1.5 * scale, .0001], [np.inf, median + 1.5 * scale, 10]),
        maxfev=10000)
    assert np.all(np.isfinite(parameters)) and np.all(np.isfinite(covariance))

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "pdf.fonttype": 42, "axes.formatter.useoffset": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    fig.subplots_adjust(left=.08, right=.985, bottom=.16, top=.79, wspace=.30)
    for (subset, group), color in zip(points.groupby("subset"), ("#2878a6", "#d17b28", "#35835c")):
        axes[0].scatter(group.interval_index, group.ratio_minus_one * 1e6,
                        s=5, alpha=.55, color=color, rasterized=True, label=f"Subset {subset}")
    axes[0].axhline(median, color="#c63737", ls="--", lw=1.3, label="Median")
    axes[0].set(xlabel="Matched-interval index", ylabel=r"$(r-1)\times10^6$",
                title="Consecutive trigger intervals")
    axes[0].legend(loc="lower right", fontsize=12, markerscale=2.5)
    axes[1].stairs(counts, edges, fill=True, color="#528bab", alpha=.6,
                   label="Central 98% (display range)")
    x = np.linspace(low, high, 3000)
    axes[1].plot(x, gaussian(x, *parameters), color="#c63737", lw=2,
                 label="Gaussian core fit")
    axes[1].set(xlabel=r"$(r-1)\times10^6$", ylabel="Intervals / bin",
                title="Interval-ratio distribution")
    axes[1].legend(loc="upper left", fontsize=12)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.08, .955, "CMS Preliminary Phase 2", fontsize=17, ha="left")
    fig.text(.08, .88, r"$r=\Delta t_{\rm FEBd}/\Delta t_{\rm scope}$; dimensionless, no clock correction",
             fontsize=15, ha="left")
    fig.savefig(OUT / "event_matching.pdf")
    plt.close(fig)
    points.to_csv(OUT / "interval_ratios.csv", index=False)
    pd.DataFrame(dict(low_scaled=edges[:-1], high_scaled=edges[1:], count=counts,
                      used_for_core_fit=core)).to_csv(OUT / "histogram.csv", index=False)
    report = dict(observable="r-1 = delta_t_FEBd / delta_t_scope - 1", units="dimensionless",
        display_multiplier=1e6, input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        input=str(INPUT.relative_to(ROOT)), matched_events=len(data), intervals=len(points),
        subset_interval_counts=points.groupby("subset").size().to_dict(),
        cross_subset_intervals=False, clock_alignment_applied=False, lowess_applied=False,
        median_ratio_minus_one=median / 1e6, histogram_display_percentiles=[1, 99],
        histogram_intervals=int(counts.sum()), intervals_outside_histogram=int(len(points) - counts.sum()),
        gaussian_mean_ratio_minus_one=float(parameters[1] / 1e6),
        gaussian_sigma_ratio=float(parameters[2] / 1e6),
        gaussian_covariance_scaled=covariance.tolist(),
        gaussian_fit_window_scaled=[median - 1.5 * scale, median + 1.5 * scale],
        gaussian_fit_method="Poisson-error-weighted Gaussian fit within median +/- 1.5 scaled MAD",
        interpretation="Width is interval-ratio spread, not intrinsic timing noise; no attribution of all tails to drift.")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Interval ratios: {len(points)} intervals, {counts.sum()} in histogram display range", flush=True)


if __name__ == "__main__":
    prepare()
