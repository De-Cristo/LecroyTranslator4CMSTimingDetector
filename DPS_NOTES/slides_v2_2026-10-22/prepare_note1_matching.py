"""Plot matched-trigger residuals in ns from a frozen timestamp snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets/note1_matching_ns"


def snapshot(source):
    columns = ["entry", "root_time_ps", "meta_file", "trigger_ps_from_meta"]
    data = pd.read_csv(source, usecols=columns)
    valid = np.isfinite(data.root_time_ps) & np.isfinite(data.trigger_ps_from_meta)
    data = data.loc[valid].copy()
    data["subset"] = pd.factorize(data.meta_file, sort=False)[0] + 1
    assert not data.entry.duplicated().any()
    assert not data.duplicated(["meta_file", "trigger_ps_from_meta"]).any()
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "matched_timestamps.csv", index=False)
    provenance = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      matched_events=len(data), subset_counts=data.groupby("subset").size().to_dict(),
                      selection="Finite matched FEBd/scope timestamps; no detector topology or charge cuts.",
                      original_ratio_plot_pair_counts=[2404, 2395, 2404])
    assert data.groupby("subset").size().tolist() == [2405, 2396, 2405]
    (OUT / "input_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-.5 * ((x - mean) / sigma) ** 2)


def prepare():
    data = pd.read_csv(OUT / "matched_timestamps.csv")
    results, fits = [], []
    for subset, group in data.groupby("subset", sort=True):
        group = group.sort_values("trigger_ps_from_meta").copy()
        # Center before fitting to avoid subtracting large absolute timestamps.
        x = (group.trigger_ps_from_meta.to_numpy() - group.trigger_ps_from_meta.iloc[0]) / 1000
        y = (group.root_time_ps.to_numpy() - group.root_time_ps.iloc[0]) / 1000
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        group["residual_ns"] = residual
        group["elapsed_scope_ns"] = x
        group["rolling_mean_ns"] = uniform_filter1d(residual, size=201, mode="reflect")
        results.append(group)
        fits.append(dict(subset=int(subset), events=len(group), slope=float(slope),
                         centered_intercept_ns=float(intercept),
                         scope_origin_ps=float(group.trigger_ps_from_meta.iloc[0]),
                         febd_origin_ps=float(group.root_time_ps.iloc[0])))
    points = pd.concat(results, ignore_index=True)
    points["event_index"] = np.arange(len(points))
    r = points.residual_ns.to_numpy()
    median = float(np.median(r))
    scale = float(1.4826 * np.median(abs(r - median)))
    edges = np.linspace(np.floor(r.min() / .01) * .01, np.ceil(r.max() / .01) * .01, 81)
    counts, edges = np.histogram(r, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    core = abs(centers - median) <= 1.5 * scale
    parameters, covariance = curve_fit(
        gaussian, centers[core], counts[core],
        p0=[float(counts.max()), median, scale],
        sigma=np.sqrt(np.maximum(counts[core], 1)), absolute_sigma=True,
        bounds=([0, median - 1.5 * scale, .0001], [np.inf, median + 1.5 * scale, 1.]),
        maxfev=10000)
    assert np.all(np.isfinite(parameters)) and np.all(np.isfinite(covariance))
    assert counts.sum() == len(points)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "pdf.fonttype": 42, "axes.formatter.useoffset": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.8))
    fig.subplots_adjust(left=.08, right=.985, bottom=.16, top=.79, wspace=.30)
    for (subset, group), color in zip(points.groupby("subset"), ("#2878a6", "#d17b28", "#35835c")):
        axes[0].scatter(group.event_index, group.residual_ns, s=4, alpha=.55,
                        color=color, rasterized=True, label=f"Subset {subset}")
    axes[0].axhline(0, color="0.4", ls="--", lw=1)
    axes[0].set(xlabel="Matched-event index", ylabel="Timing residual [ns]",
                title="After linear clock alignment")
    axes[0].legend(loc="lower left", fontsize=12, markerscale=2.5)
    axes[1].stairs(counts, edges, fill=True, color="#528bab", alpha=.55, label="All matched events")
    x = np.linspace(edges[0], edges[-1], 600)
    axes[1].plot(x, gaussian(x, *parameters), color="#c63737", lw=2,
                 label=f"Gaussian core fit\n$\\sigma$ = {parameters[2]:.3f} ns")
    axes[1].set(xlabel="Timing residual [ns]", ylabel="Events / bin",
                title="Residual distribution")
    axes[1].legend(loc="upper left", fontsize=12)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.08, .955, "CMS Preliminary Phase 2", fontsize=17, ha="left")
    fig.text(.08, .88, r"$r=t_{\rm FEBd}-(a_s t_{\rm scope}+b_s)$; before LOWESS correction",
             fontsize=15, ha="left")
    fig.savefig(OUT / "event_matching.pdf")
    plt.close(fig)

    # Diagnostic only: check whether slow structure contributes to the tails.
    interior = pd.concat([g.iloc[100:-100] for _, g in points.groupby("subset")])
    cutoff = 3 * parameters[2]
    before_tail = abs(interior.residual_ns - parameters[1]) > cutoff
    highpass = interior.residual_ns - interior.rolling_mean_ns
    after_tail = abs(highpass - parameters[1]) > cutoff
    summary = dict(events=len(points), linear_fits=fits,
                   residual_definition="(t_FEBd - (a_s*t_scope + b_s)) in ns; per-subset linear fit; no LOWESS",
                   gaussian_amplitude=float(parameters[0]), gaussian_mean_ns=float(parameters[1]),
                   gaussian_sigma_ns=float(parameters[2]), covariance=covariance.tolist(),
                   gaussian_fit_window_ns=[median - 1.5 * scale, median + 1.5 * scale],
                   gaussian_fit_method="Poisson-error-weighted fit to histogram bins within median +/- 1.5 scaled MAD",
                   histogram_bins=80, histogram_events=int(counts.sum()), histogram_clipping=False,
                   noise_interpretation="Core is an effective synchronization width, not an isolated intrinsic component.",
                   slow_structure_diagnostic=dict(window_events=201, edge_events_excluded_per_end=100,
                       threshold_ns=float(cutoff), events=len(interior),
                       tail_fraction_before=float(before_tail.mean()),
                       tail_fraction_after_rolling_mean_subtraction=float(after_tail.mean()),
                       note="Diagnostic only, not applied to displayed points; not proof that all tails are drift."),
                   input_sha256=hashlib.sha256((OUT / "matched_timestamps.csv").read_bytes()).hexdigest())
    points.to_csv(OUT / "residuals.csv", index=False)
    pd.DataFrame(dict(low_ns=edges[:-1], high_ns=edges[1:], count=counts,
                      used_for_core_fit=core)).to_csv(OUT / "histogram.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Matching residuals: N={len(points)}, Gaussian core sigma={parameters[2]:.4f} ns", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Create the frozen timestamp snapshot from matched CSV")
    args = parser.parse_args()
    if args.input:
        snapshot(args.input)
    prepare()
