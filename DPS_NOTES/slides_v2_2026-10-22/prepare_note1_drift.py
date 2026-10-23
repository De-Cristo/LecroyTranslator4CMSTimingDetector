"""Redraw clock drift for the same three matched subsets as slide 4."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
sys.path.insert(0, str(WORKSPACE / "TimeCalibration"))
from ch192_vs_trigger_lowess import _lowess_smooth

INPUT = ROOT / "assets/note1_matching_ns/matched_timestamps.csv"
OUT = ROOT / "assets/note1_drift"
FRAC = .15
ITERATIONS = 3


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    groups, summaries = [], []
    for subset, group in data.groupby("subset", sort=True):
        group = group.sort_values("trigger_ps_from_meta").copy()
        scope_origin = float(group.trigger_ps_from_meta.iloc[0])
        febd_origin = float(group.root_time_ps.iloc[0])
        u = (group.trigger_ps_from_meta.to_numpy() - scope_origin) / 1000
        t = (group.root_time_ps.to_numpy() - febd_origin) / 1000
        slope, intercept = np.polyfit(u, t, 1)
        residual = t - (slope * u + intercept)
        drift = _lowess_smooth(u, residual, frac=FRAC, it=ITERATIONS)
        assert np.all(np.isfinite(drift))
        group["elapsed_scope_s"] = u / 1e9
        group["linear_residual_ns"] = residual
        group["lowess_drift_ns"] = drift
        group["corrected_residual_ns"] = residual - drift
        groups.append(group)
        summaries.append(dict(subset=int(subset), events=len(group), slope=float(slope),
            centered_intercept_ns=float(intercept), scope_origin_ps=scope_origin,
            febd_origin_ps=febd_origin,
            pre_lowess_scaled_mad_ns=float(1.4826 * np.median(abs(residual - np.median(residual)))),
            before_sd_ns=float(np.std(residual, ddof=1)),
            after_sd_ns=float(np.std(residual - drift, ddof=1))))
    points = pd.concat(groups, ignore_index=True)

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "pdf.fonttype": 42, "axes.formatter.useoffset": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), sharey=True)
    fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.77, wspace=.14)
    lo = np.floor(points.linear_residual_ns.min() / .05) * .05
    hi = np.ceil(points.linear_residual_ns.max() / .05) * .05
    for ax, group, color in zip(axes, groups, ("#2878a6", "#d17b28", "#35835c")):
        ax.scatter(group.elapsed_scope_s, group.linear_residual_ns, s=5,
                   alpha=.45, color=color, rasterized=True)
        ax.axhline(0, color="0.5", ls="--", lw=.8)
        ax.plot(group.elapsed_scope_s, group.lowess_drift_ns, color="#c63737", lw=2,
                label="LOWESS")
        ax.set(title=f"Subset {int(group.subset.iloc[0])}",
               xlabel="Elapsed scope time [s]", ylim=(lo, hi))
        ax.grid(alpha=.18)
        ax.legend(loc="lower right", fontsize=12)
    axes[0].set_ylabel("Timing residual [ns]")
    fig.text(.075, .955, "CMS Preliminary Phase 2", fontsize=17, ha="left")
    fig.text(.075, .88, r"$r_s(u)=t_{\rm FEBd}-(a_s u+b_s)$; red curves: LOWESS drift $d_s(u)$",
             fontsize=15, ha="left")
    fig.savefig(OUT / "clock_residuals.pdf")
    plt.close(fig)
    points.to_csv(OUT / "drift_points.csv", index=False)
    report = dict(input=str(INPUT.relative_to(ROOT)),
        input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        events=len(points), subsets=summaries, linear_fit="Centered least-squares line per subset",
        residual_units="ns", horizontal_units="elapsed seconds within each subset",
        lowess_frac=FRAC, lowess_robust_iterations=ITERATIONS,
        implementation="TimeCalibration/ch192_vs_trigger_lowess.py::_lowess_smooth",
        implementation_sha256=hashlib.sha256((WORKSPACE / "TimeCalibration/ch192_vs_trigger_lowess.py").read_bytes()).hexdigest(),
        no_display_clipping=True, no_event_rejection=True,
        common_y_limits_ns=[float(lo), float(hi)],
        correction="g_s(u)=a_s*u+b_s+d_s(u); corrected residual=r_s(u)-d_s(u)",
        detector_calibration_results_unchanged=True,
        reference=dict(author="William S. Cleveland", year=1979,
            title="Robust Locally Weighted Regression and Smoothing Scatterplots",
            journal="Journal of the American Statistical Association", volume=74, pages="829-836",
            doi="10.1080/01621459.1979.10481038"))
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Clock drift: {len(points)} matched events, three subsets, LOWESS frac={FRAC}, it={ITERATIONS}", flush=True)


if __name__ == "__main__":
    prepare()
