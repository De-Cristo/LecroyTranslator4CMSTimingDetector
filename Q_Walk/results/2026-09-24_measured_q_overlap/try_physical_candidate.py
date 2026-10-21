"""Exploratory AWC prediction with whole-bar held-out evaluation."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "physical_candidate_quick_test"


def design(frame, name):
    if name == "constant":
        return np.ones((len(frame), 1))
    if name == "inverse_c1_line":
        return np.column_stack((np.ones(len(frame)), 1 / frame.c1))
    return frame.physical_feature.to_numpy()[:, None]


def fit(frame, name):
    x = design(frame, name)
    return np.linalg.lstsq(x / frame.sigma.to_numpy()[:, None],
                          frame.awc_ua / frame.sigma, rcond=None)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qstar", choices=("median", "peak", "mean"), default="median")
    args = parser.parse_args()
    output = BASE / (OUT.name if args.qstar == "median" else f"physical_candidate_{args.qstar}_test")
    output.mkdir(exist_ok=True)
    data = pd.read_csv(BASE / "awc_summary_special_10_channels_no_ch240_linear.csv")
    records = []
    with np.load(BASE / "energy_spectra/quality_selected_q.npz") as archive:
        for row in data.to_dict("records"):
            ch = row["channel"]
            source = BASE / "inputs" / f"channel_{ch}"
            summary = json.loads(next(source.glob("*summary.json")).read_text())
            q = archive[f"run4076_ch{ch}"]
            # Use only run4076 to set histogram edges, so no reference data
            # enters this internal feature even through the binning.
            counts, edges = np.histogram(q, bins=80)
            peak_bin = int(np.argmax(counts))
            peak_q = float((edges[peak_bin] + edges[peak_bin+1])/2)
            q = q[q >= summary["q_selection"]["q_cut_adc"]]
            qstar = {"peak": peak_q, "median": float(np.median(q)), "mean": float(np.mean(q))}[args.qstar]
            outside = not row["q4076_min"] <= qstar <= row["q4076_max"]
            if outside and args.qstar != "peak":
                raise ValueError(f"Ch{ch}: {args.qstar} outside measured fit support")
            c0 = summary["linear_profile_fit"]["intercept_ua_per_ns"]
            sstar = c0 + row["c1"] * qstar
            if sstar <= 0 or row["c1"] <= 0:
                raise ValueError(f"Ch{ch}: invalid physical feature")
            domain = pd.read_csv(BASE / f"channel_{ch}_comparison.csv").q_adc.to_numpy()
            row.update(qstar_adc=qstar, c0=c0, sstar=sstar,
                       qstar_definition=args.qstar,
                       qstar_outside_profile_means=outside,
                       peak_bin_low_adc=float(edges[peak_bin]),
                       peak_bin_high_adc=float(edges[peak_bin+1]),
                       peak_bin_count=int(counts[peak_bin]),
                       rho=row["c1"] * qstar / sstar,
                       physical_feature=sstar**2 / (row["c1"] * qstar),
                       sigma=(row["awc_p84"]-row["awc_p16"])/2,
                       inverse_slope_std=float(np.std(1/(c0+row["c1"]*domain))))
            records.append(row)
    frame = pd.DataFrame(records)
    models = ("constant", "inverse_c1_line", "physical")
    report = {}
    for name in models:
        beta = fit(frame, name)
        fitted = design(frame, name) @ beta
        prediction = np.zeros(len(frame))
        folds = []
        for bar in sorted(frame.bar.unique()):
            held = frame.bar.eq(bar).to_numpy()
            fold_beta = fit(frame.loc[~held], name)
            prediction[held] = design(frame.loc[held], name) @ fold_beta
            folds.append(dict(bar=int(bar), held_channels=frame.loc[held,"channel"].tolist(),
                              coefficients=fold_beta.tolist()))
        error = prediction-frame.awc_ua.to_numpy()
        curve_error = 1000 * np.abs(error) * frame.inverse_slope_std.to_numpy()
        frame[f"{name}_predicted_awc_lobo"] = prediction
        frame[f"{name}_curve_rms_ps_lobo"] = curve_error
        report[name] = dict(coefficients=beta.tolist(),
            training_chi2=float(np.sum(((fitted-frame.awc_ua)/frame.sigma)**2)),
            training_ndf=len(frame)-len(beta),
            lobo_awc_rmse_ua=float(np.sqrt(np.mean(error**2))),
            lobo_awc_mae_ua=float(np.mean(np.abs(error))),
            lobo_relative_rmse=float(np.sqrt(np.mean((error/frame.awc_ua)**2))),
            lobo_curve_rms_ps=float(np.sqrt(np.mean(curve_error**2))), folds=folds)
    frame.to_csv(output / "channel_predictions.csv", index=False)
    (output / "model_comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    ax.vlines(frame.physical_feature, frame.awc_p16, frame.awc_p84, color="#1f77b4", alpha=.65)
    ax.scatter(frame.physical_feature, frame.awc_ua, color="#1f77b4")
    xx = np.linspace(0, frame.physical_feature.max()*1.08, 200)
    k = report["physical"]["coefficients"][0]
    ax.plot(xx, k*xx, color="#c43b39", label=f"AWC = K X; K = {k:.4f} ns")
    for row in frame.itertuples():
        ax.annotate(str(row.channel), (row.physical_feature,row.awc_ua),
                    xytext=(4,5), textcoords="offset points", fontsize=8)
    ax.set(xlabel=r"$X=S_*^2/(c_1 Q_*)$ [uA/ns]", ylabel="Reference AWC [uA]",
           title=f"Physical candidate: Q* from {args.qstar}")
    ax.legend(fontsize=9)
    ax = axes[1]
    upper = max(frame.awc_ua.max(), *(frame[f"{n}_predicted_awc_lobo"].max() for n in models))*1.1
    ax.plot([0,upper],[0,upper],color="0.4",linestyle="--")
    for name, color, marker in zip(models, ("0.5", "#1f77b4", "#c43b39"), ("x","s","o")):
        rmse = report[name]["lobo_awc_rmse_ua"]
        ax.scatter(frame.awc_ua,frame[f"{name}_predicted_awc_lobo"],
                   color=color,marker=marker,label=f"{name}: RMSE {rmse:.2f} uA")
    ax.set(xlabel="Reference AWC [uA]",ylabel="Predicted AWC [uA]",
           title="Leave-one-bar-out prediction",xlim=(0,upper),ylim=(0,upper))
    ax.legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ("pdf","png"):
        fig.savefig(output / f"physical_candidate_comparison.{ext}",dpi=180)
    plt.close(fig)
    print(json.dumps({name:{k:v for k,v in item.items() if k != 'folds'}
                      for name,item in report.items()},indent=2))
    print("Q* outside outer profile means:", frame.loc[frame.qstar_outside_profile_means, "channel"].tolist())


if __name__ == "__main__":
    main()
