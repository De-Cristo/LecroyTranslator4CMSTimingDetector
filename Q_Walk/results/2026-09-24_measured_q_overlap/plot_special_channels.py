"""Plot the user-selected 11 channels from measured-overlap AWC results."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CHANNELS = (240, 159, 156, 155, 154, 151, 150, 137, 135, 134, 133)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-ch240", action="store_true")
    parser.add_argument("--fit-line", action="store_true")
    args = parser.parse_args()
    channels = tuple(ch for ch in CHANNELS if not (args.exclude_ch240 and ch == 240))
    data = pd.read_csv(OUT / "channels.csv").set_index("channel").loc[list(channels)].reset_index()
    data["inverse_c1_adc_ns_per_ua"] = 1 / data.c1
    stem = OUT / ("awc_summary_special_10_channels_no_ch240_linear" if args.exclude_ch240
                  else "awc_summary_special_11_channels_linear")
    if args.fit_line:
        stem = stem.with_name(stem.name + "_fit")
    data.to_csv(stem.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for module, color, marker in (("upstream", "#1f77b4", "o"), ("downstream", "#d62728", "s")):
        rows = data.loc[data.module.eq(module)]
        if rows.empty:
            continue
        ax.vlines(rows.inverse_c1_adc_ns_per_ua, rows.awc_p16, rows.awc_p84, color=color, alpha=.7)
        ax.scatter(rows.inverse_c1_adc_ns_per_ua, rows.awc_ua, color=color, marker=marker,
                   s=60, label=module.capitalize(), zorder=3)
    offsets = {133: (6, 6), 134: (7, 12), 135: (-32, 24), 137: (-35, -44),
               150: (12, -12), 151: (-15, -24), 154: (5, -14), 155: (6, 20),
               156: (6, -25), 159: (-27, 47), 240: (-48, 7)}
    for row in data.itertuples():
        ax.annotate(f"Ch{row.channel}", (row.inverse_c1_adc_ns_per_ua, row.awc_ua),
                    xytext=offsets[row.channel], textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="-", color="0.55", linewidth=.6))
    ax.set(xlabel=r"Inverse slope-to-Q coefficient $1/c_1$ [ADC ns/uA]",
           ylabel="Individually optimized AWC [uA]",
           title="Selected channels: AWC from measured-Q overlap")
    ax.margins(x=.08, y=.12)
    if args.fit_line:
        x = data.inverse_c1_adc_ns_per_ua.to_numpy()
        y = data.awc_ua.to_numpy()
        sigma = (data.awc_p84.to_numpy() - data.awc_p16.to_numpy()) / 2
        design = np.column_stack((np.ones(len(x)), x))
        weighted = design / sigma[:, None]
        beta = np.linalg.lstsq(weighted, y / sigma, rcond=None)[0]
        covariance = np.linalg.inv(weighted.T @ weighted)
        chi2 = float(np.sum(((y - design @ beta) / sigma)**2))
        errors = np.sqrt(np.diag(covariance))
        xx = np.linspace(x.min(), x.max(), 300)
        ax.plot(xx, beta[0] + beta[1] * xx, color="#c43b39", linewidth=1.8,
                label="Weighted linear fit")
        ax.text(.98, .04, (
            f"AWC = {beta[0]:.3f} + {beta[1]:.4f} x\n"
            f"x = 1/c1; chi2 / ndf = {chi2:.1f} / {len(x)-2}\n"
            "Weights: 1 / [(p84 - p16)/2]^2"
        ), transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=.95))
        result = dict(channels=list(channels), model="AWC = intercept + slope * (1/c1)",
                      intercept_ua=float(beta[0]), slope=float(beta[1]),
                      formal_parameter_errors=errors.tolist(), covariance=covariance.tolist(),
                      chi2=chi2, ndf=len(x)-2,
                      weights="sigma=(p84-p16)/2, weight=1/sigma^2",
                      limitations="Fixed x; x-y correlations and t0-reference uncertainty omitted. Covariance not rescaled by chi2/ndf.")
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    ax.grid(alpha=.25)
    ax.legend(loc="upper left")
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)
    assert set(data.channel) == set(channels) and len(data) == len(channels)
    assert np.isfinite(data.inverse_c1_adc_ns_per_ua).all()
    print(stem)


if __name__ == "__main__":
    main()
