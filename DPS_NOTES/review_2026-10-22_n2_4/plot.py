"""Render uncertainty and weighting diagnostics from saved N2-4 tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def save(fig, name):
    (OUT / "figures").mkdir(exist_ok=True)
    fig.text(.10, .985, r"$\bf{CMS}$ Preliminary Phase 2", va="top", fontsize=13)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"figures/{name}.{suffix}", dpi=170)
    plt.close(fig)


def main():
    summary = pd.read_csv(OUT / "uncertainties.csv").set_index("channel")
    weights = pd.read_csv(OUT / "weighting.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.subplots_adjust(left=.15, right=.98, bottom=.16, top=.88, hspace=.65, wspace=.55)
    for ch, ax in zip((133, 154, 137, 150), axes.flat):
        row = summary.loc[ch]
        for y, name, color in ((2, "slope_only", "#0072B2"),
                               (1, "reference_only", "#009E73"), (0, "combined", "#D55E00")):
            low, high = row[f"{name}_p16"], row[f"{name}_p84"]
            ax.hlines(y, low, high, color=color, lw=3)
            ax.plot([low, high], [y, y], "|", color=color, ms=9)
            ax.plot(row.awc_ua, y, "o", color=color, ms=4)
        ax.set(yticks=[0, 1, 2], yticklabels=["Both runs", "MCP only", "Slope only"],
               xlabel="AWC [uA]", ylim=(-.6, 2.6), title=f"Channel {ch}: nominal {row.awc_ua:.2f} uA")
        ax.grid(axis="x", alpha=.25)
    fig.text(.10, .035, "Bars: conditional 16th-84th percentiles. Dots: unchanged nominal AWC, not the replica median.\nBoth runs: saved slope bootstrap combined with the correlated Gaussian MCP coefficient approximation.", fontsize=10)
    save(fig, "uncertainty_budget_four_channels")

    names = ("uniform_inverse_q", "population4076", "population4399")
    values = weights.pivot(index="channel", columns="weighting", values="delta_awc_percent")[list(names)]
    fig, ax = plt.subplots(figsize=(8.5, 9))
    fig.subplots_adjust(left=.18, right=.86, bottom=.18, top=.89)
    heat = ax.imshow(values.to_numpy(), cmap="RdBu_r", vmin=-70, vmax=70, aspect="auto")
    for i in range(len(values)):
        for j in range(len(names)):
            value = values.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 50 else "black")
    ax.set(xticks=range(3), xticklabels=["Uniform 1/Q", "4076 population", "4399 population"],
           yticks=range(len(values)), yticklabels=[f"{'U' if ch < 192 else 'D'} {ch}" for ch in values.index],
           title="AWC weighting shifts on the unchanged measured support", ylabel="Module / channel")
    ax.tick_params(axis="x", rotation=20)
    fig.colorbar(heat, ax=ax, fraction=.045, pad=.03).set_label("AWC shift / |nominal AWC| [%]")
    fig.text(.10, .035, "U: upstream; D: downstream. Nominal objective: uniform Q. Channel 238 has no overlap.\nPopulation weights are count / Q-bin width, normalized on retained support.\nObjective sensitivity only: these shifts are not statistical errors or calibrated systematics.", fontsize=9)
    save(fig, "weighting_sensitivity")

    # A compact table for the note; all 21 channels remain in the CSV diagnostics.
    chosen = summary.loc[[133, 154, 137, 150]]
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Channel & AWC & Slope only & Both runs & Max. weight shift\\", r"\midrule"]
    for ch, row in chosen.iterrows():
        lines.append(f"{ch} & {row.awc_ua:.2f} & [{row.slope_only_p16:.2f}, {row.slope_only_p84:.2f}] & "
                     f"[{row.combined_p16:.2f}, {row.combined_p84:.2f}] & {row.max_weighting_shift_ua:.2f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "uncertainty_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
