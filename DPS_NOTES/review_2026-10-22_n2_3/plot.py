"""Plot N2-3 saved diagnostics without processing ROOT input."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study import evaluate

OUT = Path(__file__).resolve().parent
LABELS = {"baseline": "Nominal", "quadratic": "Quadratic", "bins8": "8 bins", "bins12": "12 bins",
          "qcut_minus5": "Q cut -5%", "qcut_plus5": "Q cut +5%", "rise100": "Rise >=100 ps", "rise100_repeak": "100 ps + new peak"}
COLORS = {"baseline": "#0072B2", "quadratic": "#D55E00"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.formatter.useoffset": False})


def save(fig, name):
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    fig.text(.10, .985, r"$\bf{CMS}$ Preliminary Phase 2", ha="left", va="top", fontsize=13)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def residual_panel(ax, channel, profiles, models):
    for variant, marker in (("baseline", "o"), ("quadratic", "s")):
        p = profiles.loc[(profiles.channel == channel) & (profiles.variant == variant) & profiles.used_in_fit]
        fit = models[(channel, variant)]
        ax.errorbar(p.q_mean_adc, p.residual, yerr=p.slope_sem_ua_per_ns, fmt=marker,
                    ms=4, capsize=2, color=COLORS[variant], label=f"{'Linear' if variant == 'baseline' else 'Quadratic'}: {fit['chi2']:.1f}/{fit['ndf']}")
    ax.axhline(0, color="0.5", lw=.8)
    ax.set(xlabel="Q [ADC]", ylabel="Profile minus model [uA/ns]", title=f"Channel {channel}")
    ax.legend(title="Diagnostic chi-square / ndf", fontsize=9, title_fontsize=9)
    ax.grid(alpha=.2)


def main():
    profiles = pd.read_csv(OUT / "profiles.csv")
    channels = pd.read_csv(OUT / "channels.csv")
    variants = pd.read_csv(OUT / "variants.csv")
    model_rows = json.loads((OUT / "models.json").read_text())
    models = {(r["channel"], r["variant"]): r for r in model_rows if r["profile_status"] == "ok"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.12, hspace=.40, wspace=.28)
    for ch, ax in zip((133, 154, 137, 150), axes.flat):
        residual_panel(ax, ch, profiles, models)
    fig.text(.10, .035, "Same selected hits and 10 bins; bars are SEM. Quadratic predictions use bin-averaged moments.", fontsize=10)
    save(fig, "profile_residuals_four_channels")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.subplots_adjust(left=.13, right=.98, top=.84, bottom=.16)
    residual_panel(ax, 137, profiles, models)
    save(fig, "profile_residuals_ch137")

    available = channels.loc[channels.alignment_status == "ok"].sort_values("channel")
    names = [n for n in LABELS if n != "baseline"]
    values = variants.pivot(index="channel", columns="variant", values="delta_awc_percent").reindex(index=available.channel, columns=names)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=.21, right=.88, top=.88, bottom=.17)
    heat = ax.imshow(values.to_numpy(), cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
    for i in range(len(values)):
        for j in range(len(names)):
            x = values.iloc[i, j]
            label = "failed" if not np.isfinite(x) else (f"{x:+.1f}" if abs(x) < 1000 else f"{x:+.1e}")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(x) and abs(x) >= 70 else "black")
    ax.set_xticks(range(len(names)), [LABELS[n] for n in names], rotation=35, ha="right")
    ax.set_yticks(range(len(values)), [f"{'U' if r.module == 'upstream' else 'D'} {int(r.channel)} ({r.common_grid_fraction:.0%})" for r in available.itertuples()])
    ax.set_title("AWC shifts on common measured charge support", pad=16)
    ax.set_ylabel("Module / channel (retained fraction of saved grid)")
    colorbar = fig.colorbar(heat, ax=ax, fraction=.045, pad=.03, extend="both")
    colorbar.set_label("AWC shift / |baseline AWC| [%]")
    fig.text(.10, .025, "U: upstream; D: downstream. Colors saturate at +/-100%; numbers show actual shifts.\nChannel 235: no support common to all variants. Channel 238: no saved MCP overlap.\nVariations are not confidence intervals.", fontsize=10)
    save(fig, "awc_sensitivity")

    for row in channels.itertuples():
        ch = int(row.channel)
        fig, axes = plt.subplots(3, 1, figsize=(10, 11))
        fig.subplots_adjust(left=.11, right=.98, top=.91, bottom=.13, hspace=.50)
        p = profiles.loc[(profiles.channel == ch) & (profiles.variant == "baseline")]
        axes[0].errorbar(p.q_mean_adc, p.slope_mean_ua_per_ns, yerr=p.slope_sem_ua_per_ns,
                         fmt="o", color="black", ms=4, capsize=2, label="Profile mean +/- SEM")
        q = np.linspace(p.q_mean_adc.min(), p.q_mean_adc.max(), 400)
        for name in ("baseline", "quadratic"):
            model = models[(ch, name)]
            axes[0].plot(q, evaluate(model, q), color=COLORS[name], label="Linear" if name == "baseline" else "Quadratic")
        axes[0].set(xlabel="Q [ADC]", ylabel="Slope [uA/ns]", title=f"Channel {ch}: fixed-sample model comparison")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=.2)
        residual_panel(axes[1], ch, profiles, models)
        cases = variants.loc[variants.channel == ch].set_index("variant").reindex(LABELS)
        if row.alignment_status == "ok":
            ok = cases.alignment_status.eq("ok")
            positions = np.arange(len(LABELS))
            axes[2].scatter(positions[ok], cases.loc[ok, "awc_ua"], color="#0072B2", s=30, zorder=3)
            axes[2].axhspan(row.baseline_p16, row.baseline_p84, color="0.7", alpha=.35, label="Baseline conditional 68% interval")
            axes[2].axhline(row.baseline_awc_ua, color="0.3", linestyle="--", linewidth=1)
            axes[2].set_xticks(positions, LABELS.values(), rotation=30, ha="right")
            axes[2].set(ylabel="AWC [uA]", title=f"Same MCP reference and same Q grid ({row.common_grid_fraction:.1%} retained)")
            axes[2].legend(fontsize=9)
            axes[2].grid(alpha=.2)
        else:
            axes[2].axis("off")
            reason = ("No charge support common to all variants; no AWC comparison."
                      if ch == 235 else "No saved measured overlap with MCP reference; no AWC comparison.")
            axes[2].text(.5, .5, reason,
                         transform=axes[2].transAxes, ha="center", wrap=True)
        fig.text(.11, .025, "Sensitivity study only: no nominal calibration replaced. SEM errors condition on the reconstructed sample.", fontsize=9)
        save(fig, f"channel_{ch}_diagnostics")


if __name__ == "__main__":
    main()
