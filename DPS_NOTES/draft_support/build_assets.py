"""Build review figures from saved results; never rerun or modify the analysis."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPS = Path(__file__).resolve().parents[1]
ROOT = DPS.parent
OVERLAP = ROOT / "Q_Walk/results/2026-09-24_measured_q_overlap"
LEGACY = ROOT / "Q_Walk/reference_quadratic_t0"
CMS = ROOT.parent / "CMS-Note/cmsnote"
N1 = DPS / "note1_mcp_reference"
N2 = DPS / "note2_dual_threshold"
MANIFEST = []
plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "axes.formatter.useoffset": False,
})


def snapshot(source, note):
    target = note / "data" / source.relative_to(ROOT.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def register(note, name, sources, kind):
    MANIFEST.append({
        "note": note.name, "figure": f"figures/{name}.pdf", "kind": kind,
        "sources": [
            {"original": str(p.relative_to(ROOT.parent)),
             "snapshot": str(snapshot(p, note).relative_to(note)),
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sources
        ],
    })


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")


def save(fig, note, name, sources, kind="redrawn from saved numerical results"):
    fig.text(.105, .965, r"$\bf{CMS}$ Preliminary Phase 2",
             ha="left", va="top", fontsize=14)
    for suffix in ("pdf", "png"):
        fig.savefig(note / f"figures/{name}.{suffix}", dpi=170)
    plt.close(fig)
    register(note, name, sources, kind)


def canvas():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.87)
    ax.grid(alpha=.2)
    return fig, ax


def legacy_image(filename, name):
    path = CMS / filename
    img = plt.imread(path)
    height = min(7., max(3.8, 8.2 * img.shape[0] / img.shape[1] + .55))
    fig = plt.figure(figsize=(8.2, height))
    ax = fig.add_axes([.02, .02, .96, .86])
    ax.imshow(img)
    ax.axis("off")
    save(fig, N1, name, [path], "legacy raster retained; data not refitted")


def reference_profiles():
    rows = []
    for ch, bar in ((133, 7), (154, 7), (137, 8), (150, 8)):
        base = f"ch192_vs_trig_lowess_bar{bar}_ch{ch}_walk_before_walk_fit_expected_refit"
        pp, cp = LEGACY / f"{base}_profile.csv", LEGACY / f"{base}_coeffs.csv"
        p = read_csv(pp)
        with cp.open() as stream:
            coefficients = next(r for r in csv.DictReader(stream) if r["fit_kind"] == "binned")
        coef = [float(coefficients[k]) for k in ("p2", "p1", "p0")]
        x = np.linspace(p["invE_mean"].min(), p["invE_mean"].max(), 400)
        fig, ax = canvas()
        ax.errorbar(p["invE_mean"], p["delta_t_mean"], yerr=p["delta_t_sem"],
                    fmt="o", ms=4.5, capsize=2, color="#2375aa", label="Profile mean +/- SEM")
        ax.plot(x, np.polyval(coef, x), color="#c03937", lw=2,
                label="Saved quadratic in 1/Q")
        ax.set(xlabel=r"$1/Q$ [ADC$^{-1}$]", ylabel=r"Time difference [ns]",
               title=f"Run 4399 | Bar {bar}, channel {ch} | {len(p)} retained bins")
        ax.legend(loc="best")
        save(fig, N1, f"reference_ch{ch}", [pp, cp])
        rows.append({"channel": ch, "bar": bar, "bins": len(p),
                     "a_ns": coef[2], "b_ns_adc": coef[1], "c_ns_adc2": coef[0]})
    with (N1 / "data/reference_coefficients.csv").open("w") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def slope_profile(ch):
    folder = OVERLAP / f"inputs/channel_{ch}"
    sp = next(folder.glob("*linear_summary.json"))
    bp = next(folder.glob("*bootstrap_coefficients.npz"))
    pp = folder / "run4076_profile.csv"
    s, p = json.loads(sp.read_text()), read_csv(pp)
    fit = s["linear_profile_fit"]
    c0, c1 = fit["intercept_ua_per_ns"], fit["coefficient_ua_per_ns_per_adc"]
    q = np.linspace(p["q_mean_adc"].min(), p["q_mean_adc"].max(), 400)
    with np.load(bp) as z:
        c = z["coefficients"]
    low, high = np.percentile(c[:, [0]] + c[:, [1]] * q, [16, 84], axis=0)
    fig, ax = canvas()
    ax.errorbar(p["q_mean_adc"], p["slope_mean_ua_per_ns"], yerr=p["slope_sd_ua_per_ns"],
                fmt="o", color="black", ms=4, capsize=3, label="Mean +/- SD of hits")
    ax.fill_between(q, low, high, color="#2375aa", alpha=.3, label="68% bootstrap interval")
    ax.plot(q, c0 + c1*q, color="#2375aa", lw=2, label="SEM-weighted linear fit")
    ax.set(xlabel="Q [ADC]", ylabel=r"$S=\Delta I/\Delta t_{12}$ [$\mu$A/ns]",
           title=f"Run 4076 | Channel {ch}")
    ax.legend(loc="lower right")
    save(fig, N2, f"slope_ch{ch}", [sp, bp, pp])


def spectrum():
    path = OVERLAP / "energy_spectra/channel_137_energy_spectrum.csv"
    p = read_csv(path)
    edges = np.r_[p["q_low_adc"], p["q_high_adc"][-1]]
    fig, ax = canvas()
    for run, color in (("4076", "#2375aa"), ("4399", "#c03937")):
        counts = p[f"run{run}_quality_count"]
        ax.stairs(counts / counts.sum(), edges, color=color, lw=1.7, label=f"Run {run}: quality sample")
        counts = p[f"run{run}_peak_right_count"]
        # Same normalization as the corresponding pre-Q-cut sample.
        ax.stairs(counts / p[f"run{run}_quality_count"].sum(), edges,
                  color=color, fill=True, alpha=.14, label=f"Run {run}: after Q cut")
    ax.set(xlabel="Q [ADC]", ylabel="Fraction of quality sample / bin",
           title="Channel 137 | Separate runs, separate event samples")
    ax.legend()
    save(fig, N2, "q_spectra_ch137", [path])


def comparison(ch):
    path = OVERLAP / f"channel_{ch}_comparison.csv"
    rp = OVERLAP / f"channel_{ch}.json"
    p, r = read_csv(path), json.loads(rp.read_text())
    fig, (ax, residual) = plt.subplots(2, 1, figsize=(8.2, 6.3), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(left=.13, right=.97, bottom=.12, top=.88, hspace=.1)
    q = p["q_adc"]
    # Preserve absent support intervals instead of drawing lines through them.
    jumps = np.flatnonzero(np.diff(q) > 1.5 * (r["overlap_max"]-r["overlap_min"]) / 2999) + 1
    def gapped(values):
        return np.insert(values, jumps, np.nan)
    x = gapped(q)
    offset = p["reference_ns"].mean()
    ax.plot(x, gapped(p["reference_ns"]-offset), color="black", lw=2,
            label="MCP quadratic (run 4399)")
    ax.plot(x, gapped(p["mapped_ns"]-offset), color="#c03937", lw=2,
            label="AWC/S(Q) + offset (run 4076)")
    ax.fill_between(x, gapped(p["mapped_p16_ns"]-offset), gapped(p["mapped_p84_ns"]-offset),
                    color="#c03937", alpha=.25, label="68% conditional bootstrap")
    ax.set(ylabel="Relative walk [ns]", title=f"Channel {ch} | Joint measured support only")
    ax.legend(loc="best", fontsize=9)
    residual.axhline(0, color="black", lw=.8)
    residual.plot(x, gapped(1000*(p["mapped_ns"]-p["reference_ns"])), color="#c03937")
    residual.set(xlabel="Q [ADC]", ylabel="Difference\n[ps]")
    for axis in (ax, residual):
        axis.grid(alpha=.2)
    save(fig, N2, f"comparison_ch{ch}", [path, rp],
         "saved curve comparison; lower panel is model-minus-model, not event resolution")


def awc_summary():
    path = OVERLAP / "awc_summary_special_10_channels_no_ch240_linear.csv"
    p = read_csv(path)
    fitpath = OVERLAP / "awc_summary_special_10_channels_no_ch240_linear_fit.json"
    fit = json.loads(fitpath.read_text())
    fig, ax = canvas()
    x, y = 1 / p["c1"], p["awc_ua"]
    # Draw percentile endpoints directly: the central fit need not equal the median.
    ax.vlines(x, p["awc_p16"], p["awc_p84"], color="#2375aa", lw=1.5)
    ax.scatter(x, y, color="#2375aa", s=40, label="Upstream channels (selected subset)")
    offsets = {135: (-23, -12), 159: (5, 10), 134: (-23, 5), 156: (5, 9), 137: (4, -14)}
    for i, (xx, yy, ch) in enumerate(zip(x, y, p["channel"])):
        ax.annotate(str(int(ch)), (xx, yy), xytext=offsets.get(int(ch), (4, 7 if i % 2 else -14)),
                    textcoords="offset points", fontsize=9)
    # Read the saved fit, not a new fit to a potentially different weighting.
    a, b = fit["intercept_ua"], fit["slope"]
    xx = np.linspace(0, x.max()*1.06, 200)
    ax.plot(xx, a+b*xx, color="#c03937", label="Saved weighted line")
    ax.set(xlabel=r"$1/c_1$ [ADC ns/$\mu$A]", ylabel=r"Reference-fitted AWC [$\mu$A]",
           title="Exploratory ten-channel subset | Linear axes", ylim=(0, p["awc_p84"].max()*1.14))
    ax.legend(loc="upper left", fontsize=9)
    save(fig, N2, "awc_summary_selected", [path, fitpath])
    path = OVERLAP / "channels.csv"
    p = read_csv(path)
    fig, ax = canvas()
    for module, marker, color in (("upstream", "o", "#2375aa"), ("downstream", "s", "#c03937")):
        # genfromtxt defaults to numeric, so use the channel number for the module.
        part = p[p["channel"] < 192] if module == "upstream" else p[p["channel"] >= 192]
        x = 1 / part["c1"]
        ax.vlines(x, part["awc_p16"], part["awc_p84"], color=color, alpha=.65)
        ax.scatter(x, part["awc_ua"], marker=marker, color=color, s=35, label=module.title())
        for xx, yy, ch in zip(x, part["awc_ua"], part["channel"]):
            if int(ch) in (130, 236, 239):
                ax.annotate(str(int(ch)), (xx, yy), xytext=(3, 4), textcoords="offset points", fontsize=9)
    ax.set(xlabel=r"$1/c_1$ [ADC ns/$\mu$A]", ylabel=r"Reference-fitted AWC [$\mu$A]",
           title="All 21 completed overlap fits | Linear axes")
    ax.legend()
    save(fig, N2, "awc_summary_all", [path])


def predictor():
    path = OVERLAP / "physical_candidate_quick_test/channel_predictions.csv"
    p = read_csv(path)
    fig, ax = canvas()
    x, y = p["awc_ua"], p["physical_predicted_awc_lobo"]
    ax.hlines(y, p["awc_p16"], p["awc_p84"], color="#2375aa", alpha=.6)
    ax.scatter(x, y, color="#2375aa", s=40)
    for xx, yy, ch in zip(x, y, p["channel"]):
        ax.annotate(str(int(ch)), (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=9)
    lim = max(x.max(), y.max())*1.1
    ax.plot([0, lim], [0, lim], "--", color="0.4", label="Equal prediction and reference")
    ax.set(xlabel=r"Reference-fitted AWC [$\mu$A]", ylabel=r"Held-out-bar prediction [$\mu$A]",
           xlim=(0, lim), ylim=(0, lim), title="Exploratory internal predictor | Median Q*")
    ax.legend(loc="upper left")
    save(fig, N2, "internal_predictor", [path, path.with_name("model_comparison.json")])


def main():
    for note in (N1, N2):
        for folder in ("figures", "data", "build"):
            (note / folder).mkdir(parents=True, exist_ok=True)
        for filename in ("CMSlogo_black_label_1024.png", "LogoOutline-Black.pdf"):
            shutil.copy2(DPS / "cms_dp_note_template" / filename, note / filename)
        for filename in ("slides_preamble.tex", "cover.tex"):
            shutil.copy2(DPS / "draft_support" / filename, note / filename)
    for filename, name in (
        ("setup.jpg", "setup"), ("mcpwave.jpg", "mcp_waveform"),
        ("photonishpkdif.jpg", "mcp_pair_timing"),
        ("event_matching_validation.png", "event_matching"),
        ("spill.jpg", "clock_relation"), ("residuals.jpg", "clock_residuals"),
        ("sipdif.jpg", "bar_end_difference"), ("bar-mcp.jpg", "bar_mcp_difference"),
        ("Multi-clock-phases.png", "clock_edges"), ("clock-jitters.png", "clock_estimators"),
    ):
        legacy_image(filename, name)
    reference_profiles()
    for ch in (133, 154, 137, 150):
        slope_profile(ch)
    spectrum()
    for ch in (137, 154, 159, 242):
        comparison(ch)
    awc_summary()
    predictor()
    for note in (N1, N2):
        (note / "figure_manifest.json").write_text(json.dumps(
            [r for r in MANIFEST if r["note"] == note.name], indent=2) + "\n")
    print(f"Prepared {len(MANIFEST)} figures with source snapshots.")


if __name__ == "__main__":
    main()
