"""Same-event bar-to-bar walk validation with fixed four-channel calibrations."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_t0_models import gaussian_core_width

OUT = Path(__file__).resolve().parent
CHECK = ROOT / "Q_Walk/results/2026-10-22_bar_pair_reference_check"
CACHE = ROOT / "Q_Walk/results/2026-10-22_clock_quality/cache/upstream_bar8_aligned.csv.gz"
WIDTH_SOURCE = ROOT / "DPS_NOTES/review_2026-10-22_bar8_validation/validate.py"
spec = importlib.util.spec_from_file_location("previous_validation", WIDTH_SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
CHANNELS = (137, 150, 234, 243)
COLORS = ("#707070", "#147da0")


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    ids = pd.read_csv(CHECK / "common_supported_event_ids.csv").event_id
    data = pd.read_csv(CACHE)
    selected = data.loc[data.event_id.isin(ids)].copy()
    assert len(selected) == 4 * len(ids)
    assert not selected.duplicated(["event_id", "channelID"]).any()
    t = selected.pivot(index="event_id", columns="channelID", values="time")
    q = selected.pivot(index="event_id", columns="channelID", values="energy")
    w = {}
    for ch in CHANNELS:
        meta = json.loads((CHECK / f"channel_{ch}_candidate.json").read_text())
        c = meta["coefficients_ns"]
        assert q[ch].between(meta["q_min_adc"], meta["q_max_adc"]).all()
        w[ch] = 1000 * (c["a"] + c["b"] / q[ch] + c["c"] / q[ch]**2)
    frame = selected.drop_duplicates("event_id").set_index("event_id").loc[t.index, ["file_id", "segment"]]
    # Subtract large absolute timestamps before averaging to retain precision.
    frame["before_ps"] = ((t[234] - t[137]) + (t[243] - t[150])) / 2
    frame["walk_difference_ps"] = (w[234] + w[243] - w[137] - w[150]) / 2
    frame["after_ps"] = frame.before_ps - frame.walk_difference_ps
    for ch in CHANNELS:
        frame[f"q_{ch}_adc"] = q[ch]
    residual = selected.pivot(index="event_id", columns="channelID", values="delta_t_ref_ps").loc[t.index]
    equivalent = ((residual[234] - residual[137]) + (residual[243] - residual[150])) / 2
    cancellation_error = float(np.max(np.abs(frame.before_ps - equivalent)))
    assert cancellation_error < .001
    assert np.isfinite(frame[["before_ps", "after_ps"]]).all().all()
    frame = frame.reset_index()
    frame.to_csv(OUT / "selected_events.csv.gz", index=False)
    stats = {label: previous.width(frame[f"{label}_ps"]) for label in ("before", "after")}
    assert all(s["status"] == "ok" for s in stats.values())
    groups = [g.index.to_numpy() for _, g in frame.groupby(["file_id", "segment"])]
    rng = np.random.default_rng(439920261022)
    samples = []
    for i in range(400):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        row = {"replica": i}
        for label in ("before", "after"):
            fit = gaussian_core_width(frame[f"{label}_ps"].to_numpy()[idx], min_count=20)
            row[f"{label}_sigma_ps"] = fit.sigma_ps if fit.status == "ok" else np.nan
        samples.append(row)
        if (i + 1) % 100 == 0:
            print(f"Bootstrap {i+1}/400", flush=True)
    boot = pd.DataFrame(samples)
    boot["improvement_ps"] = boot.before_sigma_ps - boot.after_sigma_ps
    boot["reduction_percent"] = 100 * boot.improvement_ps / boot.before_sigma_ps
    boot.to_csv(OUT / "paired_block_bootstrap.csv", index=False)
    intervals = {}
    for col in ("before_sigma_ps", "after_sigma_ps", "improvement_ps", "reduction_percent"):
        v = boot[col].dropna()
        intervals[col] = dict(p16=float(v.quantile(.16)), p84=float(v.quantile(.84)), successful=len(v))
    sensitivity = {str(k): {label: previous.width(frame[f"{label}_ps"], k=k)
                           for label in ("before", "after")} for k in (2., 3., 4.)}
    report = dict(events=len(frame), blocks=len(groups), channels=list(CHANNELS),
                  definition="((t234+t243)-(t137+t150))/2; corrected = before - (W234+W243-W137-W150)/2",
                  stats=stats, paired_block_bootstrap=intervals, core_window_sensitivity=sensitivity,
                  core_fit="truncated-normal maximum likelihood within median +/- 2.5 scaled MAD, separately per state",
                  uncertainty="400 paired file/segment block replicas; fixed calibrations and selection; conditional statistical intervals only",
                  shared_reference_cancellation_max_abs_ps=cancellation_error,
                  selection="Same 8737 events, four-channel quality, peak-right cuts and populated fit support; no final timing-tail cut",
                  caveat="In-sample validation; not a single-bar resolution measurement. Calibration/systematic uncertainties not propagated.")
    report["core_width_reduction_percent"] = float(100 * (1 - stats["after"]["sigma_ps"] / stats["before"]["sigma_ps"]))
    report["interpretation"] = "Distribution narrows and charge dependence decreases, but it remains asymmetric; core-window dependence exceeds the conditional statistical precision."
    profiles = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, ch in zip(axes.flat, CHANNELS):
        bins = pd.qcut(frame[f"q_{ch}_adc"], 10, labels=False, duplicates="drop")
        for label, color in zip(("before", "after"), COLORS):
            records = []
            for index, group in frame.groupby(bins):
                y = group[f"{label}_ps"] - stats[label]["median_ps"]
                record = dict(channel=ch, state=label, bin=int(index), n=len(group),
                              q_mean_adc=float(group[f"q_{ch}_adc"].mean()),
                              mean_ps=float(y.mean()), sem_ps=float(y.std(ddof=1)/np.sqrt(len(y))))
                records.append(record)
                profiles.append(record)
            p = pd.DataFrame(records)
            ax.errorbar(p.q_mean_adc, p.mean_ps, yerr=p.sem_ps, fmt="o-", capsize=3,
                        color=color, label=label.capitalize())
        ax.axhline(0, color="black", lw=.7)
        ax.set(title=f"Channel {ch}", xlabel="Q [ADC]", ylabel="Mean bar difference [ps], centered")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("CMS Preliminary Phase 2\nSame-event charge profiles; mean +/- SEM, no timing clipping")
    export(fig, "charge_dependence")
    pd.DataFrame(profiles).to_csv(OUT / "charge_profiles.csv", index=False)
    segment_rows = []
    for (file_id, segment), group in frame.groupby(["file_id", "segment"]):
        row = dict(file_id=file_id, segment=int(segment), n=len(group))
        for label in ("before", "after"):
            row[f"{label}_median_ps"] = float(group[f"{label}_ps"].median())
            row[f"{label}_sd_ps"] = float(group[f"{label}_ps"].std(ddof=1))
        segment_rows.append(row)
    pd.DataFrame(segment_rows).to_csv(OUT / "segment_statistics.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    edges = np.arange(-600, 620, 20)
    histogram = pd.DataFrame({"bin_low_ps": edges[:-1], "bin_high_ps": edges[1:]})
    report["outside_display_range"] = {}
    for label, color in zip(("before", "after"), COLORS):
        s = stats[label]
        centered = frame[f"{label}_ps"] - s["median_ps"]
        histogram[f"{label}_count"] = np.histogram(centered, bins=edges)[0]
        report["outside_display_range"][label] = int(((centered < edges[0]) | (centered > edges[-1])).sum())
        ax.hist(centered, bins=edges, histtype="step", lw=1.8, color=color,
                label=f"{label.capitalize()} walk: core sigma = {s['sigma_ps']:.2f} ps")
        lo, hi = s["fit_low_ps"], s["fit_high_ps"]
        x = np.linspace(lo, hi, 400)
        mu, sig = s["fit_mean_ps"], s["sigma_ps"]
        ncore = np.count_nonzero(frame[f"{label}_ps"].between(lo, hi))
        norm = ndtr((hi-mu)/sig)-ndtr((lo-mu)/sig)
        y = ncore * 20 / (norm*sig*np.sqrt(2*np.pi)) * np.exp(-.5*((x-mu)/sig)**2)
        ax.plot(x-s["median_ps"], y, color=color, ls="--", lw=1.5)
    ci = intervals["after_sigma_ps"]
    annotation = (f"After: 68% block interval [{ci['p16']:.2f}, {ci['p84']:.2f}] ps\n"
                  f"Full-sample SD: {stats['before']['rms_about_mean_ps']:.2f} -> {stats['after']['rms_about_mean_ps']:.2f} ps\n"
                  "Dashed: truncated-normal core fits; fixed calibrations")
    ax.text(.03, .97, annotation, transform=ax.transAxes, va="top", fontsize=10)
    ax.set(xlabel=r"$t_{\rm downstream\ bar\ 3}-t_{\rm reference\ bar\ 8}$ [ps], median centered",
           ylabel="Events / 20 ps", xlim=(-600, 600))
    ax.set_ylim(0, ax.get_ylim()[1]*1.27)
    ax.grid(alpha=.18)
    ax.legend(loc="upper right", bbox_to_anchor=(1, .79), fontsize=10)
    ax.set_title("CMS Preliminary Phase 2\nRun 4399 | 8,737 common events | Before and after walk correction", loc="left", pad=12)
    export(fig, "bar_pair_before_after")
    histogram.to_csv(OUT / "histogram_bins.csv", index=False)
    save_json("report.json", report)
    inputs = [Path(__file__), CACHE, WIDTH_SOURCE, ROOT / "Q_Walk/qwalk_t0_models.py",
              CHECK / "common_supported_event_ids.csv"]
    inputs += [CHECK / f"channel_{ch}_candidate.json" for ch in CHANNELS]
    save_json("input_manifest.json", [dict(path=str(p.relative_to(ROOT)),
              sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    print(json.dumps(dict(stats=stats, intervals=intervals), indent=2), flush=True)


if __name__ == "__main__":
    main()
