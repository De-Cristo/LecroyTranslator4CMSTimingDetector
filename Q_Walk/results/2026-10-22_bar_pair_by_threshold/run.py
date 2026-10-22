"""Independent threshold calibrations and timing widths; never pool thresholds."""
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
from Q_Walk.qwalk_core import decode_thresholds
from Q_Walk.qwalk_downstream_t0_reference import _quality_mask
from Q_Walk.qwalk_extended_t0_reference import selected_inv_q_edges
from Q_Walk.qwalk_run4076_safe_time2_scatter import _histogram_peak_region, _limits
from Q_Walk.qwalk_t0_models import build_comparison_profile, fit_reference_quadratic, gaussian_core_width

OUT = Path(__file__).resolve().parent
CACHE = ROOT / "Q_Walk/results/2026-10-22_clock_quality/cache/upstream_bar8_aligned.csv.gz"
WIDTH_SOURCE = ROOT / "DPS_NOTES/review_2026-10-22_bar8_validation/validate.py"
spec = importlib.util.spec_from_file_location("previous_width", WIDTH_SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
CHANNELS = (137, 150, 234, 243)
COLORS = ("#777777", "#147da0")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(fig, path):
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + ext), dpi=180, facecolor="white")
    plt.close(fig)


def interval(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return dict(p16=float(np.percentile(valid, 16)), p84=float(np.percentile(valid, 84)),
                std_ps=float(np.std(valid, ddof=1)), successful=len(valid))


def histogram(ax, events, stats):
    edges = np.arange(-400, 410, 10)
    table = pd.DataFrame(dict(bin_low_ps=edges[:-1], bin_high_ps=edges[1:]))
    for label, color in zip(("before", "after"), COLORS):
        s = stats[label]
        values = events[label + "_ps"]
        centered = values - s["median_ps"]
        table[label + "_count"] = np.histogram(centered, bins=edges)[0]
        s["outside_display_range"] = int(((centered < edges[0]) | (centered > edges[-1])).sum())
        ax.hist(centered, bins=edges, histtype="step", lw=1.5, color=color,
                label=f"{label.capitalize()}: {s['sigma_ps']:.1f} ps")
        lo, hi = s["fit_low_ps"], s["fit_high_ps"]
        x = np.linspace(lo, hi, 300)
        mu, sigma = s["fit_mean_ps"], s["sigma_ps"]
        norm = ndtr((hi-mu)/sigma)-ndtr((lo-mu)/sigma)
        y = values.between(lo, hi).sum()*10/(norm*sigma*np.sqrt(2*np.pi))*np.exp(-.5*((x-mu)/sigma)**2)
        ax.plot(x-s["median_ps"], y, color=color, ls="--", lw=1)
    ax.set(xlim=(-400, 400), xlabel="Downstream - reference bar [ps], centered", ylabel="Events / 10 ps")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    return table


def run_setting(data, key):
    step1, step2 = map(float, key)
    vth1, vth2, _ = decode_thresholds(np.array([step2]))
    vth1, vth2 = int(vth1[0]), int(vth2[0])
    folder = OUT / f"vth1_{vth1}_vth2_{vth2}"
    folder.mkdir(exist_ok=True)
    assert data.step1.nunique() == data.step2.nunique() == 1
    channel_results, fits, profiles, allowed = {}, {}, {}, []
    cuts_quality, cuts_peak = [], []
    for ch in CHANNELS:
        f = data.loc[data.channelID.eq(ch)].copy()
        prompt = f.loc[_quality_mask(f)]
        peak = _histogram_peak_region(prompt.energy.to_numpy(), _limits(prompt.energy.to_numpy(), .5, 99.5, 20.))
        cut = float(peak["half_max_q_low_adc"])
        chosen = prompt.loc[prompt.energy.ge(cut)]
        profile = build_comparison_profile(chosen, n_bins=20, min_entries=5,
                                          inv_q_edges=selected_inv_q_edges(chosen.energy, 20))
        fit = fit_reference_quadratic(profile)
        assert len(profile) >= 3
        low, high = float(1/fit.x_max), float(1/fit.x_min)
        inv = 1/chosen.energy.to_numpy()
        occupied = np.zeros(len(chosen), dtype=bool)
        for row in profile.itertuples():
            occupied |= (inv >= row.inv_q_bin_low) & (inv <= row.inv_q_bin_high)
        supported = chosen.loc[chosen.energy.between(low, high) & occupied]
        allowed.append(set(supported.event_id))
        cuts_quality.append(set(prompt.event_id))
        cuts_peak.append(set(chosen.event_id))
        record = dict(channel=ch, step1=step1, step2=step2, vth1=vth1, vth2=vth2,
                      quality_hits=len(prompt), selected_hits=len(chosen), q_cut_adc=cut,
                      q_min_adc=low, q_max_adc=high, retained_bins=len(profile),
                      coefficients_ns=dict(zip(("a", "b", "c"), (fit.coefficients[::-1]/1000).tolist())),
                      covariance_ns2_abc=(fit.covariance[::-1, ::-1]/1e6).tolist())
        save(folder / f"channel_{ch}_fit.json", record)
        profile.to_csv(folder / f"channel_{ch}_profile.csv", index=False)
        profiles[ch], fits[ch], channel_results[str(ch)] = profile, fit, record
    ids = set.intersection(*allowed)
    selected = data.loc[data.event_id.isin(ids)]
    assert len(selected) == 4*len(ids)
    t = selected.pivot(index="event_id", columns="channelID", values="time")
    q = selected.pivot(index="event_id", columns="channelID", values="energy")
    r = selected.pivot(index="event_id", columns="channelID", values="delta_t_ref_ps")
    w = {ch: fits[ch].evaluate(1/q[ch].to_numpy()) for ch in CHANNELS}
    events = selected.drop_duplicates("event_id").set_index("event_id").loc[t.index, ["file_id", "segment", "step1", "step2"]]
    events["before_ps"] = ((t[234]-t[137])+(t[243]-t[150]))/2
    events["after_ps"] = events.before_ps-(w[234]+w[243]-w[137]-w[150])/2
    events["mcp_before_ps"] = (r[137]+r[150])/2
    events["mcp_after_ps"] = events.mcp_before_ps-(w[137]+w[150])/2
    assert np.allclose(events.before_ps, ((r[234]-r[137])+(r[243]-r[150]))/2, atol=.001)
    for ch in CHANNELS:
        events[f"q_{ch}_adc"] = q[ch]
    events = events.reset_index()
    events.to_csv(folder / "selected_events.csv.gz", index=False)
    labels = ("before", "after", "mcp_before", "mcp_after")
    stats = {label: previous.width(events[label+"_ps"]) for label in labels}
    assert all(s["status"] == "ok" for s in stats.values())
    groups = [g.index.to_numpy() for _, g in events.groupby(["file_id", "segment"])]
    rng = np.random.default_rng(439920261022 + int(step2))
    values = events[[label+"_ps" for label in labels]].to_numpy()
    replicas = []
    for i in range(400):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        fits_boot = [gaussian_core_width(values[idx, j], min_count=20) for j in range(4)]
        replicas.append([f.sigma_ps if f.status == "ok" else np.nan for f in fits_boot])
    boot = pd.DataFrame(replicas, columns=labels)
    boot["improvement"] = boot.before-boot.after
    boot.to_csv(folder / "paired_block_bootstrap.csv", index=False)
    ci = {col: interval(boot[col]) for col in boot}
    assert min(x["successful"] for x in ci.values()) >= 380
    report = dict(step1=step1, step2=step2, vth1=vth1, vth2=vth2,
                  source_files=sorted(data.file_id.unique()), matched_events=int(data.event_id.nunique()),
                  all_quality_events=len(set.intersection(*cuts_quality)),
                  all_peak_events=len(set.intersection(*cuts_peak)), events=len(events), blocks=len(groups),
                  channels=channel_results, stats=stats, intervals=ci,
                  core_window_sensitivity={str(k): {label: previous.width(events[label+"_ps"], k=k)
                                                  for label in ("before", "after")} for k in (2., 3., 4.)})
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    table = histogram(ax, events, stats)
    ax.set_title(f"CMS Preliminary Phase 2\nRun 4399 | T1={vth1}, T2={vth2} DAC | {len(events):,} events", loc="left")
    export(fig, folder / "before_after")
    table.to_csv(folder / "histogram.csv", index=False)
    # Reference profiles at the lowest scanned threshold are used as slide examples, not chosen by width.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, ch in zip(axes, (137, 150)):
        p, fit = profiles[ch], fits[ch]
        ax.errorbar(p.inv_q_mean, p.delta_t_ref_mean_ps/1000+26,
                    yerr=p.delta_t_ref_sem_ps/1000, fmt="o", ms=4, color="#147da0", capsize=2)
        x = np.linspace(fit.x_min, fit.x_max, 300)
        ax.plot(x, fit.evaluate(x)/1000+26, color="#cf4534")
        ax.set(title=f"Channel {ch}", xlabel="1/Q [1/ADC]", ylabel="Time difference [ns]")
        ax.grid(alpha=.2)
    fig.suptitle(f"CMS Preliminary Phase 2 | Upstream bar 8 | T1={vth1}, T2={vth2} DAC")
    export(fig, folder / "upstream_walk_profiles")
    save(folder / "report.json", report)
    print(f"T1={vth1}: N={len(events)}, before={stats['before']['sigma_ps']:.2f}, after={stats['after']['sigma_ps']:.2f} ps", flush=True)
    return folder, events, report, boot


def main():
    data = pd.read_csv(CACHE)
    assert not data.duplicated(["event_id", "channelID"]).any()
    assert (data.groupby("event_id")[["step1", "step2"]].nunique() == 1).all().all()
    assert (data.groupby("file_id")[["step1", "step2"]].nunique() == 1).all().all()
    runs = [run_setting(group.copy(), key) for key, group in data.groupby(["step1", "step2"], sort=True)]
    assert len(runs) == 6
    rows = []
    event_sets = [set(events.event_id) for _, events, _, _ in runs]
    assert sum(map(len, event_sets)) == len(set.union(*event_sets))
    for _, events, r, boot in runs:
        row = dict(vth1=r["vth1"], vth2=r["vth2"], events=r["events"], blocks=r["blocks"])
        for state in ("before", "after", "mcp_before", "mcp_after"):
            row[state+"_sigma_ps"] = r["stats"][state]["sigma_ps"]
            row[state+"_p16_ps"] = r["intervals"][state]["p16"]
            row[state+"_p84_ps"] = r["intervals"][state]["p84"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "width_summary.csv", index=False)
    weights = (summary.events.to_numpy()-1).astype(float)
    weights /= weights.sum()
    combined = {}
    for state in ("before", "after"):
        widths = summary[state+"_sigma_ps"].to_numpy()
        replicas = np.stack([boot[state].to_numpy() for _, _, _, boot in runs])
        combined[state] = dict(width_rms_ps=float(np.sqrt(np.sum(weights*widths**2))),
                               interval=interval(np.sqrt(np.sum(weights[:, None]*replicas**2, axis=0))))
    combined["definition"] = "sqrt(sum((N_k-1)*sigma_k^2)/sum(N_k-1)); descriptive within-setting RMS, not a common operating-point resolution"
    combined["weights"] = weights.tolist()
    combined["uncertainty"] = "Independent per-setting block-bootstrap replicas combined with fixed event weights; selection/calibration fixed. No calibration/systematic uncertainty."
    save(OUT / "combined_width_summary.json", combined)
    for name, prefix, title in (("width_vs_threshold", "", "Downstream bar 3 - reference bar 8"),
                                ("reference_mcp_vs_threshold", "mcp_", "Reference bar 8 - MCP")):
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for state, color, dx in (("before", COLORS[0], -.18), ("after", COLORS[1], .18)):
            column = prefix+state
            y = summary[column+"_sigma_ps"].to_numpy()
            lo, hi = summary[column+"_p16_ps"].to_numpy(), summary[column+"_p84_ps"].to_numpy()
            x = summary.vth1.to_numpy()+dx
            # Percentile intervals need not bracket the point estimate.
            ax.vlines(x, lo, hi, color=color, lw=2)
            ax.scatter(x, y, color=color, s=45, label=state.capitalize()+" walk correction")
        ax.set(xticks=summary.vth1, xlabel="First threshold T1 [DAC setting]", ylabel="Fitted core width [ps]")
        ax.set_title("CMS Preliminary Phase 2\n"+title+" | T2=60 DAC", loc="left")
        ax.grid(alpha=.2)
        ax.legend()
        export(fig, OUT / name)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (_, events, report, _) in zip(axes.flat, runs):
        histogram(ax, events, report["stats"])
        ax.set_title(f"T1={report['vth1']} DAC | N={len(events):,}")
    fig.suptitle("CMS Preliminary Phase 2 | Run 4399 | T2=60 DAC\nIndependent threshold calibrations and distributions; dashed curves: core fits")
    export(fig, OUT / "before_after_all_thresholds")
    inputs = [Path(__file__), CACHE, WIDTH_SOURCE] + [ROOT/"Q_Walk"/name for name in
              ("qwalk_core.py", "qwalk_downstream_t0_reference.py", "qwalk_extended_t0_reference.py",
               "qwalk_run4076_safe_time2_scatter.py", "qwalk_t0_models.py")]
    save(OUT / "input_manifest.json", [dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    save(OUT / "report.json", dict(settings=[r for _, _, r, _ in runs], combined=combined,
         source=str(CACHE.relative_to(ROOT)), total_selected_events=int(summary.events.sum()),
         method="Group by step1 and step2 before charge selection and four independent channel walk fits. Clock cache already grouped by file/segment, each with one threshold setting.",
         uncertainty="400 paired file/segment bootstrap replicas per setting, fixed calibration and selection; in-sample validation.",
         verification="No repeated event IDs across settings; one setting per event and file; all timing states fit successfully; shared MCP term cancels."))
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
