"""Compare MCP and calibrated-reference walk calibration at one fixed setting."""
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

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_downstream_t0_reference import _quality_mask
from Q_Walk.qwalk_t0_models import build_comparison_profile, fit_reference_quadratic, gaussian_core_width

OLD = ROOT / "Q_Walk/results/2026-10-22_bar_pair_by_threshold/vth1_3_vth2_60"
CACHE = ROOT / "Q_Walk/results/2026-10-22_clock_quality/cache/upstream_bar8_aligned.csv.gz"
WIDTH = ROOT / "DPS_NOTES/review_2026-10-22_bar8_validation/validate.py"
spec = importlib.util.spec_from_file_location("saved_width", WIDTH)
saved_width = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saved_width)
CHANNELS = (137, 150, 234, 243)
COLORS = {"raw": "#777777", "direct": "#2678ad", "transfer": "#cf6634"}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "pdf.fonttype": 42})


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def interval(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    assert len(x) >= 380
    return dict(p16=float(np.percentile(x, 16)), p84=float(np.percentile(x, 84)),
                std_ps=float(np.std(x, ddof=1)), successful=len(x))


def frozen_fit(ch, q, report):
    c = report["channels"][str(ch)]
    assert np.all((q >= c["q_min_adc"]) & (q <= c["q_max_adc"]))
    abc = c["coefficients_ns"]
    return 1000 * (abc["a"] + abc["b"] / q + abc["c"] / q**2)


def prepare_input(report):
    snapshot = OUT / "selected_input.csv.gz"
    if snapshot.exists():
        return pd.read_csv(snapshot)
    data = pd.read_csv(CACHE)
    data = data.loc[data.step1.eq(report["step1"]) & data.step2.eq(report["step2"])].copy()
    sets = []
    for ch in CHANNELS:
        c = report["channels"][str(ch)]
        f = data.loc[data.channelID.eq(ch)]
        keep = _quality_mask(f) & f.energy.ge(c["q_cut_adc"])
        # A finite calibrated reference is required for BOTH comparison paths.
        if ch in (137, 150):
            keep &= f.energy.between(c["q_min_adc"], c["q_max_adc"])
        sets.append(set(f.loc[keep, "event_id"]))
    data = data.loc[data.event_id.isin(set.intersection(*sets))]
    cols = ["event_id", "file_id", "segment", "step1", "step2", "channelID",
            "time", "energy", "slope_ua_per_ns", "delta_t_ref_ps"]
    data = data[cols].copy()
    data.to_csv(snapshot, index=False, compression={"method": "gzip", "mtime": 0})
    return data


def profile(frame, values, edges):
    f = frame.copy()
    f["delta_t_ref_ps"] = values
    p = build_comparison_profile(f, n_bins=20, min_entries=5, inv_q_edges=edges)
    return p, fit_reference_quadratic(p)


def support(q, p, fit):
    x = 1 / q
    occupied = np.zeros(len(q), dtype=bool)
    for row in p.itertuples():
        occupied |= (x >= row.inv_q_bin_low) & (x <= row.inv_q_bin_high)
    return occupied & (x >= fit.x_min) & (x <= fit.x_max)


def histogram(ax, events, stats, keys, labels, colors, xlabel):
    edges = np.arange(-400, 410, 10)
    table = {"bin_low_ps": edges[:-1], "bin_high_ps": edges[1:]}
    for key, label, color in zip(keys, labels, colors):
        s = stats[key]
        x = events[key].to_numpy()
        z = x - s["median_ps"]
        table[key] = np.histogram(z, edges)[0]
        s["outside_display_count"] = int(np.sum((z < edges[0]) | (z > edges[-1])))
        ax.hist(z, bins=edges, histtype="step", lw=1.6, color=color,
                label=f"{label}: {s['sigma_ps']:.1f} ps")
        lo, hi, mu, sigma = (s[k] for k in ("fit_low_ps", "fit_high_ps", "fit_mean_ps", "sigma_ps"))
        grid = np.linspace(lo, hi, 300)
        norm = ndtr((hi-mu)/sigma) - ndtr((lo-mu)/sigma)
        n = np.sum((x >= lo) & (x <= hi))
        y = n * 10 / (norm*sigma*np.sqrt(2*np.pi)) * np.exp(-.5*((grid-mu)/sigma)**2)
        ax.plot(grid-s["median_ps"], y, color=color, ls="--", lw=1)
    ax.set(xlim=(-350, 350), xlabel=xlabel, ylabel="Events / 10 ps")
    ax.grid(alpha=.2)
    ax.legend(fontsize=10, loc="upper right")
    return pd.DataFrame(table)


def main():
    old = json.loads((OLD / "report.json").read_text())
    data = prepare_input(old)
    assert data.groupby("event_id").channelID.apply(lambda x: set(x) == set(CHANNELS)).all()
    assert not data.duplicated(["event_id", "channelID"]).any()
    assert data.step1.nunique() == data.step2.nunique() == 1
    q = data.pivot(index="event_id", columns="channelID", values="energy")
    t = data.pivot(index="event_id", columns="channelID", values="time")
    r = data.pivot(index="event_id", columns="channelID", values="delta_t_ref_ps")
    u = {ch: frozen_fit(ch, q[ch].to_numpy(), old) for ch in (137, 150)}
    ref_mcp = ((r[137]-u[137])+(r[150]-u[150]))/2
    models, profiles, common, shapes = {}, {}, np.ones(len(q), dtype=bool), {}
    for ch in (234, 243):
        f = data.loc[data.channelID.eq(ch)].set_index("event_id").loc[q.index]
        edges = np.linspace((1/f.energy).min(), (1/f.energy).max(), 21)
        for mode in ("direct", "transfer"):
            target = r[ch] if mode == "direct" else r[ch]-ref_mcp
            if mode == "transfer":
                # Subtract integer timestamps before applying sub-ps corrections.
                internal_target = ((t[ch]-t[137])+(t[ch]-t[150])+u[137]+u[150])/2
                assert np.allclose(target, internal_target, atol=.001, rtol=0)
                target = internal_target
            p, fit = profile(f, target, edges)
            profiles[ch, mode], models[ch, mode] = p, fit
            common &= support(q[ch].to_numpy(), p, fit)
            p.to_csv(OUT / f"channel_{ch}_{mode}_profile.csv", index=False)
            save(f"channel_{ch}_{mode}_fit.json", dict(coefficients_ps=fit.coefficients.tolist(),
                 covariance_ps2=fit.covariance.tolist(), inv_q_min=fit.x_min, inv_q_max=fit.x_max,
                 selected_events=len(f), retained_bins=len(p), n_after_clip=int(p.count_after_clip.sum())))
    q, t, r, ref_mcp = q.loc[common], t.loc[common], r.loc[common], ref_mcp.loc[common]
    ref_corr = {ch: r[ch]-frozen_fit(ch, q[ch].to_numpy(), old) for ch in (137, 150)}
    ref_avg = (ref_corr[137]+ref_corr[150])/2
    events = data.drop_duplicates("event_id").set_index("event_id").loc[q.index,
                  ["file_id", "segment", "step1", "step2"]].copy()
    events["pair_raw"] = (r[234]+r[243]-r[137]-r[150])/2
    events["pair_ref_only"] = (r[234]+r[243])/2-ref_avg
    events["ref_lr_raw"] = r[137]-r[150]
    events["ref_lr_corrected"] = ref_corr[137]-ref_corr[150]
    events["dut_lr_raw"] = r[234]-r[243]
    events["ref_mcp_raw"] = (r[137]+r[150])/2
    events["ref_mcp_corrected"] = ref_avg
    events["dut_mcp_raw"] = (r[234]+r[243])/2
    for mode in ("direct", "transfer"):
        corrected = {ch: r[ch]-models[ch, mode].evaluate(1/q[ch].to_numpy()) for ch in (234, 243)}
        events[f"pair_{mode}"] = (corrected[234]+corrected[243])/2-ref_avg
        events[f"dut_mcp_{mode}"] = (corrected[234]+corrected[243])/2
        events[f"dut_lr_{mode}"] = corrected[234]-corrected[243]
        for ch in (234, 243):
            events[f"dut_{ch}_{mode}_mcp"] = corrected[ch]
    for ch in CHANNELS:
        events[f"q_{ch}_adc"] = q[ch]
    events["ref_left_mcp"] = ref_corr[137]
    events["ref_right_mcp"] = ref_corr[150]
    events["transfer_minus_direct"] = events.pair_transfer-events.pair_direct
    events = events.reset_index()
    events.to_csv(OUT / "closure_events.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    keys = ["pair_raw", "pair_ref_only", "pair_direct", "pair_transfer", "ref_lr_raw", "ref_lr_corrected",
            "dut_lr_raw", "dut_lr_direct", "dut_lr_transfer", "ref_mcp_raw", "ref_mcp_corrected",
            "dut_mcp_raw", "dut_mcp_direct", "dut_mcp_transfer"]
    stats = {k: saved_width.width(events[k]) for k in keys}
    assert all(s["status"] == "ok" for s in stats.values())
    print('Common calibration events:', data.event_id.nunique(), 'evaluation events:', len(events), flush=True)
    print({k: round(v['sigma_ps'], 3) for k, v in stats.items()}, flush=True)
    groups = [f.index.to_numpy() for _, f in events.groupby(["file_id", "segment"])]
    rng = np.random.default_rng(20261023)
    values = events[keys].to_numpy()
    rows = []
    for rep in range(400):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        fits = [gaussian_core_width(values[idx, j], min_count=20) for j in range(len(keys))]
        rows.append([f.sigma_ps if f.status == "ok" else np.nan for f in fits])
        if (rep+1) % 100 == 0:
            print(f'Paired block bootstrap: {rep+1}/400', flush=True)
    boot = pd.DataFrame(rows, columns=keys)
    boot["pair_transfer_minus_direct"] = boot.pair_transfer-boot.pair_direct
    boot["dut_mcp_transfer_minus_direct"] = boot.dut_mcp_transfer-boot.dut_mcp_direct
    for mode in ("direct", "transfer"):
        boot[f"lr_proxy_{mode}"] = .5*np.sqrt(boot.ref_lr_corrected**2+boot[f"dut_lr_{mode}"]**2)
    boot.to_csv(OUT / "paired_block_bootstrap.csv", index=False)
    intervals = {k: interval(boot[k]) for k in boot}
    # Reproduce the previous 64.7 ps result separately, before changing the fitting sample.
    historic = pd.read_csv(OLD / "selected_events.csv.gz")
    audit = {k: saved_width.width(historic[k+"_ps"])["sigma_ps"] for k in ("before", "after")}
    assert all(abs(audit[k]-old["stats"][k]["sigma_ps"]) < 1e-6 for k in audit)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)
    for ax, ch, side in zip(axes[:2], (234, 243), ("left", "right")):
        fits = {mode: models[ch, mode] for mode in ("direct", "transfer")}
        lo = max(f.x_min for f in fits.values())
        hi = min(f.x_max for f in fits.values())
        x = np.linspace(lo, hi, 250)
        pivot = float(np.median(1/q[ch]))
        curves = {}
        for mode, label in (("direct", "Direct MCP"), ("transfer", "Via reference bar")):
            fit, p = fits[mode], profiles[ch, mode]
            offset = fit.evaluate(pivot)
            curves[mode] = fit.evaluate(x)-offset
            ax.errorbar(p.inv_q_mean, p.delta_t_ref_mean_ps-offset,
                        yerr=p.delta_t_ref_sem_ps, fmt="o" if mode == "direct" else "s",
                        ms=3, color=COLORS[mode], capsize=2, alpha=.75)
            ax.plot(x, curves[mode], color=COLORS[mode], lw=2, label=label)
        diff = curves["transfer"]-curves["direct"]
        event_x = 1/q[ch].to_numpy()
        event_diff = ((fits['transfer'].evaluate(event_x)-fits['transfer'].evaluate(pivot))-
                      (fits['direct'].evaluate(event_x)-fits['direct'].evaluate(pivot)))
        shapes[str(ch)] = dict(pivot_inv_q=float(pivot), inv_q_min=lo, inv_q_max=hi,
                              maximum_absolute_difference_ps=float(np.max(np.abs(diff))),
                              event_weighted_rms_difference_ps=float(np.sqrt(np.mean(event_diff**2))))
        pd.DataFrame(dict(inv_q=x, direct_centered_ps=curves['direct'], transfer_centered_ps=curves['transfer'],
                          difference_ps=diff)).to_csv(OUT / f"channel_{ch}_shape_comparison.csv", index=False)
        ax.set(title=f"Module under test: {side}", xlabel="1/Q [1/ADC]", ylabel="Walk relative to charge pivot [ps]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    hist = histogram(axes[2], events, stats, ["pair_raw", "pair_direct", "pair_transfer"],
                     ["Before", "Direct MCP", "Via reference"], list(COLORS.values()),
                     r"$t_{\rm DUT}-t_{\rm ref}$ [ps], centered")
    axes[2].set_title(f"Same {len(events):,} events")
    fig.suptitle("CMS Preliminary Phase 2 | T1=3, T2=60 DAC | In-sample closure", fontsize=14)
    export(fig, "transfer_closure")
    hist.to_csv(OUT / "pair_histogram.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, title, cols, labels, colors in (
        (axes[0], "Reference bar", ["ref_lr_raw", "ref_lr_corrected"], ["Before", "MCP calibrated"],
         [COLORS['raw'], COLORS['direct']]),
        (axes[1], "Module under test", ["dut_lr_raw", "dut_lr_direct", "dut_lr_transfer"],
         ["Before", "Direct MCP", "Via reference"], list(COLORS.values()))):
        tab = histogram(ax, events, stats, cols, labels, colors, r"$t_L-t_R$ [ps], centered")
        tab.to_csv(OUT / ("ref_lr_histogram.csv" if ax == axes[0] else "dut_lr_histogram.csv"), index=False)
        ax.set_title(title)
    fig.suptitle(f"CMS Preliminary Phase 2 | T1=3, T2=60 DAC | Same {len(events):,} events", fontsize=14)
    export(fig, "end_difference_check")
    # Exact sample-variance identity, unlike a quadrature of separately fitted core widths.
    covariance = {}
    for mode in ('direct', 'transfer'):
        cols = ['ref_left_mcp', 'ref_right_mcp', f'dut_234_{mode}_mcp', f'dut_243_{mode}_mcp']
        cov = np.cov(events[cols].to_numpy(), rowvar=False, ddof=1)
        a = np.array([-.5, -.5, .5, .5])
        variance = float(a @ cov @ a)
        proxy2 = float((np.var(events.ref_lr_corrected, ddof=1)+np.var(events[f'dut_lr_{mode}'], ddof=1))/4)
        cross_bar = float(np.cov(events.ref_mcp_corrected, events[f'dut_mcp_{mode}'], ddof=1)[0, 1])
        missing = float(cov[0, 1]+cov[2, 3]-2*cross_bar)
        assert abs(variance-np.var(events[f'pair_{mode}'], ddof=1)) < 1e-5
        assert abs(variance-proxy2-missing) < 1e-5
        covariance[mode] = dict(channel_order=cols, covariance_ps2=cov.tolist(), pair_variance_ps2=variance,
                                lr_proxy_variance_ps2=proxy2, covariance_remainder_ps2=missing)
    report = dict(thresholds=dict(vth1=3, vth2=60, step1=old['step1'], step2=old['step2']),
                  calibration_events=int(data.event_id.nunique()), evaluation_events=len(events), blocks=len(groups),
                  original_1461_event_widths_ps=audit, stats=stats, intervals=intervals, shape_comparison=shapes,
                  lr_core_proxy_ps={mode: float(.5*np.hypot(stats['ref_lr_corrected']['sigma_ps'],
                                                       stats[f'dut_lr_{mode}']['sigma_ps'])) for mode in ('direct', 'transfer')},
                  covariance_check=covariance,
                  uncertainty='400 paired file/segment block-bootstrap replicas; frozen calibration and selection. Conditional statistical intervals, not calibration or systematic uncertainties.',
                  limitations='In-sample closure; reference calibration is shared by both routes. No independent validation, no guarantee for all DUT modules, no extracted single-bar resolution.',
                  selection='One threshold. Original quality/charge cuts for all four channels, frozen reference fit support, identical downstream fitting input and bin edges for both routes. Route-specific 5 scaled MAD clipping. Evaluation restricted to occupied bins and common fitted support; no extrapolation.',
                  algebra='transfer target = (t_DUT - MCP) - (t_ref_corrected - MCP) = t_DUT - t_ref_corrected; MCP cancels event by event.',
                  display='One global median per timing distribution only; no per-file or per-segment centering. Curve shapes shown after separate constant subtraction at the same per-channel median inverse-Q pivot.')
    save('report.json', report)
    inputs = [Path(__file__), OUT/'selected_input.csv.gz', OLD/'report.json', OLD/'selected_events.csv.gz', WIDTH,
              ROOT/'Q_Walk/qwalk_t0_models.py', ROOT/'Q_Walk/qwalk_downstream_t0_reference.py']
    save('input_manifest.json', [dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    save('verification.json', dict(single_threshold=True, same_event_comparison=True, no_extrapolation=True,
                                  mcp_cancels_in_transfer_target=True, historical_width_reproduced=True,
                                  sample_covariance_identity=True, all_width_fits_ok=True,
                                  successful_bootstrap_min=min(x['successful'] for x in intervals.values())))
    print('Shape closure:', shapes, flush=True)
    print('Width differences:', {k:v for k,v in intervals.items() if 'minus' in k}, flush=True)
    print('LR proxies:', report['lr_core_proxy_ps'], flush=True)


if __name__ == '__main__':
    main()
