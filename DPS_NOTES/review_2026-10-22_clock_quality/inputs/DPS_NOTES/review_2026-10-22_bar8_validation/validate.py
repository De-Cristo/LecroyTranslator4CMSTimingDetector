"""N1-5 validation using frozen references; production results are never changed."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_t0_data import (
    read_synchronized_file, reconstruct_reference_delta_t,
    robust_centered_line, _robust_lowess_correction,
)
from Q_Walk.qwalk_t0_models import (
    build_comparison_profile, fit_reference_quadratic, gaussian_core_width,
    robust_sigma,
)

CHANNELS = (137, 150)
SOURCE = ROOT / "Q_Walk/results/2026-09-24_measured_q_overlap/inputs"
SEED = 439920261022


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def snapshot_inputs():
    paths = [ROOT / p for p in (
        "Q_Walk/qwalk_t0_data.py", "Q_Walk/qwalk_t0_models.py",
        "Q_Walk/qwalk_core.py", "Q_Walk/qwalk_channels.py",
        "Q_Walk/qwalk_extended_t0_reference.py",
    )]
    for ch in CHANNELS:
        stem = f"run4399_upstream_bar8_ch{ch}_t0_quadratic"
        paths += [SOURCE / f"channel_{ch}" / f"{stem}{suffix}"
                  for suffix in (".json", "_profile.csv")]
    rows = []
    for source in paths:
        target = OUT / "inputs" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        rows.append(dict(source=str(source.relative_to(ROOT)),
                         snapshot=str(target.relative_to(OUT)),
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    write_json("input_manifest.json", rows)


def reference(ch):
    base = OUT / "inputs/Q_Walk/results/2026-09-24_measured_q_overlap/inputs" / f"channel_{ch}"
    stem = f"run4399_upstream_bar8_ch{ch}_t0_quadratic"
    return json.loads((base / f"{stem}.json").read_text()), pd.read_csv(base / f"{stem}_profile.csv")


def quality(f):
    mask = np.isfinite(f.delta_t_ref_ps)
    for col in ("energy", "slope_ua_per_ns", "delta_i_ua", "delta_t12_ps"):
        mask &= np.isfinite(f[col]) & f[col].gt(0)
    return mask.to_numpy()


def read_aligned(path):
    return reconstruct_reference_delta_t(read_synchronized_file(path, bars=(8,)))


def extract(limit=None):
    paths = sorted(Path("/eos/user/l/lichengz/MTD/TB2025/sync_out_4399_example").glob("*.root"),
                   key=lambda p: int(re.search(r"4399_(\d+)", p.name)[1]))
    if limit:
        paths = paths[:limit]
    snapshot_inputs()
    records, frames = [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (path, frame) in enumerate(zip(paths, pool.map(read_aligned, paths))):
            frames.append(frame)
            records.append(dict(path=str(path), size_bytes=path.stat().st_size,
                                mtime_ns=path.stat().st_mtime_ns, matched_events=frame.event_id.nunique()))
            print(f"{i+1}/{len(paths)} {path.name}: {frame.event_id.nunique()} matched events", flush=True)
    data = pd.concat(frames, ignore_index=True)
    assert not data.duplicated(["event_id", "channelID"]).any()
    data.to_csv(OUT / "matched_hits.csv.gz", index=False)
    write_json("root_inputs.json", records)
    print("Extraction complete:", data.event_id.nunique(), "events", flush=True)


def curve(ch, q):
    meta, _ = reference(ch)
    a, b, c = [meta["coefficients_ns"][k] for k in ("a", "b", "c")]
    return 1000 * (a + b / q + c / q**2)


def width(values, k=2.5):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    fit = gaussian_core_width(x, core_nsigma=k, min_count=20)
    med, scale = float(np.median(x)), robust_sigma(x)
    lo, hi = med-k*scale, med+k*scale
    result = dict(asdict(fit), n=len(x), rms_about_mean_ps=float(np.std(x, ddof=1)),
                  median_ps=med, scaled_mad_ps=scale, fit_low_ps=lo, fit_high_ps=hi,
                  outside_fit_fraction=float(np.mean((x<lo)|(x>hi))))
    if fit.status == "ok":
        core = x[(x>=lo)&(x<=hi)]
        s = fit.sigma_ps
        def objective(mu):
            norm = ndtr((hi-mu)/s)-ndtr((lo-mu)/s)
            return len(core)*np.log(max(norm,1e-300)) + .5*np.sum(((core-mu)/s)**2)
        result["fit_mean_ps"] = float(minimize_scalar(objective, bounds=(lo,hi), method="bounded").x)
    return result


def bootstrap(frame, n=400):
    rng = np.random.default_rng(SEED)
    groups = [g.index.to_numpy() for _,g in frame.groupby(["file_id", "segment"])]
    data = frame[["D_corrected", "R_corrected"]].to_numpy()
    result = {}
    for mode in ("event", "segment_block"):
        replicas = []
        for _ in range(n):
            idx = (rng.integers(len(frame), size=len(frame)) if mode=="event" else
                   np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]))
            vals = []
            for j in (0,1):
                fit = gaussian_core_width(data[idx,j], min_count=20)
                vals.append(fit.sigma_ps if fit.status=="ok" else np.nan)
            replicas.append(vals)
        arr = np.asarray(replicas)
        pd.DataFrame(arr,columns=["D_sigma_ps","R_sigma_ps"]).to_csv(OUT/f"bootstrap_{mode}.csv",index=False)
        result[mode] = {label: dict(p16=float(np.nanpercentile(arr[:,j],16)),
                                      p84=float(np.nanpercentile(arr[:,j],84)),
                                      successful=int(np.isfinite(arr[:,j]).sum()))
                        for j,label in enumerate(("D","R"))}
        print("Bootstrap",mode,result[mode],flush=True)
    return result


def crossfit_clock(data):
    result = pd.Series(np.nan,index=data.index)
    for _, segment in data.groupby(["file_id","segment"]):
        events = segment.drop_duplicates("event_id").sort_values("entry")
        x = events.mcp_trigger_time.to_numpy()
        y = events.ch192_time.to_numpy()
        expected = np.full(len(events),np.nan)
        for fold in (0,1):
            train = np.arange(len(events))%2 != fold
            test = ~train
            if train.sum()<8:
                continue
            prediction,resid,inlier,_ = robust_centered_line(x[train],y[train])
            # The robust line prediction is affine. Interpolate in centered coordinates.
            order = np.argsort(x[train])
            xt, yt = x[train][order], prediction[order]
            slope = (yt[-1]-yt[0])/(xt[-1]-xt[0])
            base = yt[0]+slope*(x[test]-xt[0])
            correction,_ = _robust_lowess_correction(x[train],resid,inlier,x[test],frac=.15,n_iter=3)
            within = (x[test]>=xt[0])&(x[test]<=xt[-1])
            expected[test] = np.where(within,base+correction,np.nan)
        mapping = dict(zip(events.event_id,expected))
        exp = segment.event_id.map(mapping)
        result.loc[segment.index] = (segment.time-exp)-(segment.mcp_peak_time-segment.mcp_trigger_time)
    return result


def analyze(nboot=400):
    data = pd.read_csv(OUT/"matched_hits.csv.gz")
    report = dict(input_files=len(json.loads((OUT/"root_inputs.json").read_text())),
                  matched_events=int(data.event_id.nunique()),reference_reproduction={},
                  selection="exact {137,150,234,243,192}, unique MCP; both channels quality and saved measured Q support; no validation timing clip")
    parts = {}
    for ch in CHANNELS:
        f = data.loc[data.channelID.eq(ch)].copy()
        meta,p = reference(ch)
        f["quality"] = quality(f)
        f["qcut"] = f.energy.ge(meta["selection"]["q_cut_adc"])
        support = np.zeros(len(f),bool)
        for row in p.itertuples():
            support |= (1/f.energy>=row.inv_q_bin_low)&(1/f.energy<=row.inv_q_bin_high)
        support &= f.energy.between(meta["fit_domain"]["q_min_adc"],meta["fit_domain"]["q_max_adc"])
        f["support"] = support
        selected = f.loc[f.quality & f.qcut]
        edges = np.linspace(1/selected.energy.max(),1/selected.energy.min(),21)
        rebuilt = build_comparison_profile(selected,inv_q_edges=edges,min_entries=5)
        coeff = fit_reference_quadratic(rebuilt).coefficients
        expected = 1000*np.array([meta["coefficients_ns"][k] for k in ("c","b","a")])
        maxdiff = float(np.max(abs(rebuilt.delta_t_ref_mean_ps.to_numpy()-p.delta_t_ref_mean_ps.to_numpy()))) if len(rebuilt)==len(p) else None
        report["reference_reproduction"][str(ch)] = dict(quality=int(f.quality.sum()),
            peak_right=len(selected),retained_bins=len(rebuilt),max_profile_mean_difference_ps=maxdiff,
            coefficients_ps=coeff.tolist(),saved_coefficients_ps=expected.tolist())
        if report["input_files"]==60:
            assert len(rebuilt)==len(p)
            assert maxdiff<.05, (ch,maxdiff)
            assert np.array_equal(rebuilt.count_after_clip,p.count_after_clip)
        f["corrected"] = f.delta_t_ref_ps-curve(ch,f.energy.to_numpy())
        f["raw_trigger"] = (f.time-f.ch192_time)-(f.mcp_peak_time-f.mcp_trigger_time)-curve(ch,f.energy.to_numpy())
        parts[ch] = f.set_index("event_id")
    left,right = parts[137].align(parts[150],join="inner",axis=0)
    assert np.array_equal(left.segment,right.segment)
    frame = pd.DataFrame(dict(event_id=left.index,file_id=left.file_id.to_numpy(),segment=left.segment.to_numpy(),
        entry=left.entry.to_numpy(),Q137=left.energy.to_numpy(),Q150=right.energy.to_numpy(),
        mcp_amp=left.mcp_peak_amp.to_numpy(),mcp_dt=(left.mcp_peak_time-left.mcp_trigger_time).to_numpy(),
        quality=(left.quality&right.quality).to_numpy(),peak_right=(left.qcut&right.qcut).to_numpy(),
        support=(left.support&right.support).to_numpy(),
        alignment_inlier=left.ch192_alignment_inlier.to_numpy(),
        alignment_status=left.reference_alignment_status.to_numpy(),
        D_before=(left.delta_t_ref_ps-right.delta_t_ref_ps).to_numpy(),
        R_before=((left.delta_t_ref_ps+right.delta_t_ref_ps)/2).to_numpy(),
        D_corrected=(left.corrected-right.corrected).to_numpy(),
        R_corrected=((left.corrected+right.corrected)/2).to_numpy(),
        R_raw_trigger=((left.raw_trigger+right.raw_trigger)/2).to_numpy()))
    frame["log_Q_ratio"] = np.log(frame.Q137/frame.Q150)
    frame["Q_geomean"] = np.sqrt(frame.Q137*frame.Q150)
    report["selection_counts"] = dict(matched=len(frame),both_quality=int(frame.quality.sum()),
        both_peak_right=int((frame.quality&frame.peak_right).sum()),
        both_measured_support=int((frame.quality&frame.peak_right&frame.support).sum()))
    selected = frame.loc[frame.quality&frame.peak_right&frame.support].copy().reset_index(drop=True)
    selected["group"] = selected.file_id+":"+selected.segment.astype(str)
    report["alignment_status_counts"] = selected.alignment_status.value_counts().to_dict()
    report["widths"] = {name:width(selected[name]) for name in
        ("D_before","D_corrected","R_before","R_corrected","R_raw_trigger")}
    print("Baseline",report["selection_counts"],report["widths"],flush=True)
    report["window_sensitivity"] = {name:{str(k):width(selected[name],k) for k in (2.,2.5,3.,4.)}
        for name in ("D_corrected","R_corrected")}
    report["bootstrap"] = bootstrap(selected,nboot)
    write_json("baseline_report.json", report)

    # Charge-defined alternatives do not select on either evaluated timing difference.
    selections = {"baseline":np.ones(len(selected),bool),
        "central_80pct_charge":selected.Q137.between(*selected.Q137.quantile([.1,.9])) & selected.Q150.between(*selected.Q150.quantile([.1,.9])),
        "central_50pct_charge_asymmetry":selected.log_Q_ratio.between(*selected.log_Q_ratio.quantile([.25,.75])),
        "alignment_inliers":selected.alignment_inlier.astype(bool)}
    report["selection_sensitivity"] = {name:dict(n=int(np.sum(mask)),D=width(selected.loc[mask,"D_corrected"]),R=width(selected.loc[mask,"R_corrected"]))
        for name,mask in selections.items()}
    group_rows=[]
    for (file_id,seg),g in selected.groupby(["file_id","segment"]):
        if len(g)<30:
            continue
        d,r = width(g.D_corrected),width(g.R_corrected)
        group_rows.append(dict(file_id=file_id,segment=int(seg),n=len(g),
            D_median_ps=float(g.D_corrected.median()),R_median_ps=float(g.R_corrected.median()),
            D_sigma_ps=d["sigma_ps"],R_sigma_ps=r["sigma_ps"],
            D_status=d["status"],R_status=r["status"]))
    groups = pd.DataFrame(group_rows)
    groups.to_csv(OUT/"segment_stability.csv",index=False)
    report["segment_centered_diagnostic"] = {}
    for name in ("D_corrected","R_corrected"):
        centered = selected[name]-selected.groupby("group")[name].transform("median")
        report["segment_centered_diagnostic"][name] = width(centered)

    # Hold out a segment ordinal across all files for walk-fit validation.
    selected["D_crossfit_walk"] = np.nan
    selected["R_crossfit_walk"] = np.nan
    folds=[]
    for seg in sorted(data.segment.unique()):
        predictions={}
        fold=dict(segment=int(seg),channels={})
        for ch in CHANNELS:
            f=parts[ch]
            train=f.loc[f.segment.ne(seg)&f.quality&f.qcut]
            edges=np.linspace(1/train.energy.max(),1/train.energy.min(),21)
            p=build_comparison_profile(train,inv_q_edges=edges,min_entries=5)
            fit=fit_reference_quadratic(p)
            test=selected.loc[selected.segment.eq(seg)]
            q=test[f"Q{ch}"].to_numpy()
            ok=(1/q>=fit.x_min)&(1/q<=fit.x_max)
            in_bins=np.zeros(len(q),bool)
            for row in p.itertuples():
                in_bins|=(1/q>=row.inv_q_bin_low)&(1/q<=row.inv_q_bin_high)
            vals=np.full(len(q),np.nan)
            vals[ok&in_bins]=fit.evaluate(1/q[ok&in_bins])
            original=parts[ch].loc[test.event_id,"delta_t_ref_ps"].to_numpy()
            predictions[ch]=original-vals
            fold["channels"][str(ch)]=dict(training_hits=len(train),bins=len(p),coefficients_ps=fit.coefficients.tolist())
        mask=selected.segment.eq(seg)
        selected.loc[mask,"D_crossfit_walk"]=predictions[137]-predictions[150]
        selected.loc[mask,"R_crossfit_walk"]=(predictions[137]+predictions[150])/2
        folds.append(fold)
    report["walk_crossfit_folds"]=folds
    common=np.isfinite(selected.D_crossfit_walk)&np.isfinite(selected.R_crossfit_walk)
    report["walk_crossfit"] = dict(n=int(common.sum()),
        D_frozen=width(selected.loc[common,"D_corrected"]),R_frozen=width(selected.loc[common,"R_corrected"]),
        D_heldout=width(selected.loc[common,"D_crossfit_walk"]),R_heldout=width(selected.loc[common,"R_crossfit_walk"]))
    print("Held-out walk",report["walk_crossfit"],flush=True)
    # A separate check holds out trigger events from their segment's clock reconstruction.
    data["crossfit_clock_dt"] = crossfit_clock(data)
    l=data.loc[data.channelID.eq(137)].set_index("event_id").loc[selected.event_id]
    r=data.loc[data.channelID.eq(150)].set_index("event_id").loc[selected.event_id]
    selected["R_crossfit_clock"] = (l.crossfit_clock_dt.to_numpy()+r.crossfit_clock_dt.to_numpy()
        -curve(137,selected.Q137.to_numpy())-curve(150,selected.Q150.to_numpy()))/2
    common=np.isfinite(selected.R_crossfit_clock)
    report["clock_crossfit"] = dict(n=int(common.sum()),
        frozen=width(selected.loc[common,"R_corrected"]),heldout=width(selected.loc[common,"R_crossfit_clock"]))
    selected.to_csv(OUT/"selected_events.csv.gz",index=False)
    frame.to_csv(OUT/"all_events_selection.csv.gz",index=False)
    # The covariance identity is descriptive, not an intrinsic-resolution extraction.
    a=(selected.R_corrected+selected.D_corrected/2).to_numpy()
    b=(selected.R_corrected-selected.D_corrected/2).to_numpy()
    cov=float(np.cov(a,b)[0,1])
    assert np.isclose(np.var(selected.R_corrected,ddof=1)-np.var(selected.D_corrected,ddof=1)/4,cov)
    report["observed_reference_residual_covariance_ps2"]=cov
    write_json("validation_report.json",report)
    plot(selected,groups,report)


def stamp(fig,title):
    fig.suptitle(title,fontsize=15,y=.94)
    fig.text(.08,.98,r"$\bf{CMS}$ Preliminary Phase 2",fontsize=13,va="top")


def profile_plot(ax,x,y,xlabel,ylabel):
    x,y=np.asarray(x),np.asarray(y)
    edges=np.quantile(x,np.linspace(0,1,11))
    rows=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(x>=lo)&((x<hi) if i<9 else (x<=hi))
        vals=y[mask]
        rows.append(dict(x=float(np.mean(x[mask])),median=float(np.median(vals)),
                         q16=float(np.percentile(vals,16)),q84=float(np.percentile(vals,84)),n=int(mask.sum())))
    p=pd.DataFrame(rows)
    ax.scatter(x,y,s=2,alpha=.08,color=".3",rasterized=True)
    ax.errorbar(p.x,p['median'],yerr=[p['median']-p.q16,p.q84-p['median']],fmt="o-",color="#147da0",capsize=3)
    med,scale=float(np.median(y)),robust_sigma(y)
    lo,hi=med-4*scale,med+4*scale
    ax.set(xlabel=xlabel,ylabel=ylabel,ylim=(lo,hi))
    ax.text(.02,.97,f"Outside view: {np.mean((y<lo)|(y>hi)):.1%}",transform=ax.transAxes,
            va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    ax.grid(alpha=.2)
    return rows


def plot(selected,groups,report):
    plt.rcParams.update({"font.size":10,"pdf.fonttype":42})
    with PdfPages(OUT/"validation.pdf") as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,5.8))
        fig.subplots_adjust(top=.83,bottom=.21,wspace=.25)
        stamp(fig,f"Run 4399 | Bar 8 | {len(selected):,} common selected events")
        for ax,obs,xlabel in zip(axes,("D","R"),(r"$t_{137}-t_{150}$ [ps]",r"$(t_{137}+t_{150})/2-t_{MCP}^{mapped}$ [ps]")):
            corrected=selected[f"{obs}_corrected"].to_numpy()
            center=float(np.median(corrected))
            for suffix,color,label in (("before",".5","Before walk correction"),("corrected","#147da0","After channel-wise correction")):
                values=selected[f"{obs}_{suffix}"].to_numpy()
                values=values-np.median(values)
                ax.hist(values,bins=np.linspace(-400,400,81),histtype="step",color=color,label=label)
            w=report["widths"][f"{obs}_corrected"]
            lo,hi=w['fit_low_ps']-center,w['fit_high_ps']-center
            if w['status']=="ok":
                mu=w['fit_mean_ps']-center; sigma=w['sigma_ps']
                xx=np.linspace(lo,hi,250)
                norm=ndtr((hi-mu)/sigma)-ndtr((lo-mu)/sigma)
                yy=w['count']*10/(sigma*np.sqrt(2*np.pi)*norm)*np.exp(-.5*((xx-mu)/sigma)**2)
                ax.plot(xx,yy,color="#c04b27",label="Truncated-normal core fit")
            interval=report['bootstrap']['segment_block'][obs]
            ax.set(xlabel=xlabel+" (median centered)",ylabel="Events / 10 ps",xlim=(-400,400))
            ax.set_title(f"Core sigma = {w['sigma_ps']:.2f} ps; block 68% [{interval['p16']:.2f}, {interval['p84']:.2f}]")
            ax.legend(fontsize=8,loc="upper right")
            ax.grid(alpha=.2)
            ax.text(.03,.78,f"Full-sample RMS: {w['rms_about_mean_ps']:.3g} ps\nOutside fit window: {w['outside_fit_fraction']:.1%}",transform=ax.transAxes,fontsize=9,
                    bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
        fig.text(.08,.07,"No timing-tail selection. Each distribution is centered only for display; fit windows are median +/- 2.5 scaled MAD.\nBootstrap intervals are conditional on frozen clock and walk calibrations. Display range does not restrict the reported RMS.",fontsize=10)
        pdf.savefig(fig); fig.savefig(OUT/"timing_distributions.png",dpi=160); plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(12,8))
        fig.subplots_adjust(top=.85,bottom=.14,hspace=.4,wspace=.25)
        stamp(fig,"Charge and charge-asymmetry dependence")
        profiles={}
        for ax,x,y in zip(axes.flat,("Q137","Q150","log_Q_ratio","log_Q_ratio"),("R_corrected","R_corrected","D_corrected","R_corrected")):
            profiles[x+"_"+y]=profile_plot(ax,selected[x],selected[y],x+ (" [ADC]" if x.startswith("Q") else ""),y+" [ps]")
        fig.text(.08,.04,"All events enter the summaries; view is median +/- 4 scaled MAD (outside-view fractions shown). Ten equal-occupancy bins\nshow median and central 68% event spread, not SEM. Charge asymmetry is a proxy, not independent tracking.",fontsize=10)
        write_json("charge_profiles.json",profiles)
        pdf.savefig(fig);fig.savefig(OUT/"charge_dependence.png",dpi=150);plt.close(fig)

        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        fig.subplots_adjust(top=.85,bottom=.15,hspace=.25)
        stamp(fig,"Stability across file / scope-segment groups")
        for obs,color in (("D","#147da0"),("R","#c04b27")):
            axes[0].plot(np.arange(len(groups)),groups[f"{obs}_median_ps"],".",label=obs,color=color)
            axes[1].plot(np.arange(len(groups)),groups[f"{obs}_sigma_ps"],".",label=obs,color=color)
        axes[0].set_ylabel("Group median [ps]");axes[1].set(ylabel="Group core sigma [ps]",xlabel="File / segment group (at least 30 selected events)")
        for ax in axes: ax.set_yscale("symlog",linthresh=200)
        for ax in axes: ax.legend();ax.grid(alpha=.2)
        fig.text(.08,.04,"D: end difference. R: bar-average minus mapped MCP. No per-group recentering is applied to the headline distributions.\nGroup widths are diagnostic point estimates; exact identifiers and counts are saved in segment_stability.csv.",fontsize=10)
        pdf.savefig(fig);fig.savefig(OUT/"segment_stability.png",dpi=150);plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,5.8))
        fig.subplots_adjust(top=.83,bottom=.2,wspace=.27)
        stamp(fig,"Fit-window sensitivity and held-out calibration checks")
        for obs,color in (("D","#147da0"),("R","#c04b27")):
            rows=report['window_sensitivity'][obs+'_corrected']
            axes[0].plot([float(k) for k in rows],[v['sigma_ps'] for v in rows.values()],"o-",label=obs,color=color)
        axes[0].set(xlabel="Core half-window / scaled MAD",ylabel="Fitted Gaussian sigma [ps]");axes[0].legend();axes[0].grid(alpha=.2)
        wc,cc=report['walk_crossfit'],report['clock_crossfit']
        labels=['D: walk','R: walk','R: clock']
        frozen=[wc['D_frozen']['sigma_ps'],wc['R_frozen']['sigma_ps'],cc['frozen']['sigma_ps']]
        held=[wc['D_heldout']['sigma_ps'],wc['R_heldout']['sigma_ps'],cc['heldout']['sigma_ps']]
        axes[1].plot(np.arange(3)-.06,frozen,'o',label="Frozen full-data calibration")
        axes[1].plot(np.arange(3)+.06,held,'s',label="Held-out calibration")
        axes[1].set(xticks=np.arange(3),xticklabels=labels,ylabel="Gaussian core sigma [ps]")
        axes[1].legend(fontsize=9);axes[1].grid(alpha=.2)
        fig.text(.08,.05,"Walk: hold out each scope-segment ordinal across files, keeping the original clock mapping.\nClock: alternate trigger events within each segment; no extrapolation. Width pairs use identical eligible events.\nThese separate diagnostics are not a fully independent joint clock-and-walk validation.",fontsize=10)
        pdf.savefig(fig);fig.savefig(OUT/"validation_checks.png",dpi=150);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("extract","analyze"))
    p.add_argument("--limit",type=int)
    p.add_argument("--bootstrap",type=int,default=400)
    args=p.parse_args()
    if args.stage=="extract": extract(args.limit)
    else: analyze(args.bootstrap)
