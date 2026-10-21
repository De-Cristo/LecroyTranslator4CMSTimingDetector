"""Compare saved MCP references; no ROOT processing or production-fit changes."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = []


def source(relative):
    snapshot = OUT / "inputs" / relative
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, snapshot)
    SOURCES.append({"original": str(relative), "snapshot": str(snapshot.relative_to(OUT)),
                    "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    return snapshot


def table(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def profile(x, y, bins):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    ids = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    n = np.bincount(ids, minlength=bins)
    nonempty = n > 0
    xm = np.zeros(bins)
    ym = np.zeros(bins)
    xm[nonempty] = np.bincount(ids, weights=x, minlength=bins)[nonempty] / n[nonempty]
    ym[nonempty] = np.bincount(ids, weights=y, minlength=bins)[nonempty] / n[nonempty]
    return edges, ids, n, xm, ym


def rms_shape(a, b):
    residual = (a - b - np.mean(a - b)) * 1000
    return float(np.sqrt(np.mean(residual**2)))


def process(ch, bar, pdf):
    oldstem = f"ch192_vs_trig_lowess_bar{bar}_ch{ch}_walk_before_walk_fit_expected_refit"
    old = table(source(Path("Q_Walk/reference_quadratic_t0") / f"{oldstem}_profile.csv"))
    coeffpath = source(Path("Q_Walk/reference_quadratic_t0") / f"{oldstem}_coeffs.csv")
    with coeffpath.open() as stream:
        row = next(r for r in csv.DictReader(stream) if r["fit_kind"] == "binned")
    op = np.array([float(row[k]) for k in ("p2", "p1", "p0")])
    newstem = f"run4399_upstream_bar{bar}_ch{ch}_t0_quadratic"
    folder = Path("Q_Walk/results/2026-09-24_measured_q_overlap/inputs") / f"channel_{ch}"
    meta = json.loads(source(folder / f"{newstem}.json").read_text())
    new = table(source(folder / f"{newstem}_profile.csv"))
    npoly = np.array([meta["coefficients_ns"][k] for k in ("c", "b", "a")])
    old_ols = np.polyfit(old["invE_mean"], old["delta_t_mean"], 2)
    new_wls = np.polyfit(new["inv_q_mean"], new["delta_t_ref_mean_ps"] / 1000, 2,
                         w=1 / new["delta_t_ref_sem_ps"])
    assert np.allclose(old_ols, op, rtol=1e-8, atol=1e-8)
    assert np.allclose(new_wls, npoly, rtol=1e-8, atol=1e-8)

    raw = table(source(Path("archieved") /
        f"ch192_vs_trig_lowess_bar{bar}_ch{ch}_walk_before_walk_fit_data.csv"))
    x, y = raw["invE"], raw["delta_t"] / 1000 + 26
    matches = []
    for nb in range(len(old), 201):
        edges, ids, count, xm, ym = profile(x, y, nb)
        kept = count >= old["count"].min()
        if kept.sum() != len(old) or not np.array_equal(count[kept], old["count"]):
            continue
        if (np.allclose(xm[kept], old["invE_mean"], rtol=1e-10, atol=1e-14)
                and np.allclose(ym[kept], old["delta_t_mean"], rtol=1e-10, atol=1e-12)):
            matches.append((nb, edges, ids, count, kept, ym))
    assert len(matches) == 1, (ch, "ambiguous or absent archived-profile reconstruction")
    nb, edges, ids, count, kept, ym = matches[0]
    kept_events = kept[ids]
    _, _, n20, x20, y20 = profile(x[kept_events], y[kept_events], 20)
    rebin = np.polyfit(x20[n20 > 0], y20[n20 > 0], 2)
    assert n20.sum() == old["count"].sum()
    old_wls = np.polyfit(old["invE_mean"], old["delta_t_mean"], 2,
                         w=1 / old["delta_t_sem"])

    oldq, newq = 1 / old["invE_mean"], 1 / new["inv_q_mean"]
    lo, hi = max(oldq.min(), newq.min()), min(oldq.max(), newq.max())
    q = np.linspace(lo, hi, 20001)
    old_support, new_support = np.zeros(q.size, bool), np.zeros(q.size, bool)
    for i in np.flatnonzero(kept):
        old_support |= (1/q >= edges[i]) & (1/q <= edges[i+1])
    for p in new:
        new_support |= (1/q >= p["inv_q_bin_low"]) & (1/q <= p["inv_q_bin_high"])
    joint = old_support & new_support
    assert joint.any()
    oldcurve = np.polyval(op, 1/q)
    newcurve = np.polyval(npoly, 1/q) + 26
    offset = float(np.mean(newcurve[joint] - oldcurve[joint]))
    residual = 1000 * (newcurve - offset - oldcurve)
    r = residual[joint]
    baseline = float(oldcurve[joint].mean())
    result = dict(channel=ch, old_requested_bins_reconstructed=nb,
                  old_retained_bins=len(old), new_requested_bins=20,
                  new_retained_bins=len(new), overlap_q_min_adc=float(lo), overlap_q_max_adc=float(hi),
                  supported_grid_points=int(joint.sum()), excluded_grid_points=int((~joint).sum()),
                  extra_offset_removed_after_26ns_ps=offset*1000,
                  shape_rms_ps=float(np.sqrt(np.mean(r*r))), max_abs_shape_ps=float(np.max(abs(r))),
                  peak_to_peak_shape_ps=float(np.ptp(r)),
                  old_walk_span_ps=float(np.ptp(oldcurve[joint])*1000),
                  new_walk_span_ps=float(np.ptp(newcurve[joint])*1000),
                  old_profile_event_count=int(old["count"].sum()),
                  new_profile_event_count=int(new["count_after_clip"].sum()),
                  legacy_mean_reconstruction_max_error_ps=float(np.max(abs(ym[kept]-old["delta_t_mean"]))*1000),
                  same_legacy_events_rebin20_rms_ps=rms_shape(np.polyval(rebin,1/q[joint]),oldcurve[joint]),
                  same_old_profile_weighting_only_rms_ps=rms_shape(np.polyval(old_wls,1/q[joint]),oldcurve[joint]))
    save_csv(OUT / f"channel_{ch}_curves.csv", [dict(q_adc=float(qq),joint_support=bool(ok),
        old_ns=float(aa),new_plus26_ns=float(bb),new_aligned_ns=float(bb-offset),
        aligned_new_minus_old_ps=float(rr)) for qq,ok,aa,bb,rr in zip(q,joint,oldcurve,newcurve,residual)])
    (OUT / f"channel_{ch}_details.json").write_text(json.dumps({**result,
        "old_coefficients_c_b_a_ns":op.tolist(), "new_coefficients_c_b_a_ns":npoly.tolist(),
        "same_event_rebin20_coefficients_c_b_a_ns":rebin.tolist(),
        "old_profile_wls_coefficients_c_b_a_ns":old_wls.tolist()},indent=2)+"\n")

    fig, (ax, diff) = plt.subplots(2, 1, figsize=(9, 7.3), sharex=True,
                                  gridspec_kw={"height_ratios":[3,1.2]})
    fig.subplots_adjust(left=.13,right=.97,top=.87,bottom=.12,hspace=.10)
    fig.text(.13,.965,r"$\bf{CMS}$ Preliminary Phase 2",fontsize=15)
    fig.text(.13,.927,f"Run 4399 | Channel {ch} | Reference-version comparison",fontsize=12)
    ax.plot(q,np.where(joint,oldcurve-baseline,np.nan),color="#1874a3",lw=2,label="Note1 archived quadratic")
    ax.plot(q,np.where(joint,newcurve-offset-baseline,np.nan),color="#c04b27",lw=2,ls="--",label="Note2 updated quadratic, offset aligned")
    for qq,yy,ee,color,marker,label in (
        (oldq,old["delta_t_mean"]-baseline,old["delta_t_sem"],"#1874a3","o","Old means +/- SEM"),
        (newq,new["delta_t_ref_mean_ps"]/1000+26-offset-baseline,new["delta_t_ref_sem_ps"]/1000,"#c04b27","s","New means +/- SEM")):
        visible=(qq>=lo)&(qq<=hi)
        ax.errorbar(qq[visible],yy[visible],yerr=ee[visible],fmt=marker,ms=4,capsize=2,color=color,alpha=.8,label=label)
    ax.set_ylabel("Walk relative to common mean [ns]")
    ax.legend(fontsize=9,loc="upper right")
    diff.axhline(0,color=".4",lw=1)
    diff.plot(q,np.where(joint,residual,np.nan),color="#c04b27",lw=2)
    diff.set(xlabel="Q [ADC]",ylabel="New - old\n[ps]")
    for axis in (ax,diff): axis.grid(alpha=.2)
    fig.text(.13,.03,f"Shape RMS {result['shape_rms_ps']:.2f} ps; max |difference| {result['max_abs_shape_ps']:.2f} ps.  Offset only; no scale fit.",fontsize=10)
    pdf.savefig(fig)
    fig.savefig(OUT / f"channel_{ch}_comparison.png",dpi=160)
    plt.close(fig)
    return result


def main():
    plt.rcParams.update({"font.size":11,"pdf.fonttype":42})
    for path in ("TimeCalibration/walk_fit_from_csv.py",
                 "TimeCalibration/ch192_vs_trigger_lowess.py",
                 "Q_Walk/qwalk_extended_t0_reference.py",
                 "Q_Walk/qwalk_downstream_t0_reference.py",
                 "Q_Walk/qwalk_t0_data.py", "Q_Walk/qwalk_t0_models.py"):
        source(Path(path))
    with PdfPages(OUT / "reference_comparison.pdf") as pdf:
        rows = [process(ch,bar,pdf) for ch,bar in ((133,7),(154,7),(137,8),(150,8))]
    save_csv(OUT / "summary.csv", rows)
    (OUT / "input_manifest.json").write_text(json.dumps(SOURCES,indent=2)+"\n")
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
