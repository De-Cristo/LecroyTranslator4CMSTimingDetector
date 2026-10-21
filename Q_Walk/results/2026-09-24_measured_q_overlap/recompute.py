"""Recompute shape closure on measured profile support, with a free offset."""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_extended_awc_closure import CHANNELS, _paths, _module, _bar, _walk, _plot_awc_summary


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def align(x, y):
    """Uniform-Q least squares for y = AWC*x + offset."""
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean()
    denominator = np.sum(xc * xc, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError("Flat inverse slope")
    awc = np.sum(xc * yc, axis=-1) / denominator
    offset = y.mean() - awc * x.mean(axis=-1)
    return awc, offset


def process(channel):
    source = _paths(channel, ROOT / "Q_Walk/results/run4076_extended_profiles_v1",
                    ROOT / "Q_Walk/reference_quadratic_t0_extended_full_q_v2")
    snap = OUT / "inputs" / f"channel_{channel}"
    snap.mkdir(parents=True, exist_ok=True)
    # Reuse the saved input snapshot on subsequent runs.
    for path in source:
        if not (snap / path.name).exists():
            shutil.copy2(path, snap / path.name)
    summary = json.loads((snap / source[0].name).read_text())
    reference = json.loads((snap / source[2].name).read_text())
    ref_csv = ROOT / reference["profile_csv"]
    if not (snap / ref_csv.name).exists():
        shutil.copy2(ref_csv, snap / ref_csv.name)
    profile = summary["profile"]["points"]
    save_csv(snap / "run4076_profile.csv", profile)
    ref = np.genfromtxt(snap / ref_csv.name, delimiter=",", names=True)
    ref = ref[(ref["count_after_clip"] > 0) & np.isfinite(ref["inv_q_mean"])]
    q4076 = np.array([p["q_mean_adc"] for p in profile])
    q4399 = 1 / ref["inv_q_mean"]
    lo, hi = max(q4076.min(), q4399.min()), min(q4076.max(), q4399.max())
    if hi <= lo:
        raise ValueError("No measured profile overlap")
    grid = np.linspace(lo, hi, 3000)
    mask4076 = np.zeros(grid.size, dtype=bool)
    mask4399 = mask4076.copy()
    for p in profile:
        mask4076 |= (grid >= p["q_low_adc"]) & (grid <= p["q_high_adc"])
    for p in ref:
        mask4399 |= (grid >= 1 / p["inv_q_bin_high"]) & (grid <= 1 / p["inv_q_bin_low"])
    mask = mask4076 & mask4399
    q = grid[mask]
    if q.size < 3:
        raise ValueError("Insufficient jointly populated support")
    fit = summary["linear_profile_fit"]
    coef = np.array([fit["intercept_ua_per_ns"], fit["coefficient_ua_per_ns_per_adc"]])
    with np.load(snap / source[1].name) as archive:
        boot = archive["coefficients"]
    s = coef[0] + coef[1] * q
    if np.any(s <= 0):
        raise ValueError("Non-positive central slope")
    y = _walk(q, [reference["coefficients_ns"][k] for k in ("a", "b", "c")])
    awc, offset = align(1 / s, y)
    bs = boot[:, [0]] + boot[:, [1]] * q
    valid = np.all(np.isfinite(bs) & (bs > 0), axis=1)
    ba, bo = align(1 / bs[valid], y)
    if not ba.size:
        raise ValueError("No valid bootstrap replicas")
    prediction = awc / s + offset
    bands = np.percentile(ba[:, None] / bs[valid] + bo[:, None], [16, 84], axis=0)
    percentiles = np.percentile(ba, [16, 50, 84])
    residual = (prediction - y) * 1000
    assert abs(np.mean(residual)) < 1e-7
    assert q.min() >= lo and q.max() <= hi
    row = dict(channel=channel, module=_module(channel), bar=_bar(channel),
               q4076_min=float(q4076.min()), q4076_max=float(q4076.max()),
               q4399_min=float(q4399.min()), q4399_max=float(q4399.max()),
               overlap_min=float(q.min()), overlap_max=float(q.max()),
               grid_points=int(q.size), excluded_gap_grid_points=int((~mask).sum()),
               awc_ua=float(awc), offset_ns=float(offset), awc_p16=float(percentiles[0]),
               awc_p50=float(percentiles[1]), awc_p84=float(percentiles[2]),
               rms_ps=float(np.sqrt(np.mean(residual**2))),
               c1=float(coef[1]), bootstrap_valid=int(valid.sum()),
               profile4076_points_in_overlap=int(((q4076 >= lo) & (q4076 <= hi)).sum()),
               profile4399_points_in_overlap=int(((q4399 >= lo) & (q4399 <= hi)).sum()))
    (OUT / f"channel_{channel}.json").write_text(json.dumps(row, indent=2) + "\n")
    np.savez_compressed(OUT / f"channel_{channel}_bootstrap.npz", awc=ba, offset_ns=bo)
    save_csv(OUT / f"channel_{channel}_comparison.csv", [dict(q_adc=float(a), reference_ns=float(b),
        mapped_ns=float(c), mapped_p16_ns=float(d), mapped_p84_ns=float(e))
        for a,b,c,d,e in zip(q,y,prediction,*bands)])
    fig, (ax, support) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    baseline = y.mean()
    # The fit uses only joint support; display each model over its own range.
    def padded(values):
        result = np.full(grid.size, np.nan)
        result[mask] = values
        return result
    full4076 = np.linspace(q4076.min(), q4076.max(), 3000)
    full4399 = np.linspace(q4399.min(), q4399.max(), 3000)
    populated4076 = np.zeros(full4076.size, dtype=bool)
    populated4399 = np.zeros(full4399.size, dtype=bool)
    for p in profile:
        populated4076 |= (full4076 >= p["q_low_adc"]) & (full4076 <= p["q_high_adc"])
    for p in ref:
        populated4399 |= (full4399 >= 1 / p["inv_q_bin_high"]) & (full4399 <= 1 / p["inv_q_bin_low"])
    full_slope = coef[0] + coef[1] * full4076
    mapped = np.full(full4076.size, np.nan)
    positive = full_slope > 0
    mapped[positive] = awc / full_slope[positive] + offset - baseline
    reference_curve = _walk(full4399, [reference["coefficients_ns"][k] for k in ("a", "b", "c")]) - baseline
    for xx, yy, populated, color, label in (
        (full4399, reference_curve, populated4399, "black", "Run4399 t0 quadratic"),
        (full4076, mapped, populated4076, "#c43b39", "Run4076 AWC / S(Q) + offset"),
    ):
        ax.plot(xx, yy, color=color, linestyle="--", linewidth=1.4)
        ax.plot(xx, np.where(populated, yy, np.nan), color=color, label=label)
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    for index, (start, end) in enumerate(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))):
        ax.axvspan(grid[start], grid[end-1], color="green", alpha=.10,
                   label="Matching region" if index == 0 else None)
    ax.plot([], [], color="0.4", linestyle="--", label="Gap in retained profile bins")
    ax.fill_between(grid, padded(bands[0]-baseline), padded(bands[1]-baseline),
                    color="#c43b39", alpha=.22, label="68% bootstrap interval")
    ax.set(xlabel="Q [ADC]", ylabel="Walk relative to overlap mean [ns]",
           title=f"Channel {channel}: measured-Q overlap")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(.03, .04, f"AWC = {awc:.3f} uA [{percentiles[0]:.3f}, {percentiles[2]:.3f}]\n"
            f"Q = {q.min():.1f} to {q.max():.1f} ADC; RMS = {row['rms_ps']:.2f} ps",
            transform=ax.transAxes, fontsize=9)
    support.scatter(q4076, np.ones(q4076.size), color="#1f77b4", marker="|", s=160)
    support.scatter(q4399, np.zeros(q4399.size), color="#c43b39", marker="|", s=160)
    support.axvspan(lo, hi, color="green", alpha=.12)
    support.set(yticks=[0, 1], yticklabels=["4399 profile", "4076 profile"],
                ylim=(-.5, 1.5), xlabel="Measured profile Q coordinates [ADC]")
    ax.set_xlim(support.get_xlim())
    for axis in (ax, support):
        axis.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"channel_{channel}_overlap_comparison_full_range.{suffix}", dpi=180)
    plt.close(fig)
    return row, dict(channel=channel, module=_module(channel), coefficients=coef,
                     awc=float(awc), bootstrap_awc=ba)


def main():
    rows, items, failures = [], [], []
    for channel in CHANNELS:
        try:
            row, item = process(channel)
            rows.append(row)
            items.append(item)
            print(f"Ch{channel}: Q={row['overlap_min']:.1f}..{row['overlap_max']:.1f}, AWC={row['awc_ua']:.3f}", flush=True)
        except ValueError as exc:
            failures.append(dict(channel=channel, error=str(exc)))
    save_csv(OUT / "channels.csv", rows)
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    for stem, excluded in [("awc_summary_linear", ()), ("awc_summary_filtered_linear", (130,139,240,242,244))]:
        _plot_awc_summary(items, OUT, stem, "AWC from measured-Q overlap", excluded_channels=excluded, log_x=False)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (OUT / "inputs").rglob("*") if p.is_file()}
    (OUT / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Saved {len(rows)} channels; {len(failures)} failures", flush=True)


if __name__ == "__main__":
    main()
