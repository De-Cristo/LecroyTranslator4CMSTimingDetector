"""Frozen run4076 model/selection sensitivity; never overwrite nominal results."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_channels import DOWNSTREAM_BAR_CHANNELS, UPSTREAM_BAR_CHANNELS
from Q_Walk.qwalk_core import calculate_slope, decode_thresholds, read_root_hits
from Q_Walk.qwalk_run4076_peak_right_profile import _build_profile
from Q_Walk.qwalk_run4076_safe_time2_scatter import _histogram_peak_region, _limits
from Q_Walk.qwalk_two_hit_pair_profile import select_pair_with_exactly_two_module_hits

BASE = ROOT / "Q_Walk/results/2026-10-22_clock_quality"
VARIANTS = (
    ("baseline", 10, 1, 1.0, 0, False),
    ("quadratic", 10, 2, 1.0, 0, False),
    ("bins8", 8, 1, 1.0, 0, False),
    ("bins12", 12, 1, 1.0, 0, False),
    ("qcut_minus5", 10, 1, 0.95, 0, False),
    ("qcut_plus5", 10, 1, 1.05, 0, False),
    ("rise100", 10, 1, 1.0, 100, False),
    ("rise100_repeak", 10, 1, 1.0, 100, True),
)


def fit_profile(q, slope, bins=10, degree=1):
    profile = pd.DataFrame(_build_profile(q, slope, bins=bins))
    center = float((np.min(q) + np.max(q)) / 2)
    scale = float((np.max(q) - np.min(q)) / 2)
    if not scale > 0:
        raise ValueError("No charge span")
    edges = np.linspace(np.min(q), np.max(q), bins + 1)
    assignment = np.clip(np.digitize(q, edges) - 1, 0, bins - 1)
    z = (q - center) / scale
    # Fit bin-averaged moments, not mean(Q)^2, for the quadratic alternative.
    moments = np.array([[np.mean(z[assignment == i] ** k) for k in range(degree + 1)]
                        for i in range(bins) if np.any(assignment == i)])
    sem = profile.slope_sem_ua_per_ns.to_numpy()
    y = profile.slope_mean_ua_per_ns.to_numpy()
    valid = np.isfinite(sem) & (sem > np.sqrt(np.finfo(float).eps) * max(1, np.max(abs(y))))
    if valid.sum() <= degree + 1:
        raise ValueError("Insufficient bins with usable SEM for residual degrees of freedom")
    design = moments[valid] / sem[valid, None]
    coeff, _, rank, _ = np.linalg.lstsq(design, y[valid] / sem[valid], rcond=None)
    if rank != degree + 1:
        raise ValueError("Rank-deficient profile fit")
    prediction = moments @ coeff
    residual = y - prediction
    chi2 = float(np.sum((residual[valid] / sem[valid]) ** 2))
    ndf = int(valid.sum() - degree - 1)
    profile["prediction"] = prediction
    profile["residual"] = residual
    profile["used_in_fit"] = valid
    return dict(degree=degree, center=center, scale=scale, coefficients=coeff.tolist(),
                chi2=chi2, ndf=ndf, chi2_ndf=chi2 / ndf,
                fitted_bins=int(valid.sum()), minimum_fit_bin_count=int(profile.loc[valid, "count"].min()),
                max_abs_residual_ua_per_ns=float(np.max(abs(residual[valid])))), profile


def evaluate(model, q):
    return np.polynomial.polynomial.polyval((np.asarray(q) - model["center"]) / model["scale"], model["coefficients"])


def support_mask(profile, q):
    mask = (q >= profile.q_mean_adc.min()) & (q <= profile.q_mean_adc.max())
    populated = np.zeros(len(q), dtype=bool)
    for p in profile.itertuples():
        populated |= (q >= p.q_low_adc) & (q <= p.q_high_adc)
    return mask & populated


def align_inverse(slope, reference):
    slope = np.asarray(slope, dtype=float)
    if not np.isfinite(slope).all() or np.any(slope <= 0):
        raise ValueError("Non-positive or nonfinite slope on matching support")
    x = 1 / slope
    xc = x - x.mean(axis=-1, keepdims=True)
    denominator = np.sum(xc * xc, axis=-1)
    if np.any(denominator <= np.finfo(float).eps**2 * np.sum(x * x, axis=-1)):
        raise ValueError("Flat inverse slope: AWC is unidentifiable")
    awc = np.sum(xc * (reference - reference.mean()), axis=-1) / denominator
    offset = reference.mean() - awc * x.mean(axis=-1)
    return awc, offset


def minimum_on_support(model, q):
    minimum = float(evaluate(model, q).min())
    if model["degree"] == 2 and model["coefficients"][2] > 0:
        vertex = model["center"] - model["scale"] * model["coefficients"][1] / (2 * model["coefficients"][2])
        # Include an interior quadratic minimum within a retained grid segment.
        splits = np.flatnonzero(np.diff(q) > 1.5 * np.median(np.diff(q))) + 1
        if any(segment[0] <= vertex <= segment[-1] for segment in np.split(q, splits)):
            minimum = min(minimum, float(evaluate(model, vertex)))
    return minimum


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = sorted((BASE / "inputs").glob("channel_*/*_profile_linear_summary.json"))
    summaries = [json.loads(p.read_text()) for p in paths]
    assert len(summaries) == 22 and len({s["input"] for s in summaries}) == 1
    root_path = Path(summaries[0]["input"])
    hits = read_root_hits(root_path, time2_mode="safe")
    print(f"Loaded {len(hits)} hits", flush=True)
    rows, profiles, grids, channel_results, hashes = [], [], [], [], {}
    modules = {"upstream": UPSTREAM_BAR_CHANNELS, "downstream": DOWNSTREAM_BAR_CHANNELS}
    for path, saved in zip(paths, summaries):
        ch = saved["target_channel"]
        data = select_pair_with_exactly_two_module_hits(hits, saved["required_exact_pair"], modules[saved["selection_module"]])
        data = data.loc[data.channelID.eq(ch)]
        q = data.energy.to_numpy(dtype=float)
        dt = data.time2.to_numpy(dtype=float) - data.time.to_numpy(dtype=float)
        _, _, di = decode_thresholds(data.step2.to_numpy())
        slope = calculate_slope(data.time.to_numpy(), data.time2.to_numpy(), di)
        dc = (data.t2coarse.to_numpy(dtype=np.int64) - data.t1coarse.to_numpy(dtype=np.int64)) % 1024
        prompt = np.isfinite(q) & np.isfinite(slope) & (di > 0) & (dt > 0) & (dc == 0)
        models, tables, variants = {}, {}, {}
        for name, bins, degree, factor, minimum, repeak in VARIANTS:
            row = dict(channel=ch, module=saved["selection_module"], variant=name,
                       degree=degree, bins=bins, minimum_rise_ps=minimum, alignment_status="not_evaluated")
            try:
                quality = prompt & (dt >= minimum)
                cut = saved["q_selection"]["q_cut_adc"] * factor
                if repeak:
                    cut = _histogram_peak_region(q[quality], _limits(q[quality], 0.5, 99.5, 20.0))["half_max_q_low_adc"]
                mask = quality & (q >= cut)
                if mask.sum() < 50:
                    raise ValueError("Fewer than 50 selected hits")
                model, table = fit_profile(q[mask], slope[mask], bins, degree)
                row.update(model, q_cut_adc=float(cut), selected_hits=int(mask.sum()), profile_status="ok")
                models[name], tables[name] = model, table
                profiles.append(table.assign(channel=ch, variant=name))
                if name == "baseline":
                    assert mask.sum() == saved["q_selection"]["selected_hits"]
                    np.testing.assert_allclose(table[list(saved["profile"]["points"][0])], pd.DataFrame(saved["profile"]["points"]), rtol=1e-10, atol=1e-9, equal_nan=True)
                    c0 = model["coefficients"][0] - model["coefficients"][1] * model["center"] / model["scale"]
                    c1 = model["coefficients"][1] / model["scale"]
                    np.testing.assert_allclose([c0, c1], [saved["linear_profile_fit"]["intercept_ua_per_ns"], saved["linear_profile_fit"]["coefficient_ua_per_ns_per_adc"]], rtol=1e-9, atol=1e-9)
            except ValueError as exc:
                row.update(profile_status="failed", error=str(exc))
            variants[name] = row
        result = dict(channel=ch, module=saved["selection_module"],
                      baseline_chi2_ndf=models["baseline"]["chi2_ndf"],
                      quadratic_chi2_ndf=models["quadratic"]["chi2_ndf"],
                      delta_chi2=models["baseline"]["chi2"] - models["quadratic"]["chi2"],
                      profile_failures=sum(r["profile_status"] != "ok" for r in variants.values()))
        grid_path = BASE / f"channel_{ch}_comparison.csv"
        if not grid_path.exists():
            result["alignment_status"] = "no_saved_reference_overlap"
            for row in variants.values():
                row["alignment_status"] = "no_saved_reference_overlap"
        else:
            original = pd.read_csv(grid_path)
            original_q = original.q_adc.to_numpy()
            original_y = original.reference_ns.to_numpy()
            nominal_path = BASE / f"channel_{ch}.json"
            nominal = json.loads(nominal_path.read_text())
            a0, b0 = align_inverse(evaluate(models["baseline"], original_q), original_y)
            np.testing.assert_allclose([a0, b0], [nominal["awc_ua"], nominal["offset_ns"]], rtol=1e-8, atol=1e-8)
            common = np.ones(len(original), dtype=bool)
            for table in tables.values():
                common &= support_mask(table, original_q)
            x, y = original_q[common], original_y[common]
            result.update(original_awc_ua=float(a0), original_grid_points=len(original),
                          common_grid_points=len(x), common_grid_fraction=len(x) / len(original),
                          saved_nominal_reproduced=True)
            if len(x) < 3:
                result["alignment_status"] = "insufficient_common_support"
                for row in variants.values():
                    row["alignment_status"] = "insufficient_common_support"
            else:
                result.update(alignment_status="ok", common_q_min=float(x.min()), common_q_max=float(x.max()))
                grids.append(pd.DataFrame(dict(channel=ch, q_adc=x, reference_ns=y)))
                for name, model in models.items():
                    row = variants[name]
                    try:
                        min_s = minimum_on_support(model, x)
                        if min_s <= 0:
                            raise ValueError("Non-positive model on common support")
                        s = evaluate(model, x)
                        a, b = align_inverse(s, y)
                        residual = 1000 * (a / s + b - y)
                        row.update(alignment_status="ok", awc_ua=float(a), offset_ns=float(b),
                                   rms_ps=float(np.sqrt(np.mean(residual**2))), min_slope=min_s,
                                   negative_awc=bool(a <= 0), mean_residual_ps=float(residual.mean()))
                    except ValueError as exc:
                        row.update(alignment_status="failed", alignment_error=str(exc))
                baseline = variants["baseline"]
                boot_path = path.parent / saved["linear_profile_bootstrap"]["coefficients_file"]
                with np.load(boot_path) as archive:
                    coeff = archive["coefficients"]
                bs = coeff[:, [0]] + coeff[:, [1]] * x
                valid = np.all(np.isfinite(bs) & (bs > 0), axis=1)
                ba, _ = align_inverse(bs[valid], y)
                low, mid, high = np.percentile(ba, [16, 50, 84])
                result.update(baseline_awc_ua=baseline["awc_ua"], baseline_p16=float(low), baseline_p50=float(mid),
                              baseline_p84=float(high), baseline_bootstrap_valid=int(valid.sum()),
                              support_only_awc_shift=baseline["awc_ua"] - float(a0))
                for row in variants.values():
                    if row["alignment_status"] == "ok":
                        row["delta_awc_ua"] = row["awc_ua"] - baseline["awc_ua"]
                        row["delta_awc_percent"] = 100 * row["delta_awc_ua"] / abs(baseline["awc_ua"]) if baseline["awc_ua"] != 0 else None
                differences = [abs(r["delta_awc_ua"]) for r in variants.values() if r["alignment_status"] == "ok"]
                result.update(max_abs_awc_shift=max(differences), alignment_failures=sum(r["alignment_status"] != "ok" for r in variants.values()),
                              conditional_half_span=float((high - low) / 2),
                              envelope_over_conditional_half_span=max(differences) / ((high - low) / 2))
                hashes[str(boot_path.relative_to(ROOT))] = sha256(boot_path)
            for p in (grid_path, nominal_path):
                hashes[str(p.relative_to(ROOT))] = sha256(p)
        rows.extend(variants.values())
        channel_results.append(result)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        print(f"ch{ch}: chi2/ndf linear={result['baseline_chi2_ndf']:.2f}, quadratic={result['quadratic_chi2_ndf']:.2f}; {result['alignment_status']}; max shift={result.get('max_abs_awc_shift', 'n/a')}", flush=True)
    pd.concat(profiles, ignore_index=True).to_csv(OUT / "profiles.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(OUT / "common_grid.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "coefficients"} for r in rows]).to_csv(OUT / "variants.csv", index=False)
    pd.DataFrame(channel_results).to_csv(OUT / "channels.csv", index=False)
    (OUT / "models.json").write_text(json.dumps(rows, indent=2) + "\n")
    report = dict(channels=channel_results, variants=[v[0] for v in VARIANTS], input=str(root_path),
                  input_sha256=sha256(root_path), input_hashes=hashes,
                  statement="Sensitivity envelope only, not a confidence interval; no nominal results replaced",
                  code_hashes={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "Q_Walk/qwalk_core.py", ROOT / "Q_Walk/qwalk_channels.py", ROOT / "Q_Walk/qwalk_two_hit_pair_profile.py", ROOT / "Q_Walk/qwalk_run4076_peak_right_profile.py", ROOT / "Q_Walk/qwalk_run4076_safe_time2_scatter.py"]})
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
