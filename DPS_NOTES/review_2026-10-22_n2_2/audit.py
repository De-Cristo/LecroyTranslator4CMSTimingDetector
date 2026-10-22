"""Replay the frozen run-4076 selections and profiles for review N2-2."""
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
from Q_Walk.qwalk_run4076_peak_right_profile import _build_profile, _fit_linear_profile
from Q_Walk.qwalk_run4076_safe_time2_scatter import _histogram_peak_region, _limits
from Q_Walk.qwalk_two_hit_pair_profile import select_pair_with_exactly_two_module_hits

INPUTS = ROOT / "Q_Walk/results/2026-10-22_clock_quality/inputs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = sorted(INPUTS.glob("channel_*/*_profile_linear_summary.json"))
    assert len(paths) == 22
    summaries = [json.loads(p.read_text()) for p in paths]
    input_paths = {s["input"] for s in summaries}
    assert len(input_paths) == 1
    root_path = Path(input_paths.pop())
    hits = read_root_hits(root_path, time2_mode="safe")
    print(f"Read {len(hits)} hits in {hits.entry.nunique()} events", flush=True)
    modules = {"upstream": UPSTREAM_BAR_CHANNELS, "downstream": DOWNSTREAM_BAR_CHANNELS}
    rows, hashes = [], {}
    for path, saved in zip(paths, summaries):
        ch = saved["target_channel"]
        pair = saved["required_exact_pair"]
        module = saved["selection_module"]
        selected_pair = select_pair_with_exactly_two_module_hits(hits, pair, modules[module])
        data = selected_pair.loc[selected_pair.channelID.eq(ch)]
        q = data.energy.to_numpy(dtype=float)
        t1 = data.time.to_numpy(dtype=float)
        t2 = data.time2.to_numpy(dtype=float)
        dt = t2 - t1
        v1, v2, di = decode_thresholds(data.step2.to_numpy())
        slope = calculate_slope(t1, t2, di)
        dc = (data.t2coarse.to_numpy(dtype=np.int64) - data.t1coarse.to_numpy(dtype=np.int64)) % 1024
        # The frozen summaries predate the optional 100 ps minimum-rise cut.
        prompt = np.isfinite(q) & np.isfinite(slope) & (di > 0) & (dt > 0) & (dc == 0)
        peak = _histogram_peak_region(q[prompt], _limits(q[prompt], 0.5, 99.5, 20.0))
        q_cut = peak["half_max_q_low_adc"]
        selected = prompt & (q >= q_cut)
        profile = _build_profile(q[selected], slope[selected], bins=10)
        actual = pd.DataFrame(profile)
        stored = pd.DataFrame(saved["profile"]["points"])
        csv_path = path.parent / "run4076_profile.csv"
        csv = pd.read_csv(csv_path)
        fit = _fit_linear_profile(profile)
        fit_keys = ("intercept_ua_per_ns", "coefficient_ua_per_ns_per_adc")
        checks = {
            "safe_mode": saved["time2_mode"] == "safe",
            "pair_count": len(data) == saved["pair_channel_hits"],
            "prompt_count": int(prompt.sum()) == saved["prompt_channel_hits"],
            "selected_count": int(selected.sum()) == saved["q_selection"]["selected_hits"],
            "histogram_boundary": bool(np.isclose(q_cut, saved["q_selection"]["q_cut_adc"], rtol=1e-12, atol=1e-10)),
            "profile_replay": actual.shape == stored.shape and bool(np.allclose(actual, stored, rtol=1e-10, atol=1e-9, equal_nan=True)),
            "csv_matches_summary": csv.shape == stored.shape and bool(np.allclose(csv, stored, rtol=1e-12, atol=1e-10, equal_nan=True)),
            "linear_coefficients": all(np.isclose(fit[k], saved["linear_profile_fit"][k], rtol=1e-9, atol=1e-9) for k in fit_keys),
            "linear_covariance": bool(np.allclose(fit["covariance"], saved["linear_profile_fit"]["covariance"], rtol=1e-9, atol=1e-12)),
        }
        # Independently solve SEM-weighted least squares without normal equations.
        x = actual.q_mean_adc.to_numpy()
        y = actual.slope_mean_ua_per_ns.to_numpy()
        sem = actual.slope_sem_ua_per_ns.to_numpy()
        valid = np.isfinite(sem) & (sem > np.sqrt(np.finfo(float).eps) * max(1, np.max(abs(y))))
        checks["undefined_sem_only_singletons"] = bool(np.array_equal(np.isnan(sem), actual["count"].to_numpy() == 1))
        checks["fit_bin_count"] = int(valid.sum()) == saved["linear_profile_fit"]["profile_bins_used"]
        design = np.column_stack((np.ones(valid.sum()), x[valid]))
        independent, *_ = np.linalg.lstsq(design / sem[valid, None], y[valid] / sem[valid], rcond=None)
        checks["independent_weighted_fit"] = bool(np.allclose(independent, [fit[k] for k in fit_keys], rtol=1e-9, atol=1e-9))
        boot = saved["linear_profile_bootstrap"]
        boot_path = path.parent / boot["coefficients_file"]
        with np.load(boot_path) as archive:
            coeff = archive["coefficients"]
        checks["bootstrap_replicas"] = coeff.shape == (1000, 2) and bool(np.isfinite(coeff).all())
        checks["bootstrap_percentiles"] = all(np.allclose(np.percentile(coeff[:, i], [16, 50, 84]), boot[k], rtol=1e-12, atol=1e-10) for i, k in enumerate(("intercept_ua_per_ns_percentiles", "coefficient_ua_per_ns_per_adc_percentiles")))
        checks["fixed_bootstrap_edges"] = bool(np.allclose(boot["fixed_q_bin_edges_adc"], np.linspace(q[selected].min(), q[selected].max(), 11)))
        threshold_values = sorted({(float(a), int(b), int(c), float(d)) for a, b, c, d in zip(data.step2[prompt], v1[prompt], v2[prompt], di[prompt])})
        missing = data.missing_calib_t1.to_numpy() | data.missing_calib_t2.to_numpy()
        checks["selected_calibration_coverage"] = not bool((selected & missing).any())
        row = dict(channel=ch, module=module, pair=pair, pair_hits=len(data),
                   prompt_hits=int(prompt.sum()), selected_hits=int(selected.sum()),
                   q_cut_adc=q_cut, threshold_settings=threshold_values,
                   selected_dt_min_ps=float(dt[selected].min()), selected_dt_max_ps=float(dt[selected].max()),
                   selected_missing_fine_calibration=int((selected & missing).sum()),
                   prompt_missing_fine_calibration=int((prompt & missing).sum()),
                   prompt_below_100ps=int((prompt & (dt < 100)).sum()),
                   selected_below_100ps=int((selected & (dt < 100)).sum()),
                   populated_profile_bins=len(profile), fitted_profile_bins=int(valid.sum()),
                   checks={k: bool(v) for k, v in checks.items()})
        rows.append(row)
        for p in (path, csv_path, boot_path):
            hashes[str(p.relative_to(ROOT))] = sha256(p)
        print(f"ch{ch}: selected={row['selected_hits']} missing={row['selected_missing_fine_calibration']} short={row['selected_below_100ps']} failed={[k for k,v in checks.items() if not v]}", flush=True)
    scripts = ("qwalk_core.py", "qwalk_channels.py", "qwalk_two_hit_pair_profile.py",
               "qwalk_run4076_peak_right_profile.py", "qwalk_run4076_safe_time2_scatter.py")
    report = dict(status="passed" if all(all(r["checks"].values()) for r in rows) else "mismatch",
                  input=str(root_path), input_sha256=sha256(root_path),
                  total_hits=len(hits), total_events=int(hits.entry.nunique()), channels=rows,
                  input_hashes=hashes,
                  code_hashes={f"Q_Walk/{s}": sha256(ROOT / "Q_Walk" / s) for s in scripts})
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["status"] == "passed", "Inspect audit.json for mismatches"


if __name__ == "__main__":
    main()
