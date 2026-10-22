"""Audit four-channel reference compatibility without changing frozen inputs."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_downstream_t0_reference import _quality_mask
from Q_Walk.qwalk_extended_t0_reference import selected_inv_q_edges
from Q_Walk.qwalk_run4076_safe_time2_scatter import _histogram_peak_region, _limits
from Q_Walk.qwalk_t0_models import build_comparison_profile, fit_reference_quadratic

OUT = Path(__file__).resolve().parent
CURRENT = ROOT / "Q_Walk/results/2026-10-22_clock_quality"
CACHE = CURRENT / "cache/upstream_bar8_aligned.csv.gz"


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def compare_shapes(old, new, low, high):
    q = np.linspace(low, high, 1000)
    delta = new.evaluate(1 / q) - old.evaluate(1 / q)
    centered = delta - np.mean(delta)
    return dict(q_min_adc=low, q_max_adc=high,
                mean_offset_ps=float(np.mean(delta)),
                centered_rms_ps=float(np.sqrt(np.mean(centered**2))),
                centered_max_abs_ps=float(np.max(np.abs(centered))),
                weighting="uniform Q grid on common fit support; constant offset removed")


def main():
    data = pd.read_csv(CACHE)
    assert not data.duplicated(["event_id", "channelID"]).any()
    assert data.groupby("event_id").channelID.agg(frozenset).eq(
        frozenset((137, 150, 234, 243))).all()
    events = data.drop_duplicates("event_id")
    shared = ["expected_ch192_time", "mcp_peak_time", "mcp_trigger_time"]
    assert (data.groupby("event_id")[shared].nunique(dropna=False) == 1).all().all()
    report = dict(matched_events=len(events), source_files=int(events.file_id.nunique()),
                  clock_status=events.reference_alignment_status.value_counts().to_dict(),
                  channels={}, scope="Compatibility audit and candidate refits only; no slide or frozen calibration changes.")
    inputs = [CACHE, Path(__file__), ROOT / "Q_Walk/qwalk_t0_models.py",
              ROOT / "Q_Walk/qwalk_t0_data.py", ROOT / "Q_Walk/qwalk_extended_t0_reference.py",
              ROOT / "Q_Walk/qwalk_downstream_t0_reference.py",
              ROOT / "Q_Walk/qwalk_run4076_safe_time2_scatter.py"]
    all_quality, all_peak, all_support = [], [], []
    for ch in (137, 150, 234, 243):
        f = data.loc[data.channelID.eq(ch)].copy()
        prompt = f.loc[_quality_mask(f)].copy()
        peak = _histogram_peak_region(prompt.energy.to_numpy(),
                                     _limits(prompt.energy.to_numpy(), .5, 99.5, 20.0))
        cut = float(peak["half_max_q_low_adc"])
        selected = prompt.loc[prompt.energy.ge(cut)]
        edges = selected_inv_q_edges(selected.energy, 20)
        profile = build_comparison_profile(selected, n_bins=20, min_entries=5, inv_q_edges=edges)
        fit = fit_reference_quadratic(profile)
        coeff = dict(a=float(fit.coefficients[2]/1000), b=float(fit.coefficients[1]/1000),
                     c=float(fit.coefficients[0]/1000))
        result = dict(channel=ch, quality_hits=len(prompt), q_cut_adc=cut,
                      selected_hits=len(selected), retained_bins=len(profile),
                      coefficients_ns=coeff,
                      covariance_ns2_abc=(fit.covariance/1e6)[::-1, ::-1].tolist(),
                      q_min_adc=float(1/fit.x_max), q_max_adc=float(1/fit.x_min),
                      method="20 uniform full-selected-range inverse-Q bins; 5 scaled-MAD bin clipping; SEM-weighted quadratic",
                      clock_quality="October refreshed finite-trigger segmentation and <=1000 ps pre-LOWESS scaled-MAD gate")
        profile.to_csv(OUT / f"channel_{ch}_current_profile.csv", index=False)
        if ch in (137, 150):
            stem = f"run4399_upstream_bar8_ch{ch}_t0_quadratic"
            meta_path = CURRENT / "references" / f"{stem}.json"
            profile_path = CURRENT / "references" / f"{stem}_profile.csv"
            meta = json.loads(meta_path.read_text())
            saved = pd.read_csv(profile_path)
            assert np.array_equal(profile.count_after_clip, saved.count_after_clip)
            diff = float(np.max(np.abs(profile.delta_t_ref_mean_ps - saved.delta_t_ref_mean_ps)))
            old_coeff = meta["coefficients_ns"]
            assert np.allclose(list(coeff.values()), [old_coeff[k] for k in coeff], rtol=1e-7, atol=1e-7)
            assert diff < .001
            result["upstream_reproduction"] = dict(profile_mean_max_difference_ps=diff,
                                                    exact_bin_counts=True, coefficients_agree=True)
        else:
            stem = f"run4399_bar8_downbar3_ch{ch}_t0_quadratic"
            meta_path = ROOT / "Q_Walk/reference_quadratic_t0" / f"{stem}.json"
            profile_path = meta_path.with_name(f"{stem}_profile.csv")
            old_meta = json.loads(meta_path.read_text())
            old_profile = pd.read_csv(profile_path)
            old_fit = fit_reference_quadratic(old_profile)
            saved_coeff = old_meta["coefficients_ns"]
            assert np.allclose(old_fit.coefficients/1000,
                               [saved_coeff[k] for k in ("c", "b", "a")], rtol=1e-7, atol=1e-7)
            low = max(1/old_fit.x_max, 1/fit.x_max)
            high = min(1/old_fit.x_min, 1/fit.x_min)
            assert low < high
            result["legacy"] = old_meta["selection"]
            result["legacy_profile_reproduces_saved_coefficients"] = True
            result["old_vs_current_shape"] = compare_shapes(old_fit, fit, low, high)
            # Hold the old cut and bin edges fixed to distinguish clock/sample changes from rebinning.
            old_edges = np.r_[old_profile.inv_q_bin_low.iloc[0], old_profile.inv_q_bin_high]
            fixed_selected = prompt.loc[prompt.energy.ge(old_meta["selection"]["q_cut_adc"])]
            fixed_profile = build_comparison_profile(fixed_selected, n_bins=20, min_entries=5,
                                                     inv_q_edges=old_edges)
            fixed_fit = fit_reference_quadratic(fixed_profile)
            fixed_profile.to_csv(OUT / f"channel_{ch}_current_clock_legacy_bins_profile.csv", index=False)
            result["old_vs_current_clock_legacy_bins_shape"] = compare_shapes(
                old_fit, fixed_fit, max(1/old_fit.x_max, 1/fixed_fit.x_max),
                min(1/old_fit.x_min, 1/fixed_fit.x_min))
        inputs.extend([meta_path, profile_path])
        all_quality.append(set(prompt.event_id))
        all_peak.append(set(selected.event_id))
        supported = selected.energy.between(1/fit.x_max, 1/fit.x_min).to_numpy()
        inv = 1/selected.energy.to_numpy()
        occupied = np.zeros(len(selected), dtype=bool)
        for row in profile.itertuples():
            occupied |= (inv >= row.inv_q_bin_low) & (inv <= row.inv_q_bin_high)
        all_support.append(set(selected.loc[supported & occupied, "event_id"]))
        save(f"channel_{ch}_candidate.json", result)
        report["channels"][str(ch)] = result
    report["common_four_channel_events"] = dict(
        quality=len(set.intersection(*all_quality)),
        peak_right=len(set.intersection(*all_peak)),
        populated_fit_support=len(set.intersection(*all_support)))
    pd.Series(sorted(set.intersection(*all_support)), name="event_id").to_csv(
        OUT / "common_supported_event_ids.csv", index=False)
    save("report.json", report)
    save("input_manifest.json", [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
