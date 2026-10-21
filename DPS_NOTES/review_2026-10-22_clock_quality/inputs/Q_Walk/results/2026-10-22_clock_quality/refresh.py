"""Refresh measured-support AWC and bar-8 validation after the clock-quality rerun.

First run qwalk_extended_t0_reference.py with the directories in README.md.
Historical outputs and run4076 fits are never overwritten.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / "Q_Walk/results/2026-09-24_measured_q_overlap"
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closure():
    module = load("overlap_refresh", OLD / "recompute.py")
    original_paths = module._paths
    module._paths = lambda ch, slopes, refs: original_paths(ch, slopes, OUT / "references")
    module.OUT = OUT
    module.main()
    special = load("special_refresh", OLD / "plot_special_channels.py")
    special.OUT = OUT
    for argv in (["refresh", "--exclude-ch240"], ["refresh", "--exclude-ch240", "--fit-line"]):
        sys.argv = argv
        special.main()
    # Only the run4076 Q distributions are consumed by the internal predictor.
    spectra = OUT / "energy_spectra"
    spectra.mkdir(exist_ok=True)
    with np.load(OLD / "energy_spectra/quality_selected_q.npz") as source:
        np.savez_compressed(spectra / "quality_selected_q.npz",
                            **{key: source[key] for key in source.files if key.startswith("run4076_")})
    predictor = load("predictor_refresh", OLD / "try_physical_candidate.py")
    predictor.BASE = OUT
    sys.argv = ["refresh"]
    predictor.main()
    from Q_Walk.qwalk_downstream_t0_reference import _quality_mask
    data4399 = pd.read_csv(OUT / "cache/upstream_bar8_aligned.csv.gz")
    data4399 = data4399.loc[data4399.channelID.eq(137)]
    q4399 = data4399.loc[_quality_mask(data4399), "energy"].to_numpy()
    with np.load(spectra / "quality_selected_q.npz") as source:
        q4076 = source["run4076_ch137"]
    folder = OUT / "inputs/channel_137"
    slope = json.loads(next(folder.glob("*linear_summary.json")).read_text())
    reference = json.loads(next(folder.glob("*t0_quadratic.json")).read_text())
    edges = np.linspace(min(q4076.min(), q4399.min()), max(q4076.max(), q4399.max()), 81)
    table = dict(q_low_adc=edges[:-1], q_high_adc=edges[1:])
    for run, q, cut in ((4076, q4076, slope["q_selection"]["q_cut_adc"]),
                        (4399, q4399, reference["selection"]["q_cut_adc"])):
        for label, values in (("quality", q), ("peak_right", q[q >= cut])):
            counts, _ = np.histogram(values, bins=edges)
            assert counts.sum() == len(values)
            table[f"run{run}_{label}_count"] = counts
            table[f"run{run}_{label}_poisson_error"] = np.sqrt(counts)
    pd.DataFrame(table).to_csv(spectra / "channel_137_energy_spectrum.csv", index=False)
    previous = pd.read_csv(OLD / "channels.csv").set_index("channel")
    current = pd.read_csv(OUT / "channels.csv").set_index("channel")
    compared = current[["awc_ua", "rms_ps", "grid_points"]].join(
        previous[["awc_ua", "rms_ps", "grid_points"]], lsuffix="_new", rsuffix="_old")
    compared["awc_change_ua"] = compared.awc_ua_new - compared.awc_ua_old
    compared["awc_change_percent"] = 100 * compared.awc_change_ua / compared.awc_ua_old
    compared.to_csv(OUT / "reference_update_awc_impact.csv")


def validation():
    target = ROOT / "DPS_NOTES/review_2026-10-22_clock_quality"
    target.mkdir(exist_ok=True)
    module = load("timing_refresh", ROOT / "DPS_NOTES/review_2026-10-22_bar8_validation/validate.py")
    module.OUT = target

    def reference(ch):
        stem = f"run4399_upstream_bar8_ch{ch}_t0_quadratic"
        return (json.loads((OUT / "references" / f"{stem}.json").read_text()),
                pd.read_csv(OUT / "references" / f"{stem}_profile.csv"))

    module.reference = reference
    data = pd.read_csv(OUT / "cache/upstream_bar8_aligned.csv.gz")
    data = data.loc[data.channelID.isin([137, 150])].copy()
    data.to_csv(target / "matched_hits.csv.gz", index=False)
    report = json.loads((OUT / "reference_plots/run4399_extended_t0_reference_report.json").read_text())
    (target / "root_inputs.json").write_text(json.dumps(report["input_files"], indent=2) + "\n")
    shutil.copy2(OUT / "reference_plots/upstream_bar8_clock_groups.csv", target / "clock_groups.csv")
    module.analyze(400)
    audit = pd.read_csv(target / "clock_groups.csv")
    scales = audit.scaled_mad_ps.to_numpy()
    sensitivity = {str(threshold): dict(retained_groups=int(np.sum(np.isfinite(scales) & (scales <= threshold))),
                   retained_events=int(audit.loc[np.isfinite(scales) & (scales <= threshold), "events"].sum()))
                   for threshold in (1000, 10000, 100000, 1000000)}
    (target / "clock_threshold_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n")
    sources = [ROOT / "Q_Walk/qwalk_t0_data.py", ROOT / "Q_Walk/qwalk_extended_t0_reference.py",
               ROOT / "Q_Walk/qwalk_t0_models.py", ROOT / "Q_Walk/qwalk_core.py",
               ROOT / "DPS_NOTES/review_2026-10-22_bar8_validation/validate.py", Path(__file__)]
    sources += list((OUT / "references").glob("*bar8_ch1*.json"))
    sources += list((OUT / "references").glob("*bar8_ch1*_profile.csv"))
    manifest = []
    for source in sources:
        dest = target / "inputs" / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest.append(dict(source=str(source.relative_to(ROOT)), snapshot=str(dest.relative_to(target)),
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    (target / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def archived_comparison():
    module = load("archive_refresh", ROOT / "DPS_NOTES/review_2026-10-21_reference_comparison/compare.py")
    module.OUT = ROOT / "DPS_NOTES/review_2026-10-22_clock_quality/reference_comparison"
    module.OUT.mkdir(exist_ok=True)
    original_source = module.source

    def source(relative):
        if "2026-09-24_measured_q_overlap/inputs" in str(relative):
            relative = (OUT / "references" / relative.name).relative_to(ROOT)
        return original_source(relative)

    module.source = source
    module.main()


if __name__ == "__main__":
    closure()
    validation()
    archived_comparison()
