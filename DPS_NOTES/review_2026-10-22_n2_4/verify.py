"""Independently verify N2-4 covariance, weighted fits and mixture probabilities."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy.special import ndtr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / "Q_Walk/results/2026-10-22_clock_quality"


def main():
    report = json.loads((OUT / "report.json").read_text())
    for name, digest in report["input_and_code_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    rows = pd.read_csv(OUT / "uncertainties.csv").set_index("channel")
    alternatives = pd.read_csv(OUT / "weighting.csv")
    assert len(rows) == 21 and len(alternatives) == 84
    assert set(rows.index) == set(pd.read_csv(BASE / "channels.csv").channel)
    ref_count = 0
    for path in (BASE / "references").glob("*.json"):
        metadata = json.loads(path.read_text())
        p = pd.read_csv(path.with_name(path.stem + "_profile.csv"))
        x = 1000 * p.inv_q_mean.to_numpy()
        design = np.column_stack((np.ones(len(x)), x, x**2))
        sem = p.delta_t_ref_sem_ps.to_numpy() / 1000
        weighted = design / sem[:, None]
        coef = np.linalg.lstsq(weighted, p.delta_t_ref_mean_ps.to_numpy() / 1000 / sem, rcond=None)[0]
        transform = np.diag([1., 1000., 1e6])
        cov = transform @ np.linalg.inv(weighted.T @ weighted) @ transform
        np.testing.assert_allclose(transform @ coef, [metadata["coefficients_ns"][k] for k in ("a", "b", "c")], rtol=1e-7)
        np.testing.assert_allclose(cov, metadata["covariance_ns2_abc"], rtol=1e-6)
        ref_count += 1
    fits_checked = 0
    for ch, row in rows.iterrows():
        old = json.loads((BASE / f"channel_{ch}.json").read_text())
        np.testing.assert_allclose([row.awc_ua, row.offset_ns], [old["awc_ua"], old["offset_ns"]], rtol=1e-8)
        np.testing.assert_allclose([row.slope_only_p16, row.slope_only_p50, row.slope_only_p84],
                                   [old[f"awc_p{k}"] for k in (16, 50, 84)], rtol=1e-8)
        with np.load(OUT / f"channel_{ch}_propagation.npz") as data:
            q = data["q"]
            beta, cov = data["reference_coefficients"], data["reference_covariance"]
            basis = np.column_stack((np.ones(len(q)), 1 / q, 1 / q**2))
            coeff = data["slope_coefficients"]
            slope = coeff[0] + coeff[1] * q
            design = np.column_stack((1 / slope, np.ones(len(q))))
            grid = pd.read_csv(BASE / f"channel_{ch}_comparison.csv")
            np.testing.assert_array_equal(q, grid.q_adc)
            for r in alternatives.loc[alternatives.channel == ch].itertuples():
                w = data[f"weight_{r.weighting}"]
                matrix = np.linalg.lstsq(design * np.sqrt(w[:, None]), basis * np.sqrt(w[:, None]), rcond=None)[0]
                np.testing.assert_allclose(matrix @ beta, [r.awc_ua, r.offset_ns], rtol=1e-7, atol=1e-8)
                assert r.grid_points == len(q)
                fits_checked += 1
            for i in (0, len(data["bootstrap_map"]) // 2, len(data["bootstrap_map"]) - 1):
                c = data["slope_bootstrap_coefficients"][i]
                d = np.column_stack((1 / (c[0] + c[1] * q), np.ones(len(q))))
                matrix = np.linalg.lstsq(d, basis, rcond=None)[0]
                np.testing.assert_allclose(matrix, data["bootstrap_map"][i], rtol=1e-7, atol=1e-8)
                np.testing.assert_allclose(matrix @ cov @ matrix.T, data["conditional_covariance"][i], rtol=1e-7)
            means = data["conditional_means"][:, 0]
            sigmas = np.sqrt(data["conditional_covariance"][:, 0, 0])
            for k in (16, 50, 84):
                np.testing.assert_allclose(ndtr((row[f"combined_p{k}"] - means) / sigmas).mean(), k / 100, atol=1e-9)
            assert np.all(data["bootstrap_map"][:, 0, 0] == 0)
            assert row.slope_bootstrap_invalid == 0
            # Total mixture variance includes, rather than replaces, reference variance.
            assert np.var(means) + np.mean(sigmas**2) >= np.var(means)
        shifts = alternatives.loc[alternatives.channel == ch, "delta_awc_ua"]
        np.testing.assert_allclose(abs(shifts).max(), row.max_weighting_shift_ua)
    for path in (OUT / "figures").glob("*.pdf"):
        reader = PdfReader(path)
        assert len(reader.pages) == 1 and "Preliminary Phase 2" in reader.pages[0].extract_text()
    assert len(list((OUT / "figures").glob("*.pdf"))) == 2
    result = dict(status="passed", independently_reproduced_reference_covariances=ref_count,
                  independently_recomputed_weighted_fits=fits_checked,
                  verified_combined_interval_channels=len(rows), original_results_unchanged=True,
                  method="fixed-calibration statistical propagation; not a total uncertainty budget")
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
