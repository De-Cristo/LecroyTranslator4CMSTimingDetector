"""Independently verify saved N2-3 arithmetic and provenance, without ROOT replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    report = json.loads((OUT / "report.json").read_text())
    for path, expected in {**report["input_hashes"], **report["code_hashes"]}.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert hashlib.sha256(Path(report["input"]).read_bytes()).hexdigest() == report["input_sha256"]
    channels = pd.read_csv(OUT / "channels.csv").set_index("channel")
    variants = pd.read_csv(OUT / "variants.csv")
    profiles = pd.read_csv(OUT / "profiles.csv")
    grids = pd.read_csv(OUT / "common_grid.csv")
    models = json.loads((OUT / "models.json").read_text())
    assert len(channels) == 22 and len(models) == len(variants) == 176
    assert channels.saved_nominal_reproduced.fillna(False).sum() == 21
    assert sum(channels.alignment_status == "ok") == 20
    assert channels.loc[235, "common_grid_points"] == 0
    assert 238 not in set(grids.channel)
    assert variants.profile_status.eq("ok").all()
    checked_alignments = 0
    for m in models:
        ch, name = m["channel"], m["variant"]
        p = profiles.loc[(profiles.channel == ch) & (profiles.variant == name)]
        good = p.loc[p.used_in_fit]
        chi2 = np.sum((good.residual / good.slope_sem_ua_per_ns) ** 2)
        np.testing.assert_allclose(chi2, m["chi2"], rtol=1e-9)
        np.testing.assert_allclose(p.slope_mean_ua_per_ns - p.prediction, p.residual, atol=1e-12)
        assert m["ndf"] == len(good) - m["degree"] - 1
        if name == "quadratic":
            baseline = variants.loc[(variants.channel == ch) & (variants.variant == "baseline")].iloc[0]
            assert m["chi2"] <= baseline.chi2 + 1e-8
            assert m["selected_hits"] == baseline.selected_hits
        if m["alignment_status"] != "ok":
            continue
        g = grids.loc[grids.channel == ch]
        q, y = g.q_adc.to_numpy(), g.reference_ns.to_numpy()
        assert q.min() >= p.q_mean_adc.min() - 1e-9
        assert q.max() <= p.q_mean_adc.max() + 1e-9
        covered = np.zeros(len(q), dtype=bool)
        for row in p.itertuples():
            covered |= (q >= row.q_low_adc - 1e-9) & (q <= row.q_high_adc + 1e-9)
        assert covered.all()
        z = (q - m["center"]) / m["scale"]
        slope = sum(c * z**k for k, c in enumerate(m["coefficients"]))
        assert (slope > 0).all()
        design = np.column_stack((1 / slope, np.ones(len(q))))
        awc, offset = np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose([awc, offset], [m["awc_ua"], m["offset_ns"]], rtol=1e-8, atol=1e-8)
        residual = 1000 * (design @ np.array([awc, offset]) - y)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), m["rms_ps"], rtol=1e-8)
        assert abs(residual.mean()) < 1e-7
        np.testing.assert_allclose(awc - channels.loc[ch, "baseline_awc_ua"], m["delta_awc_ua"], atol=1e-8)
        checked_alignments += 1
    assert checked_alignments == 160
    for ch, row in channels.loc[channels.alignment_status == "ok"].iterrows():
        v = variants.loc[variants.channel == ch]
        np.testing.assert_allclose(abs(v.delta_awc_ua).max(), row.max_abs_awc_shift)
        old = pd.read_csv(ROOT / f"Q_Walk/results/2026-10-22_clock_quality/channel_{ch}_comparison.csv")
        g = grids.loc[grids.channel == ch]
        # Compare nearest saved grid coordinate, retaining gaps rather than filling them.
        indices = np.searchsorted(old.q_adc.to_numpy(), g.q_adc.to_numpy())
        indices = np.clip(indices, 0, len(old) - 1)
        for i, x, y in zip(indices, g.q_adc, g.reference_ns):
            candidates = old.iloc[max(0, i - 1):i + 1]
            match = candidates.iloc[np.argmin(abs(candidates.q_adc.to_numpy() - x))]
            np.testing.assert_allclose([x, y], [match.q_adc, match.reference_ns], rtol=0, atol=1e-9)
    pdfs = sorted((OUT / "figures").glob("*.pdf"))
    assert len(pdfs) == 25
    for path in pdfs:
        reader = PdfReader(path)
        assert len(reader.pages) == 1
        assert "Preliminary Phase 2" in reader.pages[0].extract_text(), path
    result = dict(status="passed", channels=22, profile_fits=176,
                  independently_recomputed_alignments=checked_alignments,
                  nominal_replays_recorded=21, diagnostic_figures=len(pdfs),
                  scientific_status="N2-3 remains open for model/range decision; diagnostics complete",
                  frozen_inputs_unchanged=True)
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
