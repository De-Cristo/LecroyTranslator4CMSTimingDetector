"""Propagate saved profile SEM through the original unweighted quadratic fit."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent / "note1_mcp_reference/data/sandbox_test/Q_Walk/reference_quadratic_t0"
OUTPUT = ROOT / "assets/note2_quadratic_uncertainties"
CHANNELS = ((7, "Left", 133), (7, "Right", 154), (8, "Left", 137), (8, "Right", 150))


def propagate(x, y, sem):
    x, y, sem = (np.asarray(a, dtype=float) for a in (x, y, sem))
    assert x.ndim == 1 and x.shape == y.shape == sem.shape and x.size > 3
    assert np.all(np.isfinite([x, y, sem])) and np.all(sem > 0)
    center, scale = float(x.mean()), float(np.ptp(x) / 2)
    assert scale > 0
    z = (x - center) / scale
    design = np.column_stack((np.ones(x.size), z, z**2))
    assert np.linalg.matrix_rank(design) == 3
    # Centering stabilizes the solve; transform back to a + b*x + c*x^2.
    transform = np.array([[1, -center / scale, center**2 / scale**2],
                          [0, 1 / scale, -2 * center / scale**2],
                          [0, 0, 1 / scale**2]])
    response = transform @ np.linalg.pinv(design)
    coefficients = response @ y
    covariance = (response * sem) @ (response * sem).T
    assert np.allclose(coefficients, np.polyfit(x, y, 2)[::-1], rtol=1e-9, atol=1e-9)
    return coefficients, covariance, response


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, tables, records = [], [], []
    for bar, side, channel in CHANNELS:
        base = f"ch192_vs_trig_lowess_bar{bar}_ch{channel}_walk_before_walk_fit_expected_refit"
        profile_path = INPUT / f"{base}_profile.csv"
        coeff_path = INPUT / f"{base}_coeffs.csv"
        profile = np.genfromtxt(profile_path, delimiter=",", names=True)
        with coeff_path.open(newline="") as stream:
            original = next(row for row in csv.DictReader(stream) if row["fit_kind"] == "binned")
        expected = np.array([float(original[k]) for k in ("p0", "p1", "p2")])
        coefficients, covariance, _ = propagate(
            profile["invE_mean"], profile["delta_t_mean"], profile["delta_t_sem"])
        assert np.allclose(coefficients, expected, rtol=1e-9, atol=1e-9)
        errors = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(errors, errors)
        rows.append(dict(bar=bar, sipm=side, channel=channel, bins=len(profile),
                         a_ns=expected[0], a_stat_ns=errors[0],
                         b_ns_adc=expected[1], b_stat_ns_adc=errors[1],
                         c_ns_adc2=expected[2], c_stat_ns_adc2=errors[2]))
        cells = [f"${expected[0]:.3f} \\pm {errors[0]:.3f}$",
                 f"${expected[1]:.1f} \\pm {errors[1]:.1f}$",
                 f"${expected[2]/1e4:.2f} \\pm {errors[2]/1e4:.2f}$"]
        tables.append(f"{bar} & {side} & " + " & ".join(cells) + r"\\")
        records.append(dict(channel=channel, bar=bar, sipm=side, bins=len(profile),
                            selected_profile_hits=int(profile["count"].sum()),
                            coefficients_abc=expected.tolist(), standard_errors=errors.tolist(),
                            covariance_abc=covariance.tolist(), correlation_abc=correlation.tolist(),
                            original_fit_reproduced=True,
                            sources=[dict(path=str(p.relative_to(ROOT.parent)),
                                          sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                     for p in (profile_path, coeff_path)]))
    with (OUTPUT / "coefficients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    header = [r"\begin{tabular}{rrrrr}", r"\toprule",
              r"Bar & SiPM & $a_i$ [ns] & $b_i$ [ns ADC] & $c_i$ [$10^4$ ns ADC$^2$]\\",
              r"\midrule"]
    (OUTPUT / "table.tex").write_text("\n".join(header + tables + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    report = dict(
        method="Original unweighted least squares in x=1/Q; propagate independent saved bin-mean SEM through the fixed linear estimator.",
        covariance="J diag(SEM^2) J^T, with J the unweighted least-squares response in coefficient order [a,b,c]. No residual chi-square scaling.",
        assumptions="Fixed bin coordinates, binning and selection; independent bin-mean errors. Shared clock/calibration correlations are not propagated.",
        exclusions=["clock and other calibration systematics", "charge-coordinate uncertainty",
                    "selection and binning variation", "functional-form uncertainty", "inter-bin correlations"],
        units=["ns", "ns ADC", "ns ADC^2"],
        time_convention="Saved profile means are already in ns with the historical +26 ns display offset; no additional conversion or offset is applied.",
        table_c_scale=10000, original_coefficients_unchanged=True, channels=records)
    (OUTPUT / "covariance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
