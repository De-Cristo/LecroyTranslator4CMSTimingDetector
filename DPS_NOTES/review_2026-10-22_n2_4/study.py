"""Propagate full MCP covariance and audit weights without changing nominal fits."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / "Q_Walk/results/2026-10-22_clock_quality"
sys.path.insert(0, str(ROOT))
from Q_Walk.qwalk_t0_models import fit_reference_quadratic


def alignment_map(slope, q, weights):
    """Return M with (AWC, offset) = M @ (a,b,c), for one or many slopes."""
    slope, q, weights = (np.asarray(x, dtype=float) for x in (slope, q, weights))
    if (not np.isfinite(slope).all() or np.any(slope <= 0)
            or not np.isfinite(q).all() or np.any(q <= 0)
            or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid slope, charge or weights")
    w = weights / weights.sum()
    x = 1 / slope
    xmean = np.sum(x * w, axis=-1, keepdims=True)
    xc = x - xmean
    denominator = np.sum(w * xc**2, axis=-1)
    if np.any(denominator <= np.finfo(float).eps**2 * np.sum(w * x**2, axis=-1)):
        raise ValueError("Flat inverse slope")
    basis = np.column_stack((np.ones(len(q)), 1 / q, 1 / q**2))
    mean_basis = w @ basis
    awc = ((w * xc) @ (basis - mean_basis)) / denominator[..., None]
    # The free offset absorbs the constant reference coefficient exactly.
    awc[..., 0] = 0
    offset = mean_basis - xmean * awc
    return np.stack((awc, offset), axis=-2)


def propagated_covariance(matrix, covariance):
    result = matrix @ covariance @ np.swapaxes(matrix, -1, -2)
    if np.any(np.diagonal(result, axis1=-2, axis2=-1) < 0):
        raise ValueError("Negative propagated variance")
    return result


def mixture_percentiles(means, sigmas, probabilities=(.16, .5, .84)):
    """Quantiles of an equally weighted Gaussian mixture over slope replicas."""
    means, sigmas = np.asarray(means), np.asarray(sigmas)
    if not np.isfinite(means).all() or not np.isfinite(sigmas).all() or np.any(sigmas < 0):
        raise ValueError("Invalid Gaussian components")
    if np.all(sigmas == 0):
        return np.quantile(means, probabilities)
    if np.any(sigmas == 0):
        raise ValueError("Mixed point-mass and Gaussian components are unsupported")
    lo, hi = np.min(means - 12 * sigmas), np.max(means + 12 * sigmas)
    return np.array([brentq(lambda value: ndtr((value - means) / sigmas).mean() - p,
                           lo, hi, xtol=1e-10) for p in probabilities])


def density_weights(q, low, high, counts):
    low, high, counts = (np.asarray(x, dtype=float) for x in (low, high, counts))
    order = np.argsort(low)
    low, high, counts = low[order], high[order], counts[order]
    if np.any(high <= low) or np.any(counts <= 0):
        raise ValueError("Invalid populated bin")
    index = np.clip(np.searchsorted(low, q, side="right") - 1, 0, len(low) - 1)
    if np.any(q < low[index] - 1e-8) or np.any(q > high[index] + 1e-8):
        raise ValueError("Grid outside populated support")
    return counts[index] / (high[index] - low[index])


def check_covariance(covariance):
    std = np.sqrt(np.diag(covariance))
    if not np.isfinite(covariance).all() or np.any(std <= 0):
        raise ValueError("Invalid reference covariance")
    correlation = covariance / np.outer(std, std)
    np.testing.assert_allclose(correlation, correlation.T, atol=1e-10)
    if np.linalg.eigvalsh(correlation).min() <= 0:
        raise ValueError("Reference covariance is not positive definite")


def main():
    rows, alternatives, checks, hashes = [], [], [], {}

    def record(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    for reference_path in sorted((BASE / "references").glob("*.json")):
        reference = json.loads(reference_path.read_text())
        ch = reference["channel"]
        beta = np.array([reference["coefficients_ns"][key] for key in ("a", "b", "c")])
        covariance = np.asarray(reference["covariance_ns2_abc"])
        check_covariance(covariance)
        reference_profile_path = reference_path.with_name(reference_path.stem + "_profile.csv")
        reference_profile = pd.read_csv(reference_profile_path)
        refit = fit_reference_quadratic(reference_profile)
        np.testing.assert_allclose(refit.coefficients[::-1] / 1000, beta, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(refit.covariance[::-1, ::-1] / 1e6, covariance, rtol=1e-8)
        yref = reference_profile.delta_t_ref_mean_ps.to_numpy() / 1000
        invq = reference_profile.inv_q_mean.to_numpy()
        chi2 = np.sum(((yref - np.polynomial.polynomial.polyval(invq, beta)) /
                       (reference_profile.delta_t_ref_sem_ps.to_numpy() / 1000))**2)
        checks.append(dict(channel=ch, coefficient_and_covariance_replay=True,
                           reference_chi2=float(chi2), ndf=len(invq) - 3,
                           covariance_scaled_by_chi2=False))
        record(reference_path)
        record(reference_profile_path)
        grid_path = BASE / f"channel_{ch}_comparison.csv"
        if not grid_path.exists():
            assert ch == 238
            continue
        grid = pd.read_csv(grid_path)
        q, y = grid.q_adc.to_numpy(), grid.reference_ns.to_numpy()
        np.testing.assert_allclose(y, np.polynomial.polynomial.polyval(1 / q, beta), atol=1e-11)
        folder = BASE / f"inputs/channel_{ch}"
        summary_path = next(folder.glob("*_profile_linear_summary.json"))
        summary = json.loads(summary_path.read_text())
        slope_profile = pd.DataFrame(summary["profile"]["points"])
        fit = summary["linear_profile_fit"]
        coeff = np.array([fit["intercept_ua_per_ns"], fit["coefficient_ua_per_ns_per_adc"]])
        boot_path = folder / summary["linear_profile_bootstrap"]["coefficients_file"]
        with np.load(boot_path) as data:
            boot = data["coefficients"]
        slope = coeff[0] + coeff[1] * q
        bs = boot[:, [0]] + boot[:, [1]] * q
        valid = np.isfinite(bs).all(axis=1) & (bs > 0).all(axis=1)
        bs = bs[valid]
        weights = dict(uniform_q=np.ones(len(q)), uniform_inverse_q=1 / q**2,
                       population4076=density_weights(q, slope_profile.q_low_adc,
                                                       slope_profile.q_high_adc, slope_profile["count"]),
                       population4399=density_weights(q, 1 / reference_profile.inv_q_bin_high,
                                                       1 / reference_profile.inv_q_bin_low,
                                                       reference_profile.count_after_clip))
        m0 = alignment_map(slope, q, weights["uniform_q"])
        mb = alignment_map(bs, q, weights["uniform_q"])
        central = m0 @ beta
        means = mb @ beta
        conditional_cov = propagated_covariance(mb, covariance)
        conditional_std = np.sqrt(np.diagonal(conditional_cov, axis1=-2, axis2=-1))
        reference_cov = propagated_covariance(m0, covariance)
        old_path = BASE / f"channel_{ch}.json"
        old = json.loads(old_path.read_text())
        np.testing.assert_allclose(central, [old["awc_ua"], old["offset_ns"]], rtol=1e-8, atol=1e-8)
        slope_percentiles = np.percentile(means[:, 0], [16, 50, 84])
        np.testing.assert_allclose(slope_percentiles, [old[f"awc_p{k}"] for k in (16, 50, 84)],
                                   rtol=1e-8, atol=1e-8)
        combined = mixture_percentiles(means[:, 0], conditional_std[:, 0])
        offset_interval = mixture_percentiles(means[:, 1], conditional_std[:, 1])
        ref_sigma = float(np.sqrt(reference_cov[0, 0]))
        ref_interval = central[0] + ref_sigma * ndtri([.16, .5, .84])
        row = dict(channel=ch, module=reference["module"], bar=reference["bar"],
                   awc_ua=float(central[0]), offset_ns=float(central[1]),
                   reference_sigma_ua=ref_sigma, reference_only_p16=float(ref_interval[0]),
                   reference_only_p84=float(ref_interval[2]), grid_points=len(q),
                   slope_bootstrap_valid=int(valid.sum()), slope_bootstrap_invalid=int((~valid).sum()),
                   reference_chi2_ndf=float(chi2 / (len(invq) - 3)))
        for name, values in (("slope_only", slope_percentiles), ("combined", combined),
                             ("combined_offset", offset_interval)):
            row.update({f"{name}_p{k}": float(value) for k, value in zip((16, 50, 84), values)})
        row["interval_width_ratio"] = float((combined[2] - combined[0]) / (slope_percentiles[2] - slope_percentiles[0]))
        shifts = []
        for name, w in weights.items():
            matrix = alignment_map(slope, q, w)
            parameters = matrix @ beta
            residual = parameters[0] / slope + parameters[1] - y
            shift = float(parameters[0] - central[0])
            shifts.append(abs(shift))
            alternatives.append(dict(channel=ch, module=reference["module"], weighting=name,
                                     awc_ua=float(parameters[0]), offset_ns=float(parameters[1]),
                                     delta_awc_ua=shift, delta_awc_percent=100 * shift / abs(central[0]),
                                     rms_uniform_q_ps=float(1000 * np.sqrt(np.mean(residual**2))),
                                     rms_weighted_ps=float(1000 * np.sqrt(np.average(residual**2, weights=w))),
                                     grid_points=len(q), negative_awc=bool(parameters[0] < 0)))
        row["max_weighting_shift_ua"] = max(shifts)
        rows.append(row)
        np.savez_compressed(OUT / f"channel_{ch}_propagation.npz", q=q, nominal_map=m0,
                            bootstrap_map=mb, reference_coefficients=beta, reference_covariance=covariance,
                            conditional_means=means, conditional_covariance=conditional_cov,
                            slope_coefficients=coeff, slope_bootstrap_coefficients=boot[valid],
                            **{f"weight_{key}": value for key, value in weights.items()})
        for path in (grid_path, old_path, summary_path, boot_path):
            record(path)
        print(f"ch{ch}: AWC={central[0]:.3f}; slope={slope_percentiles[[0,2]]}; combined={combined[[0,2]]}; max weight shift={max(shifts):.3f}", flush=True)
    pd.DataFrame(rows).sort_values("channel").to_csv(OUT / "uncertainties.csv", index=False)
    pd.DataFrame(alternatives).sort_values(["channel", "weighting"]).to_csv(OUT / "weighting.csv", index=False)
    pd.DataFrame(checks).sort_values("channel").to_csv(OUT / "reference_checks.csv", index=False)
    for path in (Path(__file__), OUT / "study_definition.md", ROOT / "Q_Walk/qwalk_t0_models.py"):
        record(path)
    report = dict(reference_fits_reproduced=len(checks), completed_channels=len(rows),
                  no_overlap_channels=[238], nominal_central_values_unchanged=True,
                  interval_method="Gaussian reference covariance mixed over saved independent-run slope bootstrap replicas",
                  weights=list(weights), input_and_code_hashes=hashes,
                  exclusions="Clock calibration, shared fine-time calibration, cross-run response, selections, model bias and cross-channel correlations are not propagated")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
