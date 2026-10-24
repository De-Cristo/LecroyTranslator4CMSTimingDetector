"""Recalculate profile-level walk residuals with explicit conditional SEM bars."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets/note2_residual_profile"
PROFILE = ROOT.parent / "note1_mcp_reference/data/sandbox_test/Q_Walk/reference_quadratic_t0/ch192_vs_trig_lowess_bar8_ch137_walk_before_walk_fit_expected_refit_profile.csv"
SLOPE = ROOT.parent / "note2_dual_threshold/data/sandbox_test/Q_Walk/results/2026-10-22_clock_quality/inputs/channel_137/channel_137_safe_time2_two_upstream_peak_right_profile_linear_summary.json"
HISTORICAL = ROOT / "assets/note2_figure5/original_figure5.jpg"
AWC_UA = 20.0  # Supplied detector Note 2, Table 2, bar 8 left SiPM.


def historical_comparison(q, residual_ps):
    """Digitize marker centers only for an audit, never for calculation or fitting."""
    a = np.asarray(Image.open(HISTORICAL).convert("RGB")).astype(float)
    assert a.shape == (1030, 1630, 3)
    mask = ((a[:, :, 2] - a[:, :, 0] > 35)
            & (a[:, :, 1] - a[:, :, 0] > 20)
            & (a[:, :, 0] < 180) & (a[:, :, 2] > 130))
    distance = distance_transform_edt(mask)
    x_map = np.polyfit([127.5, 370.8, 614.3, 857.7, 1101.4, 1344.7, 1588],
                       np.arange(150, 451, 50), 1)
    y_map = np.polyfit([130.2, 303.8, 477.6, 651.4, 825.2],
                       [20, 10, 0, -10, -20], 1)
    old = []
    for charge in q:
        px = int(round((charge - x_map[1]) / x_map[0]))
        search = distance[45:930, px - 4:px + 5]
        y, x = np.unravel_index(np.argmax(search), search.shape)
        assert search[y, x] >= 8, "Historical blue marker not identified"
        old.append(float(np.polyval(y_map, y + 45)))
    old = np.asarray(old)
    offset_ps = float(np.mean(residual_ps - old))
    difference = residual_ps - offset_ps - old
    pd.DataFrame({"q_effective_adc": q, "historical_digitized_residual_ps": old,
                  "new_residual_ps": residual_ps,
                  "difference_after_constant_alignment_ps": difference}).to_csv(
                      OUTPUT / "historical_marker_audit.csv", index=False)
    return {
        "purpose": "Diagnostic comparison only; digitized points are not calculation inputs.",
        "bins": len(q), "constant_alignment_ps": offset_ps,
        "shape_difference_rms_ps": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_difference_ps": float(np.max(np.abs(difference))),
        "ps_per_pixel": float(abs(y_map[0])),
        "conclusion": "Close but not identical; present as a new calculation, not an exact reproduction of Figure 5.",
    }


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(PROFILE)
    summary = json.loads(SLOPE.read_text())
    fit = summary["linear_profile_fit"]
    c0 = fit["intercept_ua_per_ns"]
    c1 = fit["coefficient_ua_per_ns_per_adc"]
    bins = summary["profile"]["points"]
    q_min, q_max = bins[0]["q_low_adc"], bins[-1]["q_high_adc"]
    q = 1 / table.invE_mean.to_numpy()
    sem = table.delta_t_sem.to_numpy()
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(sem)) and np.all(sem > 0)
    use = (q >= q_min) & (q <= q_max)
    assert use.sum() == 43 and len(table) == 48
    slope = c0 + c1 * q[use]
    assert np.all(slope > 0)
    corrected = table.delta_t_mean.to_numpy()[use] - AWC_UA / slope
    weights = 1 / sem[use]**2
    offset = float(np.average(corrected, weights=weights))
    residual_ps = (corrected - offset) * 1000
    sem_ps = sem[use] * 1000
    charge = q[use]
    rms_ps = float(np.sqrt(np.mean(residual_ps**2)))

    # Diagnostic line only: do not subtract it or use it to change AWC.
    pivot = float(np.average(charge, weights=weights))
    design = np.column_stack((np.ones(len(charge)), charge - pivot))
    whitened = design / sem_ps[:, None]
    line = np.linalg.lstsq(whitened, residual_ps / sem_ps, rcond=None)[0]
    covariance = np.linalg.inv(whitened.T @ whitened)
    table["q_effective_adc"] = q
    table["selected"] = use
    table["selection_reason"] = np.where(use, "within_slope_domain", "outside_slope_domain")
    for name, values in (("mean_slope_ua_per_ns", slope),
                         ("correction_ns", AWC_UA / slope),
                         ("residual_ps", residual_ps),
                         ("displayed_sem_ps", sem_ps)):
        table[name] = np.nan
        table.loc[use, name] = values
    table.to_csv(OUTPUT / "residual_profile.csv", index=False)

    with plt.rc_context({"pdf.fonttype": 42, "font.size": 12}):
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        fig.subplots_adjust(left=.14, right=.98, bottom=.15, top=.73)
        ax.axhspan(-10, 10, color="#359c79", alpha=.10)
        for y in (-10, 10):
            ax.axhline(y, color="#23795d", ls="--", lw=1.2)
        ax.axhline(0, color="0.55", lw=.7)
        ax.errorbar(charge, residual_ps, yerr=sem_ps, fmt="o", ms=4.5,
                    capsize=2, lw=1, color="#2277aa", label=r"Profile residual $\pm$ SEM")
        grid = np.linspace(charge.min(), charge.max(), 200)
        ax.plot(grid, line[0] + line[1] * (grid - pivot), color="#a33f36",
                lw=1.5, label="Diagnostic linear fit")
        ax.set(xlim=(150, 450), ylim=(-30, 30), xticks=np.arange(150, 451, 50),
               yticks=np.arange(-30, 31, 10), xlabel=r"Charge proxy $Q$ [ADC]",
               ylabel=r"Residual $r(Q)$ [ps]")
        ax.grid(alpha=.2)
        ax.text(.98, .04, r"$\pm10$ ps guide", color="#23795d", ha="right",
                transform=ax.transAxes, fontsize=11)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2,
                  frameon=False, fontsize=10)
        fig.text(.14, .97, r"$\bf{CMS}$ Preliminary Phase 2", fontsize=14, va="top")
        fig.text(.14, .90, "Reference module, bar 8, left SiPM", fontsize=12, va="top")
        fig.text(.14, .84, "New calculation from saved profiles", fontsize=11, va="top")
        fig.savefig(OUTPUT / "awc_residual_profile.pdf")
        fig.savefig(OUTPUT / "awc_residual_profile.png", dpi=180)
        plt.close(fig)

    record = {
        "sources": [{"path_relative_to_DPS_NOTES": str(p.relative_to(ROOT.parent)),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in (PROFILE, SLOPE, HISTORICAL)],
        "calculation": "Profile-level mean time minus AWC/(c0+c1*Q_eff), then an inverse-variance fitted constant offset; not an event-level correction.",
        "q_definition": "Q_eff = 1 / mean(1/Q) in each saved inverse-charge bin; not mean(Q). Bin representative must lie within the slope-profile domain; full-bin containment is not asserted.",
        "time_units": "Input mean and SEM are ns; existing +26 ns display offset is not added again; residual output is ps.",
        "awc_ua": AWC_UA,
        "awc_source": "Supplied detector Note 2, Table 2, bar 8 ch137 (left SiPM): 20 uA; not fitted in this calculation.",
        "c0_ua_per_ns": c0, "c1_ua_per_ns_per_adc": c1,
        "slope_domain_adc": [q_min, q_max],
        "selected_q_range_adc": [float(charge.min()), float(charge.max())],
        "input_bins": len(table), "selected_bins": int(use.sum()),
        "exclusions": "Five bin representatives below the slope domain; no residual-based cuts or extrapolation.",
        "fitted_offset_ns": offset,
        "displayed_errors": "Saved input profile-mean SEM, converted ns to ps. Correction and fitted offset treated as fixed for display. Not total post-fit residual uncertainties.",
        "excluded_uncertainties": ["Slope-fit and AWC errors", "Shared timing/calibration systematics",
                                   "Offset-estimation covariance and resulting bin correlations",
                                   "Finite-bin correction bias and charge-coordinate uncertainty",
                                   "Transfer between independent acquisitions"],
        "bin_residual_rms_ps": rms_ps,
        "weighted_bin_residual_rms_ps": float(np.sqrt(np.average(residual_ps**2, weights=weights))),
        "central_points_outside_10ps": int((np.abs(residual_ps) > 10).sum()),
        "diagnostic_line": {"pivot_q_adc": pivot, "intercept_ps": float(line[0]),
                            "slope_ps_per_adc": float(line[1]),
                            "fixed_sem_covariance": covariance.tolist(),
                            "chi2": float(np.sum(((residual_ps - design @ line) / sem_ps)**2)),
                            "ndf": len(charge) - 2, "applied_as_correction": False},
        "interpretation": "About 6.1 ps RMS of bin-mean residuals supports the walk form at roughly 10 ps scale in this example, not a single-hit resolution, all-point bound, or test of reference-free AWC prediction.",
        "historical_comparison": historical_comparison(charge, residual_ps),
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
