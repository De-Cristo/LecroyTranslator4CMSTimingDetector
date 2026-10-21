"""Canonical profiles and curve models for the Q-walk/t0 comparison.

Polynomial coefficients are stored in descending-power ``numpy.polyval``
order and in the physical input basis. Reference fits use inverse charge
``1/Q`` and ps; slope fits use charge ``Q`` and uA/ns. The affine transfer is
``time_ps = offset_ps + scale_ps_ua_per_ns * inverse_slope_ns_per_ua``.
The transfer scale therefore has units ps*uA/ns; ``scale_ua`` remains only as
a compatibility property.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

if not hasattr(np, "RankWarning") and hasattr(np, "exceptions"):
    np.RankWarning = np.exceptions.RankWarning


PROFILE_COLUMNS = (
    "bin_index",
    "inv_q_bin_low",
    "inv_q_bin_high",
    "count_before_clip",
    "count_after_clip",
    "inv_q_mean",
    "inv_q_std",
    "inv_q_sem",
    "energy_mean",
    "energy_std",
    "energy_sem",
    "delta_t_ref_mean_ps",
    "delta_t_ref_std_ps",
    "delta_t_ref_sem_ps",
    "slope_mean_ua_per_ns",
    "slope_std_ua_per_ns",
    "slope_sem_ua_per_ns",
    "inv_slope_mean_ns_per_ua",
    "inv_slope_std_ns_per_ua",
    "inv_slope_sem_ns_per_ua",
    "inverse_mean_slope_ns_per_ua",
)


REQUIRED_METRIC_KEYS = frozenset(
    {
        "fold_status",
        "fold_reason",
        "training_event_count",
        "validation_event_count",
        "accepted_validation_event_count",
        "matched_validation_event_count",
        "matched_validation_event_status",
        "retained_training_bin_count",
        "retained_validation_bin_count",
        "matched_validation_bin_count",
        "matched_validation_bin_status",
        "accepted_inv_q_min",
        "accepted_inv_q_max",
        "accepted_q_min",
        "accepted_q_max",
        "pivot_q",
        "shape_rmse_lowess",
        "shape_max_deviation_lowess",
        "shape_rmse_quadratic",
        "shape_max_deviation_quadratic",
        "slope_validation_rmse_lowess_ua_per_ns",
        "slope_validation_rmse_quadratic_ua_per_ns",
        "correction_rmse_lowess_ps",
        "correction_mad_lowess_ps",
        "correction_rmse_quadratic_ps",
        "correction_mad_quadratic_ps",
        "weighted_chi2_lowess",
        "weighted_chi2_dof_lowess",
        "weighted_chi2_per_dof_lowess",
        "weighted_chi2_quadratic",
        "weighted_chi2_dof_quadratic",
        "weighted_chi2_per_dof_quadratic",
        "residual_slope_raw_ps_per_inv_q",
        "residual_slope_reference_ps_per_inv_q",
        "residual_slope_lowess_ps_per_inv_q",
        "residual_slope_quadratic_ps_per_inv_q",
        "robust_width_raw_ps",
        "robust_width_reference_ps",
        "robust_width_slope_lowess_ps",
        "robust_width_slope_quadratic_ps",
        "gaussian_width_raw_ps",
        "gaussian_width_reference_ps",
        "gaussian_width_slope_lowess_ps",
        "gaussian_width_slope_quadratic_ps",
        "gaussian_width_raw_uncertainty_ps",
        "gaussian_width_reference_uncertainty_ps",
        "gaussian_width_slope_lowess_uncertainty_ps",
        "gaussian_width_slope_quadratic_uncertainty_ps",
        "gaussian_width_raw_count",
        "gaussian_width_reference_count",
        "gaussian_width_slope_lowess_count",
        "gaussian_width_slope_quadratic_count",
        "gaussian_width_raw_status",
        "gaussian_width_reference_status",
        "gaussian_width_slope_lowess_status",
        "gaussian_width_slope_quadratic_status",
        "width_raw_matched_event_count",
        "width_reference_matched_event_count",
        "width_slope_lowess_matched_event_count",
        "width_slope_quadratic_matched_event_count",
    }
)

STATUS_TRACKED_METRIC_KEYS = frozenset(
    {
        key
        for key in REQUIRED_METRIC_KEYS
        if key.startswith(
            (
                "shape_",
                "slope_validation_",
                "correction_",
                "weighted_chi2_",
                "residual_slope_",
                "robust_width_",
                "gaussian_width_",
            )
        )
        and not key.endswith("_status")
    }
)
REQUIRED_METRIC_KEYS = REQUIRED_METRIC_KEYS | frozenset(
    "{}_status".format(key) for key in STATUS_TRACKED_METRIC_KEYS
)


@dataclass(frozen=True)
class SegmentFold:
    """One leave-one-``(file_id, segment)``-out split using index labels."""

    fold_id: str
    held_out_group: Tuple[Any, Any]
    train_indices: pd.Index
    validation_indices: pd.Index
    held_out_groups: Tuple[Tuple[Any, Any], ...] = ()

    @property
    def groups(self):
        """Return every indivisible ``(file_id, segment)`` validation group."""
        return self.held_out_groups or (self.held_out_group,)


@dataclass(frozen=True)
class TimingWidth:
    """Result of a Gaussian core-width fit in picoseconds."""

    sigma_ps: float
    uncertainty_ps: float
    count: int
    status: str


@dataclass(frozen=True)
class BootstrapInterval:
    """Pointwise central 68 percent bootstrap interval."""

    p16: Any
    p50: Any
    p84: Any
    finite_count: Any
    status: Any


@dataclass(frozen=True)
class BootstrapResult:
    """Deterministic bootstrap samples and their pointwise intervals."""

    samples: Mapping[str, np.ndarray]
    intervals: Mapping[str, BootstrapInterval]
    n_replicates: int
    seed: int
    minimum_success_fraction: float


@dataclass(frozen=True)
class FoldEvaluation:
    """Training artifacts and held-out exports for one segment fold."""

    fold: SegmentFold
    status: str
    reason: str
    inv_q_edges: np.ndarray
    accepted_inv_q_range: Tuple[float, float]
    accepted_q_range: Tuple[float, float]
    pivot_q: float
    training_profile: pd.DataFrame
    validation_profile: pd.DataFrame
    reference_fit: Optional["CurveFit"]
    slope_models: Optional["SlopeModels"]
    transfers: Mapping[str, "TransferFit"]
    predictions: pd.DataFrame
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class CurveFit:
    """A quadratic in physical coordinates, in descending-power order."""

    coefficients: np.ndarray
    covariance: np.ndarray
    x_min: float
    x_max: float
    x_unit: str
    y_unit: str

    def evaluate(self, x, allow_extrapolation=False):
        values = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("curve evaluation coordinates must be finite")
        tolerance = 8.0 * max(
            abs(float(np.spacing(self.x_min))),
            abs(float(np.spacing(self.x_max))),
        )
        outside = (values < self.x_min - tolerance) | (
            values > self.x_max + tolerance
        )
        if not allow_extrapolation and np.any(outside):
            raise ValueError(
                "curve evaluation coordinates are outside the training domain"
            )
        result = np.polyval(self.coefficients, values)
        return float(result) if result.ndim == 0 else result

    __call__ = evaluate


@dataclass(frozen=True)
class SlopeModels:
    """Weighted quadratic and Gaussian-kernel local-linear slope models."""

    quadratic_fit: CurveFit
    q_profile: np.ndarray
    slope_profile_ua_per_ns: np.ndarray
    slope_sem_ua_per_ns: np.ndarray
    bandwidth_q: float
    q_min: float
    q_max: float

    def _checked_q(self, q):
        values = np.asarray(q, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("Q evaluation coordinates must be finite")
        tolerance = 64.0 * np.finfo(float).eps * max(
            1.0, abs(self.q_min), abs(self.q_max)
        )
        if np.any(values < self.q_min - tolerance) or np.any(
            values > self.q_max + tolerance
        ):
            raise ValueError("Q evaluation coordinates are outside the training domain")
        return values

    def evaluate_quadratic(self, q):
        values = self._checked_q(q)
        result = self.quadratic_fit.evaluate(values)
        return result

    def evaluate_lowess(self, q):
        """Evaluate the LOWESS-like Gaussian local-linear model."""
        values = self._checked_q(q)
        flat_values = values.reshape(-1)
        relative_q = (
            self.q_profile[None, :] - flat_values[:, None]
        ) / self.bandwidth_q
        with np.errstate(over="ignore", invalid="ignore"):
            log_weights = (
                -0.5 * relative_q**2
                - 2.0 * np.log(self.slope_sem_ua_per_ns)[None, :]
            )
        row_max = np.max(log_weights, axis=1)
        if np.any(~np.isfinite(row_max)):
            raise ValueError(
                "local-linear fit has no finite effective support"
            )
        weights = np.exp(log_weights - row_max[:, None])
        weight_sum = np.sum(weights, axis=1)
        if np.any(~np.isfinite(weight_sum)) or np.any(weight_sum <= 0.0):
            raise ValueError(
                "local-linear fit has no finite effective support"
            )
        weights /= weight_sum[:, None]

        support = np.sum(weights > np.finfo(float).eps, axis=1)
        if np.any(support < 2):
            raise ValueError(
                "local-linear fit requires effective support from at least two points"
            )

        local_q_center = np.sum(weights * relative_q, axis=1)
        centered_q = relative_q - local_q_center[:, None]
        local_variance = np.sum(weights * centered_q**2, axis=1)
        if np.any(~np.isfinite(local_variance)) or np.any(
            local_variance <= np.finfo(float).tiny
        ):
            raise ValueError("local-linear weighted design is rank deficient")

        local_slope_center = np.sum(
            weights * self.slope_profile_ua_per_ns[None, :], axis=1
        )
        centered_slope = (
            self.slope_profile_ua_per_ns[None, :]
            - local_slope_center[:, None]
        )
        local_covariance = np.sum(
            weights * centered_q * centered_slope, axis=1
        )
        local_gradient = local_covariance / local_variance
        result = local_slope_center - local_gradient * local_q_center
        if np.any(~np.isfinite(result)):
            raise ValueError("local-linear fit produced a non-finite prediction")

        reshaped = result.reshape(values.shape)
        return float(reshaped) if reshaped.ndim == 0 else reshaped

    def evaluate(self, q, model="lowess"):
        if model == "lowess":
            return self.evaluate_lowess(q)
        if model == "quadratic":
            return self.evaluate_quadratic(q)
        raise ValueError("model must be 'lowess' or 'quadratic'")


@dataclass(frozen=True)
class TransferFit:
    """Weighted affine transfer from inverse slope (ns/uA) to time (ps)."""

    offset_ps: float
    scale_ps_ua_per_ns: float
    covariance: np.ndarray
    inv_slope_min_ns_per_ua: float
    inv_slope_max_ns_per_ua: float

    @property
    def scale_ua(self):
        """Compatibility alias for the historical, dimensionally vague name."""
        return self.scale_ps_ua_per_ns

    def evaluate(self, inverse_slope, allow_extrapolation=False):
        values = np.asarray(inverse_slope, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("inverse-slope evaluation values must be finite")
        tolerance = 8.0 * max(
            abs(float(np.spacing(self.inv_slope_min_ns_per_ua))),
            abs(float(np.spacing(self.inv_slope_max_ns_per_ua))),
        )
        outside = (
            values < self.inv_slope_min_ns_per_ua - tolerance
        ) | (values > self.inv_slope_max_ns_per_ua + tolerance)
        if not allow_extrapolation and np.any(outside):
            raise ValueError(
                "inverse-slope evaluation values are outside the training domain"
            )
        result = self.offset_ps + self.scale_ps_ua_per_ns * values
        return float(result) if result.ndim == 0 else result

    __call__ = evaluate


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("{} must be a positive integer".format(name))
    if value <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _profile_edges(inv_q, n_bins, inv_q_edges):
    if inv_q_edges is not None:
        edges = np.asarray(inv_q_edges, dtype=float)
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(~np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
        ):
            raise ValueError(
                "inv_q_edges must be a finite, strictly increasing one-dimensional array"
            )
        return edges

    if inv_q.size == 0:
        return None
    q1, q3 = np.quantile(inv_q, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr > 0.0 and np.isfinite(iqr):
        population = inv_q[
            (inv_q >= q1 - 1.5 * iqr) & (inv_q <= q3 + 1.5 * iqr)
        ]
        lower = float(np.min(population))
        upper = float(np.max(population))
    else:
        lower = float(np.min(inv_q))
        upper = float(np.max(inv_q))
    if not lower < upper:
        raise ValueError("default inverse-Q window requires a nonzero populated range")
    return np.linspace(lower, upper, n_bins + 1)


def _mad_timing_mask(values, mad_nsigma):
    median = float(np.median(values))
    deviation = np.abs(values - median)
    mad = float(np.median(deviation))
    if mad > 0.0:
        return deviation <= mad_nsigma * 1.4826 * mad
    scale = max(1.0, abs(median), float(np.max(np.abs(values))))
    tolerance = 32.0 * np.finfo(float).eps * scale
    return deviation <= tolerance


def _sample_statistics(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size == 1 or np.all(values == values[0]):
        std = 0.0
    else:
        std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(values.size)
    return mean, std, float(sem)


def _clamp_to_edge_ulps(values, edges, ulps=4.0):
    original = np.asarray(values, dtype=float)
    clamped = original.copy()
    best_distance = np.full(original.shape, np.inf, dtype=float)
    for edge in np.asarray(edges, dtype=float):
        distance = np.abs(original - edge)
        tolerance = ulps * abs(float(np.spacing(edge)))
        snap = (distance <= tolerance) & (distance <= best_distance)
        clamped[snap] = edge
        best_distance[snap] = distance[snap]
    return clamped


def build_comparison_profile(
    frame,
    n_bins=12,
    min_entries=5,
    mad_nsigma=5.0,
    inv_q_edges=None,
):
    """Build matched inverse-Q profiles with per-bin timing MAD clipping.

    Bins are ``[low, high)`` except that the final bin includes its high edge.
    The timing-derived clip mask is applied to every exported observable.
    """
    n_bins = _positive_integer(n_bins, "n_bins")
    min_entries = _positive_integer(min_entries, "min_entries")
    if not np.isfinite(mad_nsigma) or mad_nsigma <= 0.0:
        raise ValueError("mad_nsigma must be finite and positive")

    required = ("energy", "delta_t_ref_ps", "slope_ua_per_ns")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("missing required profile columns: {}".format(missing))

    energy = frame["energy"].to_numpy(dtype=float)
    timing = frame["delta_t_ref_ps"].to_numpy(dtype=float)
    slope = frame["slope_ua_per_ns"].to_numpy(dtype=float)
    valid = (
        np.isfinite(energy)
        & (energy > 0.0)
        & np.isfinite(slope)
        & (slope > 0.0)
        & np.isfinite(timing)
    )
    energy = energy[valid]
    timing = timing[valid]
    slope = slope[valid]
    inv_q = 1.0 / energy
    inv_slope = 1.0 / slope
    edges = _profile_edges(inv_q, n_bins, inv_q_edges)
    if edges is None:
        return pd.DataFrame(columns=PROFILE_COLUMNS)
    if inv_q_edges is not None:
        inv_q = _clamp_to_edge_ulps(inv_q, edges)

    assignments = np.searchsorted(edges, inv_q, side="right") - 1
    assignments[inv_q == edges[-1]] = len(edges) - 2
    in_window = (
        (assignments >= 0)
        & (assignments < len(edges) - 1)
        & (inv_q >= edges[0])
        & (inv_q <= edges[-1])
    )

    records = []
    for bin_index in range(len(edges) - 1):
        positions = np.flatnonzero(in_window & (assignments == bin_index))
        count_before = int(positions.size)
        if count_before == 0:
            continue
        clip_mask = _mad_timing_mask(timing[positions], float(mad_nsigma))
        retained = positions[clip_mask]
        count_after = int(retained.size)
        if count_after < min_entries:
            continue

        inv_q_stats = _sample_statistics(inv_q[retained])
        energy_stats = _sample_statistics(energy[retained])
        timing_stats = _sample_statistics(timing[retained])
        slope_stats = _sample_statistics(slope[retained])
        inv_slope_stats = _sample_statistics(inv_slope[retained])
        records.append(
            {
                "bin_index": bin_index,
                "inv_q_bin_low": float(edges[bin_index]),
                "inv_q_bin_high": float(edges[bin_index + 1]),
                "count_before_clip": count_before,
                "count_after_clip": count_after,
                "inv_q_mean": inv_q_stats[0],
                "inv_q_std": inv_q_stats[1],
                "inv_q_sem": inv_q_stats[2],
                "energy_mean": energy_stats[0],
                "energy_std": energy_stats[1],
                "energy_sem": energy_stats[2],
                "delta_t_ref_mean_ps": timing_stats[0],
                "delta_t_ref_std_ps": timing_stats[1],
                "delta_t_ref_sem_ps": timing_stats[2],
                "slope_mean_ua_per_ns": slope_stats[0],
                "slope_std_ua_per_ns": slope_stats[1],
                "slope_sem_ua_per_ns": slope_stats[2],
                "inv_slope_mean_ns_per_ua": inv_slope_stats[0],
                "inv_slope_std_ns_per_ua": inv_slope_stats[1],
                "inv_slope_sem_ns_per_ua": inv_slope_stats[2],
                "inverse_mean_slope_ns_per_ua": 1.0 / slope_stats[0],
            }
        )

    return pd.DataFrame.from_records(records, columns=PROFILE_COLUMNS)


def _normalized_fit_weights(sem):
    """Return ``1/sem`` weights normalized without overflow."""
    log_weights = -np.log(sem)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    if np.any(~np.isfinite(weights)) or not np.any(weights > 0.0):
        raise ValueError("fit weights are non-finite or numerically zero")
    return weights


def _check_weighted_design(design, fit_weights, parameter_count, context):
    weighted_design = design * fit_weights[:, None]
    try:
        rank = int(np.linalg.matrix_rank(weighted_design))
        condition = float(np.linalg.cond(weighted_design))
    except np.linalg.LinAlgError as error:
        raise ValueError("{} weighted design conditioning failed".format(context)) from error
    if rank < parameter_count:
        raise ValueError("{} weighted design is rank deficient".format(context))
    condition_limit = 1.0 / np.sqrt(np.finfo(float).eps)
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError("{} weighted design is ill-conditioned".format(context))
    return weighted_design


def _fit_quadratic(x, y, sem, x_unit, y_unit, require_positive_x=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sem = np.asarray(sem, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or sem.ndim != 1 or not (
        x.shape == y.shape == sem.shape
    ):
        raise ValueError("x, y, and sem must be one-dimensional arrays of equal length")
    if x.size < 3 or np.unique(x).size < 3:
        raise ValueError("a quadratic fit requires at least three distinct x values")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(
        ~np.isfinite(sem)
    ):
        raise ValueError("x, y, and sem must be finite and sem must be positive")
    if np.any(sem <= 0.0):
        raise ValueError("sem values must be finite and positive")
    if require_positive_x and np.any(x <= 0.0):
        raise ValueError("fit coordinates must be finite and positive")

    fit_weights = _normalized_fit_weights(sem)
    inverse_variance = fit_weights**2
    inverse_variance /= np.sum(inverse_variance)
    x_center = float(np.sum(inverse_variance * x))
    x_scale = float(np.max(np.abs(x - x_center)))
    if not np.isfinite(x_scale) or x_scale <= 0.0:
        raise ValueError("quadratic fit coordinates are rank deficient")
    z = (x - x_center) / x_scale

    y_center = float(np.sum(inverse_variance * y))
    y_scale = float(np.max(np.abs(y - y_center)))
    if not np.isfinite(y_scale):
        raise ValueError("quadratic fit values cannot be scaled")
    if y_scale == 0.0:
        y_scale = 1.0
    scaled_y = (y - y_center) / y_scale

    design = np.column_stack([z**2, z, np.ones_like(z)])
    _check_weighted_design(design, fit_weights, 3, "quadratic fit")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.RankWarning)
            scaled_coefficients, normalized_covariance = np.polyfit(
                z,
                scaled_y,
                2,
                w=fit_weights,
                cov="unscaled",
            )
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        RuntimeWarning,
        np.RankWarning,
    ) as error:
        raise ValueError("quadratic fit numerical solve failed") from error

    z_coefficients = np.asarray(scaled_coefficients, dtype=float) * y_scale
    z_coefficients[-1] += y_center
    transform = np.array(
        [
            [1.0 / x_scale**2, 0.0, 0.0],
            [-2.0 * x_center / x_scale**2, 1.0 / x_scale, 0.0],
            [
                x_center**2 / x_scale**2,
                -x_center / x_scale,
                1.0,
            ],
        ]
    )
    coefficients = transform @ z_coefficients
    minimum_sem = float(np.min(sem))
    sem_over_x_scale = minimum_sem / x_scale
    scaled_transform = np.array(
        [
            [sem_over_x_scale / x_scale, 0.0, 0.0],
            [
                -2.0 * (x_center / x_scale) * sem_over_x_scale,
                sem_over_x_scale,
                0.0,
            ],
            [
                (x_center / x_scale) ** 2 * minimum_sem,
                -(x_center / x_scale) * minimum_sem,
                minimum_sem,
            ],
        ]
    )
    covariance = (
        scaled_transform
        @ np.asarray(normalized_covariance, dtype=float)
        @ scaled_transform.T
    )
    if np.any(~np.isfinite(coefficients)):
        raise ValueError("quadratic fit produced non-finite coefficients")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("quadratic fit produced non-finite covariance")
    return CurveFit(
        coefficients=np.asarray(coefficients, dtype=float),
        covariance=np.asarray(covariance, dtype=float),
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        x_unit=x_unit,
        y_unit=y_unit,
    )


def fit_reference_quadratic(profile):
    """Fit reference time versus inverse Q using ``np.polyfit(w=1/SEM)``."""
    return _fit_quadratic(
        profile["inv_q_mean"].to_numpy(dtype=float),
        profile["delta_t_ref_mean_ps"].to_numpy(dtype=float),
        profile["delta_t_ref_sem_ps"].to_numpy(dtype=float),
        x_unit="1/Q",
        y_unit="ps",
    )


def fit_slope_quadratic(profile):
    """Fit slope versus Q; coefficients are ``[Q^2, Q, constant]``."""
    slope = profile["slope_mean_ua_per_ns"].to_numpy(dtype=float)
    if np.any(~np.isfinite(slope)) or np.any(slope <= 0.0):
        raise ValueError("profile slope means must be finite and positive")
    return _fit_quadratic(
        profile["energy_mean"].to_numpy(dtype=float),
        slope,
        profile["slope_sem_ua_per_ns"].to_numpy(dtype=float),
        x_unit="Q",
        y_unit="uA/ns",
    )


def fit_slope_models(profile, bandwidth_q=20.0, positivity_grid_size=1001):
    """Fit and densely validate weighted quadratic and local-linear models."""
    if not np.isfinite(bandwidth_q) or bandwidth_q <= 0.0:
        raise ValueError("bandwidth_q must be finite and positive")
    positivity_grid_size = _positive_integer(
        positivity_grid_size, "positivity_grid_size"
    )
    if positivity_grid_size < 3:
        raise ValueError("positivity_grid_size must be at least three")

    q = profile["energy_mean"].to_numpy(dtype=float)
    slope = profile["slope_mean_ua_per_ns"].to_numpy(dtype=float)
    sem = profile["slope_sem_ua_per_ns"].to_numpy(dtype=float)
    quadratic = fit_slope_quadratic(profile)
    if np.any(~np.isfinite(sem)) or np.any(sem <= 0.0):
        raise ValueError("slope SEM values must be finite and positive")

    order = np.argsort(q, kind="mergesort")
    models = SlopeModels(
        quadratic_fit=quadratic,
        q_profile=q[order].copy(),
        slope_profile_ua_per_ns=slope[order].copy(),
        slope_sem_ua_per_ns=sem[order].copy(),
        bandwidth_q=float(bandwidth_q),
        q_min=float(np.min(q)),
        q_max=float(np.max(q)),
    )
    dense_q = np.linspace(models.q_min, models.q_max, positivity_grid_size)
    for model_name in ("lowess", "quadratic"):
        dense_slope = np.asarray(models.evaluate(dense_q, model=model_name))
        if np.any(~np.isfinite(dense_slope)) or np.any(dense_slope <= 0.0):
            raise ValueError(
                "{} slope curve must remain finite and positive on the training domain".format(
                    model_name
                )
            )
    return models


def normalize_endpoints(values):
    """Map the first endpoint to one and the last to zero."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("endpoint normalization requires at least two values")
    if np.any(~np.isfinite(values)):
        raise ValueError("cannot normalize a non-finite curve")
    scale = float(values[0] - values[-1])
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(float(values[0])), abs(float(values[-1]))
    )
    if abs(scale) <= tolerance:
        raise ValueError("cannot normalize a flat curve")
    return (values - values[-1]) / scale


def fit_affine_transfer(inverse_slope, reference_time_ps, reference_sem_ps):
    """Fit ``reference_time = A + K * inverse_slope`` with SEM weights."""
    inverse_slope = np.asarray(inverse_slope, dtype=float)
    reference_time_ps = np.asarray(reference_time_ps, dtype=float)
    reference_sem_ps = np.asarray(reference_sem_ps, dtype=float)
    if (
        inverse_slope.ndim != 1
        or reference_time_ps.ndim != 1
        or reference_sem_ps.ndim != 1
        or not (
            inverse_slope.shape
            == reference_time_ps.shape
            == reference_sem_ps.shape
        )
    ):
        raise ValueError("transfer inputs must be one-dimensional arrays of equal length")
    if inverse_slope.size < 2 or np.unique(inverse_slope).size < 2:
        raise ValueError("affine transfer requires at least two distinct inverse slopes")
    if (
        np.any(~np.isfinite(inverse_slope))
        or np.any(inverse_slope <= 0.0)
        or np.any(~np.isfinite(reference_time_ps))
        or np.any(~np.isfinite(reference_sem_ps))
    ):
        raise ValueError("transfer values must be finite and inverse slopes must be positive")
    if np.any(reference_sem_ps <= 0.0):
        raise ValueError("transfer SEM values must be finite and positive")

    fit_weights = _normalized_fit_weights(reference_sem_ps)
    inverse_variance = fit_weights**2
    inverse_variance /= np.sum(inverse_variance)
    u_center = float(np.sum(inverse_variance * inverse_slope))
    u_scale = float(np.max(np.abs(inverse_slope - u_center)))
    if not np.isfinite(u_scale) or u_scale <= 0.0:
        raise ValueError("affine transfer weighted design is rank deficient")
    z = (inverse_slope - u_center) / u_scale

    y_center = float(np.sum(inverse_variance * reference_time_ps))
    y_scale = float(np.max(np.abs(reference_time_ps - y_center)))
    if not np.isfinite(y_scale):
        raise ValueError("affine transfer values cannot be scaled")
    if y_scale == 0.0:
        y_scale = 1.0
    scaled_y = (reference_time_ps - y_center) / y_scale

    design = np.column_stack([np.ones_like(z), z])
    weighted_design = _check_weighted_design(
        design, fit_weights, 2, "affine transfer"
    )
    try:
        information = weighted_design.T @ weighted_design
        normalized_covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError as error:
        raise ValueError("affine transfer numerical solve failed") from error

    scaled_y_mean = float(np.sum(inverse_variance * scaled_y))
    z_mean = float(np.sum(inverse_variance * z))
    centered_z = z - z_mean
    denominator = float(np.sum(inverse_variance * centered_z**2))
    if not np.isfinite(denominator) or denominator <= np.finfo(float).tiny:
        raise ValueError("affine transfer weighted design is rank deficient")
    scaled_slope = float(
        np.sum(
            inverse_variance
            * centered_z
            * (scaled_y - scaled_y_mean)
        )
        / denominator
    )
    scaled_intercept = scaled_y_mean - scaled_slope * z_mean

    center_intercept = y_center + y_scale * scaled_intercept
    center_slope = y_scale * scaled_slope
    scale = float(center_slope / u_scale)
    offset = float(center_intercept - scale * u_center)
    minimum_sem = float(np.min(reference_sem_ps))
    sem_over_u_scale = minimum_sem / u_scale
    scaled_transform = np.array(
        [
            [
                minimum_sem,
                -(u_center / u_scale) * minimum_sem,
            ],
            [0.0, sem_over_u_scale],
        ]
    )
    covariance = scaled_transform @ normalized_covariance @ scaled_transform.T
    if np.any(~np.isfinite([offset, scale])) or np.any(
        ~np.isfinite(covariance)
    ):
        raise ValueError("affine transfer produced non-finite fit parameters")
    return TransferFit(
        offset_ps=offset,
        scale_ps_ua_per_ns=scale,
        covariance=covariance,
        inv_slope_min_ns_per_ua=float(np.min(inverse_slope)),
        inv_slope_max_ns_per_ua=float(np.max(inverse_slope)),
    )


def evaluate_slope_correction(
    slope_models, transfer_fit, q, model="lowess", pivot_q=None
):
    """Evaluate ``K * (1/S(Q) - 1/S(Q_pivot))`` on the training domain.

    If omitted, ``pivot_q`` is the high-Q edge learned from the training
    profile, never from evaluation data.
    """
    if pivot_q is None:
        pivot_q = slope_models.q_max
    pivot_q = float(pivot_q)
    slope = np.asarray(slope_models.evaluate(q, model=model), dtype=float)
    pivot_slope = float(slope_models.evaluate(pivot_q, model=model))
    transfer_parameters = np.array(
        [transfer_fit.offset_ps, transfer_fit.scale_ps_ua_per_ns], dtype=float
    )
    if np.any(~np.isfinite(transfer_parameters)):
        raise ValueError("slope correction requires finite transfer parameters")
    if (
        np.any(~np.isfinite(slope))
        or not np.isfinite(pivot_slope)
        or np.any(slope <= 0.0)
        or pivot_slope <= 0.0
    ):
        raise ValueError("slope correction requires finite positive model values")
    correction = transfer_fit.scale_ps_ua_per_ns * (
        1.0 / slope - 1.0 / pivot_slope
    )
    if np.any(~np.isfinite(correction)):
        raise ValueError("slope correction produced non-finite values")
    return correction


def build_segment_folds(frame):
    """Build deterministic leave-one-segment-out folds from dataframe labels."""
    missing = [column for column in ("file_id", "segment") if column not in frame]
    if missing:
        raise ValueError("missing fold columns: {}".format(missing))
    if not frame.index.is_unique:
        raise ValueError("segment folds require a unique dataframe index")
    if frame[["file_id", "segment"]].isna().any(axis=None):
        raise ValueError("file_id and segment must not contain missing values")

    group_values = frame[["file_id", "segment"]].drop_duplicates()
    folds = []
    for file_id, segment in group_values.itertuples(index=False, name=None):
        validation_mask = (
            frame["file_id"].eq(file_id) & frame["segment"].eq(segment)
        ).to_numpy(dtype=bool)
        validation_indices = frame.index[validation_mask].copy()
        train_indices = frame.index[~validation_mask].copy()
        folds.append(
            SegmentFold(
                fold_id="{}:{}".format(file_id, segment),
                held_out_group=(file_id, segment),
                train_indices=train_indices,
                validation_indices=validation_indices,
            )
        )
    return folds


def build_cross_file_segment_folds(frame):
    """Hold out each segment ordinal across files while keeping groups intact.

    Beam-energy settings are stored in separate files. Grouping the same scope
    segment ordinal across files gives every validation fold charge coverage
    while no individual ``(file_id, segment)`` group is split.
    """
    missing = [column for column in ("file_id", "segment") if column not in frame]
    if missing:
        raise ValueError("missing fold columns: {}".format(missing))
    if not frame.index.is_unique:
        raise ValueError("segment folds require a unique dataframe index")
    if frame[["file_id", "segment"]].isna().any(axis=None):
        raise ValueError("file_id and segment must not contain missing values")

    folds = []
    for segment in sorted(frame["segment"].unique()):
        validation_mask = frame["segment"].eq(segment).to_numpy(dtype=bool)
        validation_indices = frame.index[validation_mask].copy()
        train_indices = frame.index[~validation_mask].copy()
        groups = tuple(
            map(
                tuple,
                frame.loc[
                    validation_indices, ["file_id", "segment"]
                ].drop_duplicates().to_numpy(),
            )
        )
        folds.append(
            SegmentFold(
                fold_id="segment:{}".format(segment),
                held_out_group=groups[0],
                train_indices=train_indices,
                validation_indices=validation_indices,
                held_out_groups=groups,
            )
        )
    return folds


def robust_sigma(values):
    """Return the median absolute deviation scaled to Gaussian sigma."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("timing values must be one-dimensional")
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def gaussian_core_width(values, core_nsigma=2.5, min_count=5):
    """Fit a truncated-normal likelihood inside the fixed robust core.

    The uncertainty is obtained from the inverse observed Hessian in
    ``(mean, log(sigma))`` coordinates. Fit failures remain explicit; the
    uncorrected standard deviation of the truncated sample is never returned.
    """
    if not np.isfinite(core_nsigma) or core_nsigma <= 0.0:
        raise ValueError("core_nsigma must be finite and positive")
    min_count = _positive_integer(min_count, "min_count")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("timing values must be one-dimensional")
    finite_values = values[np.isfinite(values)]
    if finite_values.size < min_count:
        return TimingWidth(
            np.nan, np.nan, int(finite_values.size), "insufficient_events"
        )

    median = float(np.median(finite_values))
    scale = robust_sigma(finite_values)
    if not np.isfinite(scale) or scale <= 0.0:
        return TimingWidth(
            np.nan, np.nan, int(finite_values.size), "zero_robust_sigma"
        )
    core = finite_values[
        np.abs(finite_values - median) <= float(core_nsigma) * scale
    ]
    if core.size < min_count:
        return TimingWidth(
            np.nan, np.nan, int(core.size), "insufficient_core_events"
        )

    lower = median - float(core_nsigma) * scale
    upper = median + float(core_nsigma) * scale

    def negative_log_likelihood(parameters):
        mean, log_sigma = parameters
        sigma = np.exp(log_sigma)
        standardized = (core - mean) / sigma
        lower_cdf = ndtr((lower - mean) / sigma)
        upper_cdf = ndtr((upper - mean) / sigma)
        normalization = upper_cdf - lower_cdf
        if (
            not np.isfinite(normalization)
            or normalization <= np.finfo(float).tiny
        ):
            return np.inf
        return float(
            core.size * (log_sigma + np.log(normalization))
            + 0.5 * np.sum(standardized**2)
        )

    initial = np.array([median, np.log(scale)], dtype=float)
    bounds = (
        (lower - 5.0 * scale, upper + 5.0 * scale),
        (np.log(scale) - np.log(10.0), np.log(scale) + np.log(10.0)),
    )
    fit = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not fit.success or np.any(~np.isfinite(fit.x)):
        return TimingWidth(np.nan, np.nan, int(core.size), "fit_failed")

    mean, log_sigma = np.asarray(fit.x, dtype=float)
    sigma = float(np.exp(log_sigma))
    steps = np.array(
        [max(1.0e-5 * scale, 1.0e-7), 1.0e-5], dtype=float
    )
    center_value = negative_log_likelihood((mean, log_sigma))
    hessian = np.empty((2, 2), dtype=float)
    for axis in range(2):
        offset = np.zeros(2, dtype=float)
        offset[axis] = steps[axis]
        hessian[axis, axis] = (
            negative_log_likelihood(fit.x + offset)
            - 2.0 * center_value
            + negative_log_likelihood(fit.x - offset)
        ) / steps[axis] ** 2
    offset_a = np.array([steps[0], steps[1]])
    offset_b = np.array([steps[0], -steps[1]])
    hessian[0, 1] = hessian[1, 0] = (
        negative_log_likelihood(fit.x + offset_a)
        - negative_log_likelihood(fit.x + offset_b)
        - negative_log_likelihood(fit.x - offset_b)
        + negative_log_likelihood(fit.x - offset_a)
    ) / (4.0 * steps[0] * steps[1])
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return TimingWidth(np.nan, np.nan, int(core.size), "hessian_failed")
    if (
        not np.isfinite(sigma)
        or sigma <= 0.0
        or np.any(~np.isfinite(covariance))
        or covariance[1, 1] <= 0.0
    ):
        return TimingWidth(np.nan, np.nan, int(core.size), "hessian_failed")
    uncertainty = float(sigma * np.sqrt(covariance[1, 1]))
    if not np.isfinite(uncertainty) or uncertainty <= 0.0:
        return TimingWidth(np.nan, np.nan, int(core.size), "hessian_failed")
    return TimingWidth(sigma, uncertainty, int(core.size), "ok")


def stratified_bootstrap(
    frame,
    statistic,
    n_replicates=1000,
    seed=4399,
    group_columns=("file_id", "segment"),
    min_success_fraction=0.5,
):
    """Bootstrap events within segments and summarize finite replicates.

    Each sample has a unique ``RangeIndex`` and carries its original dataframe
    label in ``bootstrap_source_index``. Pointwise percentiles use finite
    replicates only and require at least ``min_success_fraction`` (default 50%)
    of all requested replicates; otherwise the interval is explicitly invalid.
    """
    n_replicates = _positive_integer(n_replicates, "n_replicates")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")
    if not callable(statistic):
        raise ValueError("statistic must be callable")
    if (
        not np.isfinite(min_success_fraction)
        or min_success_fraction <= 0.0
        or min_success_fraction > 1.0
    ):
        raise ValueError(
            "min_success_fraction must be finite and in the interval (0, 1]"
        )
    group_columns = tuple(group_columns)
    missing = [column for column in group_columns if column not in frame]
    if missing:
        raise ValueError("missing bootstrap group columns: {}".format(missing))
    if frame.empty:
        raise ValueError("cannot bootstrap an empty dataframe")
    source_column = "bootstrap_source_index"
    if source_column in frame:
        raise ValueError("frame already contains bootstrap_source_index")

    bootstrap_frame = frame.copy()
    bootstrap_frame[source_column] = frame.index.to_numpy(copy=True)

    grouped_positions = [
        np.asarray(positions, dtype=int)
        for positions in bootstrap_frame.groupby(
            list(group_columns), sort=False, dropna=False
        ).indices.values()
    ]
    rng = np.random.default_rng(int(seed))
    collected = {}
    expected_shapes = {}
    for _ in range(n_replicates):
        sampled_positions = np.concatenate(
            [
                rng.choice(positions, size=positions.size, replace=True)
                for positions in grouped_positions
            ]
        )
        sample = bootstrap_frame.iloc[sampled_positions].reset_index(drop=True)
        result = statistic(sample)
        if not isinstance(result, Mapping) or not result:
            raise ValueError("bootstrap statistic must return a non-empty mapping")
        if collected and set(result) != set(collected):
            raise ValueError("bootstrap statistic keys changed between replicates")
        for key, value in result.items():
            array = np.asarray(value, dtype=float)
            if key not in collected:
                collected[key] = []
                expected_shapes[key] = array.shape
            if array.shape != expected_shapes[key]:
                raise ValueError(
                    "bootstrap statistic shape changed for {!r}".format(key)
                )
            collected[key].append(array)

    samples = {
        key: np.stack(values, axis=0) for key, values in collected.items()
    }
    intervals = {}
    minimum_count = int(np.ceil(min_success_fraction * n_replicates))
    for key, values in samples.items():
        finite = np.isfinite(values)
        finite_count = np.sum(finite, axis=0)
        sufficient = finite_count >= minimum_count
        finite_values = np.where(finite, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            percentiles = np.nanpercentile(
                finite_values, [16.0, 50.0, 84.0], axis=0
            )
        percentiles = np.where(sufficient, percentiles, np.nan)
        interval_values = []
        for value in percentiles:
            interval_values.append(float(value) if value.ndim == 0 else value)
        if np.ndim(finite_count) == 0:
            exported_count = int(finite_count)
            exported_status = (
                "ok" if bool(sufficient) else "insufficient_finite_replicates"
            )
        else:
            exported_count = finite_count
            exported_status = np.where(
                sufficient, "ok", "insufficient_finite_replicates"
            )
        intervals[key] = BootstrapInterval(
            *interval_values,
            finite_count=exported_count,
            status=exported_status,
        )
    return BootstrapResult(
        samples,
        intervals,
        n_replicates,
        int(seed),
        float(min_success_fraction),
    )


def _valid_event_mask(frame):
    energy = frame["energy"].to_numpy(dtype=float)
    timing = frame["delta_t_ref_ps"].to_numpy(dtype=float)
    slope = frame["slope_ua_per_ns"].to_numpy(dtype=float)
    return (
        np.isfinite(energy)
        & (energy > 0.0)
        & np.isfinite(timing)
        & np.isfinite(slope)
        & (slope > 0.0)
    )


def _validate_segment_fold(frame, fold):
    if not isinstance(fold, SegmentFold):
        raise ValueError("fold must be a SegmentFold")
    if frame[["file_id", "segment"]].isna().any(axis=None):
        raise ValueError("file_id and segment must not contain missing values")
    if any(pd.isna(value) for group in fold.groups for value in group):
        raise ValueError("held_out_group must not contain missing values")
    if not fold.train_indices.is_unique or not fold.validation_indices.is_unique:
        raise ValueError("fold train and validation indices must each be unique")
    if len(fold.train_indices.intersection(fold.validation_indices)):
        raise ValueError("fold training and validation indices must be disjoint")

    partition = fold.train_indices.append(fold.validation_indices)
    missing = frame.index.difference(partition)
    extra = partition.difference(frame.index)
    if len(partition) != len(frame) or len(missing) or len(extra):
        raise ValueError("fold indices must form a complete partition of the frame")

    validation_groups = set(
        map(
            tuple,
            frame.loc[
                fold.validation_indices, ["file_id", "segment"]
            ].to_numpy(),
        )
    )
    held_out_groups = set(fold.groups)
    if validation_groups != held_out_groups:
        raise ValueError("fold validation rows must exactly match held_out_group")
    training_groups = set(
        map(
            tuple,
            frame.loc[
                fold.train_indices, ["file_id", "segment"]
            ].drop_duplicates().to_numpy(),
        )
    )
    if held_out_groups.intersection(training_groups):
        raise ValueError("fold training rows must exclude held_out_group")


def _transfer_domain_mask(transfer, inverse_slope):
    inverse_slope = np.asarray(inverse_slope, dtype=float)
    tolerance = 8.0 * max(
        abs(float(np.spacing(transfer.inv_slope_min_ns_per_ua))),
        abs(float(np.spacing(transfer.inv_slope_max_ns_per_ua))),
    )
    return (
        np.isfinite(inverse_slope)
        & (inverse_slope > 0.0)
        & (
            inverse_slope
            >= transfer.inv_slope_min_ns_per_ua - tolerance
        )
        & (
            inverse_slope
            <= transfer.inv_slope_max_ns_per_ua + tolerance
        )
    )


def _empty_metrics(fold, training_count, validation_count):
    metrics = {key: np.nan for key in REQUIRED_METRIC_KEYS}
    held_out_files = {group[0] for group in fold.groups}
    held_out_segments = {group[1] for group in fold.groups}
    metrics.update(
        {
            "fold_id": fold.fold_id,
            "held_out_file_id": (
                next(iter(held_out_files)) if len(held_out_files) == 1 else "multiple"
            ),
            "held_out_segment": (
                next(iter(held_out_segments))
                if len(held_out_segments) == 1
                else "multiple"
            ),
            "fold_status": "pending",
            "fold_reason": "",
            "training_event_count": int(training_count),
            "validation_event_count": int(validation_count),
            "accepted_validation_event_count": 0,
            "matched_validation_event_count": 0,
            "matched_validation_event_status": "not_evaluated",
            "retained_training_bin_count": 0,
            "retained_validation_bin_count": 0,
            "matched_validation_bin_count": 0,
            "matched_validation_bin_status": "not_evaluated",
            "width_raw_matched_event_count": 0,
            "width_reference_matched_event_count": 0,
            "width_slope_lowess_matched_event_count": 0,
            "width_slope_quadratic_matched_event_count": 0,
        }
    )
    for key in STATUS_TRACKED_METRIC_KEYS:
        metrics["{}_status".format(key)] = "not_evaluated"
    return metrics


def _set_metric(metrics, key, value, status="ok"):
    metrics[key] = value
    metrics["{}_status".format(key)] = status


def _prediction_frame(validation, accepted_q_range=None):
    predictions = pd.DataFrame(index=validation.index)
    predictions["source_index"] = validation.index.to_numpy(copy=True)
    for column in (
        "file_id",
        "segment",
        "energy",
        "delta_t_ref_ps",
        "slope_ua_per_ns",
    ):
        predictions[column] = validation[column].to_numpy(copy=True)
    for column in ("event_id", "channelID"):
        if column in validation:
            predictions[column] = validation[column].to_numpy(copy=True)
    energy = validation["energy"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        predictions["inv_q"] = 1.0 / energy
    valid = _valid_event_mask(validation)
    statuses = np.full(len(validation), "invalid_measurement", dtype=object)
    if accepted_q_range is None:
        statuses[valid] = "not_evaluated"
    else:
        q_min, q_max = accepted_q_range
        in_range = valid & (energy >= q_min) & (energy <= q_max)
        statuses[valid] = "outside_training_range"
        statuses[in_range] = "accepted"
    predictions["prediction_status"] = statuses
    predictions["reference_prediction_status"] = statuses.copy()
    comparison_status = statuses.copy()
    comparison_status[comparison_status == "accepted"] = "not_evaluated"
    predictions["comparison_status"] = comparison_status
    for model in ("lowess", "quadratic"):
        model_status = statuses.copy()
        model_status[model_status == "accepted"] = "not_evaluated"
        predictions[
            "slope_{}_prediction_status".format(model)
        ] = model_status
    for column in (
        "reference_prediction_ps",
        "slope_lowess_prediction_ps",
        "slope_quadratic_prediction_ps",
        "reference_correction_ps",
        "slope_lowess_correction_ps",
        "slope_quadratic_correction_ps",
        "raw_residual_ps",
        "reference_corrected_residual_ps",
        "slope_lowess_corrected_residual_ps",
        "slope_quadratic_corrected_residual_ps",
        "slope_lowess_transfer_residual_ps",
        "slope_quadratic_transfer_residual_ps",
    ):
        predictions[column] = np.nan
    return predictions.reset_index(drop=True)


def _fold_result(
    fold,
    status,
    reason,
    edges,
    accepted_inv_q_range,
    accepted_q_range,
    pivot_q,
    training_profile,
    validation_profile,
    reference_fit,
    slope_models,
    transfers,
    predictions,
    metrics,
):
    metrics["fold_status"] = status
    metrics["fold_reason"] = reason
    return FoldEvaluation(
        fold=fold,
        status=status,
        reason=reason,
        inv_q_edges=np.asarray(edges, dtype=float),
        accepted_inv_q_range=accepted_inv_q_range,
        accepted_q_range=accepted_q_range,
        pivot_q=float(pivot_q),
        training_profile=training_profile,
        validation_profile=validation_profile,
        reference_fit=reference_fit,
        slope_models=slope_models,
        transfers=dict(transfers),
        predictions=predictions,
        metrics=metrics,
    )


def _residual_slope(inv_q, residual):
    inv_q = np.asarray(inv_q, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if inv_q.size < 2 or np.unique(inv_q).size < 2:
        return float("nan")
    return float(np.polyfit(inv_q, residual, 1)[0])


def _shape_metrics(reference, candidate):
    reference_shape = normalize_endpoints(np.asarray(reference, dtype=float))
    candidate_shape = normalize_endpoints(np.asarray(candidate, dtype=float))
    residual = candidate_shape - reference_shape
    return float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual)))


def _record_width_metrics(metrics, name, values):
    centered = np.asarray(values, dtype=float)
    centered = centered[np.isfinite(centered)]
    if centered.size:
        centered = centered - np.median(centered)
    robust_key = "robust_width_{}_ps".format(name)
    robust_width = robust_sigma(centered)
    _set_metric(
        metrics,
        robust_key,
        robust_width,
        "ok" if np.isfinite(robust_width) else "insufficient_events",
    )
    gaussian = gaussian_core_width(centered)
    _set_metric(
        metrics,
        "gaussian_width_{}_ps".format(name),
        gaussian.sigma_ps,
        gaussian.status,
    )
    _set_metric(
        metrics,
        "gaussian_width_{}_uncertainty_ps".format(name),
        gaussian.uncertainty_ps,
        gaussian.status,
    )
    _set_metric(
        metrics,
        "gaussian_width_{}_count".format(name),
        gaussian.count,
        "ok",
    )
    metrics["gaussian_width_{}_status".format(name)] = gaussian.status


def evaluate_segment_fold(
    frame,
    fold,
    n_bins=12,
    min_entries=5,
    mad_nsigma=5.0,
    bandwidth_q=20.0,
    min_validation_events=5,
    min_validation_bins=2,
):
    """Fit training rows and export held-out predictions and metrics.

    Endpoint-normalized shape and slope metrics require at least two retained
    held-out profile bins. ``min_validation_bins`` may raise that threshold,
    but cannot lower it below two.
    """
    required = (
        "file_id",
        "segment",
        "energy",
        "delta_t_ref_ps",
        "slope_ua_per_ns",
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("missing fold evaluation columns: {}".format(missing))
    if not frame.index.is_unique:
        raise ValueError("fold evaluation requires a unique dataframe index")
    n_bins = _positive_integer(n_bins, "n_bins")
    min_entries = _positive_integer(min_entries, "min_entries")
    if not np.isfinite(mad_nsigma) or mad_nsigma <= 0.0:
        raise ValueError("mad_nsigma must be finite and positive")
    if not np.isfinite(bandwidth_q) or bandwidth_q <= 0.0:
        raise ValueError("bandwidth_q must be finite and positive")
    min_validation_events = _positive_integer(
        min_validation_events, "min_validation_events"
    )
    min_validation_bins = _positive_integer(
        min_validation_bins, "min_validation_bins"
    )
    if min_validation_bins < 2:
        raise ValueError("min_validation_bins must be at least two")
    _validate_segment_fold(frame, fold)

    training = frame.loc[fold.train_indices]
    validation = frame.loc[fold.validation_indices]
    metrics = _empty_metrics(fold, len(training), len(validation))
    empty_profile = pd.DataFrame(columns=PROFILE_COLUMNS)
    predictions = _prediction_frame(validation)
    valid_training = _valid_event_mask(training)
    training_inv_q = 1.0 / training.loc[
        valid_training, "energy"
    ].to_numpy(dtype=float)
    try:
        edges = _profile_edges(training_inv_q, n_bins, None)
    except ValueError:
        return _fold_result(
            fold, "skipped", "insufficient_training_bins", [], (np.nan, np.nan),
            (np.nan, np.nan), np.nan, empty_profile, empty_profile, None, None,
            {}, predictions, metrics,
        )
    if edges is None:
        return _fold_result(
            fold, "skipped", "insufficient_training_bins", [], (np.nan, np.nan),
            (np.nan, np.nan), np.nan, empty_profile, empty_profile, None, None,
            {}, predictions, metrics,
        )

    training_profile = build_comparison_profile(
        training,
        min_entries=min_entries,
        mad_nsigma=mad_nsigma,
        inv_q_edges=edges,
    )
    metrics["retained_training_bin_count"] = int(len(training_profile))
    if len(training_profile) < 3:
        return _fold_result(
            fold, "skipped", "insufficient_training_bins", edges, (np.nan, np.nan),
            (np.nan, np.nan), np.nan, training_profile, empty_profile, None, None,
            {}, predictions, metrics,
        )

    try:
        reference_fit = fit_reference_quadratic(training_profile)
        slope_models = fit_slope_models(training_profile, bandwidth_q=bandwidth_q)
        accepted_q_min = max(slope_models.q_min, 1.0 / reference_fit.x_max)
        accepted_q_max = min(slope_models.q_max, 1.0 / reference_fit.x_min)
        if not accepted_q_min < accepted_q_max:
            raise ValueError("reference and slope training domains do not overlap")
        accepted_q_range = (float(accepted_q_min), float(accepted_q_max))
        accepted_inv_q_range = (1.0 / accepted_q_max, 1.0 / accepted_q_min)
        pivot_q = float(accepted_q_max)

        training_q = training_profile["energy_mean"].to_numpy(dtype=float)
        training_y = training_profile["delta_t_ref_mean_ps"].to_numpy(dtype=float)
        training_sem = training_profile["delta_t_ref_sem_ps"].to_numpy(dtype=float)
        transfers = {
            model: fit_affine_transfer(
                1.0 / slope_models.evaluate(training_q, model=model),
                training_y,
                training_sem,
            )
            for model in ("lowess", "quadratic")
        }
    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ) as error:
        return _fold_result(
            fold, "failed", "model_fit_failed: {}".format(error), edges,
            (np.nan, np.nan), (np.nan, np.nan), np.nan, training_profile,
            empty_profile, None, None, {}, predictions, metrics,
        )

    metrics.update(
        {
            "accepted_inv_q_min": accepted_inv_q_range[0],
            "accepted_inv_q_max": accepted_inv_q_range[1],
            "accepted_q_min": accepted_q_range[0],
            "accepted_q_max": accepted_q_range[1],
            "pivot_q": pivot_q,
        }
    )
    predictions = _prediction_frame(validation, accepted_q_range)
    accepted_mask = predictions["prediction_status"].eq("accepted").to_numpy()
    accepted_count = int(np.sum(accepted_mask))
    metrics["accepted_validation_event_count"] = accepted_count
    validation_profile = empty_profile

    accepted = predictions.loc[accepted_mask]
    q = accepted["energy"].to_numpy(dtype=float)
    inv_q = accepted["inv_q"].to_numpy(dtype=float)
    raw = accepted["delta_t_ref_ps"].to_numpy(dtype=float)
    if accepted_count:
        try:
            reference_prediction = np.asarray(
                reference_fit.evaluate(inv_q), dtype=float
            )
            reference_pivot = float(reference_fit.evaluate(1.0 / pivot_q))
            reference_correction = reference_prediction - reference_pivot
            if np.any(~np.isfinite(reference_prediction)):
                raise FloatingPointError(
                    "reference prediction produced non-finite values"
                )
        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            OverflowError,
        ) as error:
            predictions.loc[
                accepted.index, "reference_prediction_status"
            ] = "numerical_failure"
            return _fold_result(
                fold, "failed",
                "reference_prediction_failed: {}".format(error), edges,
                accepted_inv_q_range, accepted_q_range, pivot_q,
                training_profile, validation_profile, reference_fit,
                slope_models, transfers, predictions, metrics,
            )

        predictions.loc[
            accepted.index, "reference_prediction_status"
        ] = "accepted"
        predictions.loc[
            accepted.index, "reference_prediction_ps"
        ] = reference_prediction
        predictions.loc[
            accepted.index, "reference_correction_ps"
        ] = reference_correction
        predictions.loc[accepted.index, "raw_residual_ps"] = raw
        predictions.loc[
            accepted.index, "reference_corrected_residual_ps"
        ] = raw - reference_correction

        for model in ("lowess", "quadratic"):
            status_column = "slope_{}_prediction_status".format(model)
            try:
                slope = np.asarray(
                    slope_models.evaluate(q, model=model), dtype=float
                )
                if np.any(~np.isfinite(slope)) or np.any(slope <= 0.0):
                    raise FloatingPointError(
                        "{} slope prediction is non-finite or non-positive".format(
                            model
                        )
                    )
                inverse_slope = 1.0 / slope
                slope_domain = (
                    (q >= slope_models.q_min)
                    & (q <= slope_models.q_max)
                )
                transfer_domain = _transfer_domain_mask(
                    transfers[model], inverse_slope
                )
                valid_model = slope_domain & transfer_domain
                model_status = np.full(
                    accepted_count, "outside_slope_model_domain", dtype=object
                )
                model_status[slope_domain] = "outside_transfer_domain"
                model_status[valid_model] = "accepted"
                predictions.loc[
                    accepted.index, status_column
                ] = model_status
                if not np.any(valid_model):
                    continue

                valid_rows = accepted.index[valid_model]
                valid_q = q[valid_model]
                valid_inverse_slope = inverse_slope[valid_model]
                absolute_prediction = np.asarray(
                    transfers[model].evaluate(valid_inverse_slope), dtype=float
                )
                correction = evaluate_slope_correction(
                    slope_models,
                    transfers[model],
                    valid_q,
                    model=model,
                    pivot_q=pivot_q,
                )
                if (
                    np.any(~np.isfinite(absolute_prediction))
                    or np.any(~np.isfinite(correction))
                ):
                    raise FloatingPointError(
                        "{} transferred prediction is non-finite".format(model)
                    )
            except (
                ValueError,
                FloatingPointError,
                np.linalg.LinAlgError,
                OverflowError,
            ) as error:
                not_excluded = predictions.loc[
                    accepted.index, status_column
                ].eq("not_evaluated") | predictions.loc[
                    accepted.index, status_column
                ].eq("accepted")
                predictions.loc[
                    accepted.index[not_excluded.to_numpy()], status_column
                ] = "numerical_failure"
                return _fold_result(
                    fold, "failed",
                    "{}_prediction_failed: {}".format(model, error), edges,
                    accepted_inv_q_range, accepted_q_range, pivot_q,
                    training_profile, validation_profile, reference_fit,
                    slope_models, transfers, predictions, metrics,
                )

            predictions.loc[
                valid_rows, "slope_{}_prediction_ps".format(model)
            ] = absolute_prediction
            predictions.loc[
                valid_rows, "slope_{}_correction_ps".format(model)
            ] = correction
            predictions.loc[
                valid_rows,
                "slope_{}_corrected_residual_ps".format(model),
            ] = raw[valid_model] - correction
            predictions.loc[
                valid_rows, "slope_{}_transfer_residual_ps".format(model)
            ] = absolute_prediction - raw[valid_model]

    model_common = (
        predictions["slope_lowess_prediction_status"].eq("accepted")
        & predictions["slope_quadratic_prediction_status"].eq("accepted")
        & predictions["reference_prediction_status"].eq("accepted")
    )
    generally_accepted = predictions["prediction_status"].eq("accepted")
    predictions.loc[
        generally_accepted, "comparison_status"
    ] = "excluded_by_model_domain"
    predictions.loc[model_common, "comparison_status"] = "accepted"
    matched_event_count = int(model_common.sum())
    metrics["matched_validation_event_count"] = matched_event_count
    metrics["matched_validation_event_status"] = (
        "ok"
        if matched_event_count >= min_validation_events
        else "insufficient_events"
    )
    validation_profile = build_comparison_profile(
        validation.iloc[np.flatnonzero(model_common.to_numpy())],
        min_entries=min_entries,
        mad_nsigma=mad_nsigma,
        inv_q_edges=edges,
    )
    metrics["retained_validation_bin_count"] = int(len(validation_profile))

    if accepted_count < min_validation_events:
        return _fold_result(
            fold, "skipped", "insufficient_validation_events", edges,
            accepted_inv_q_range, accepted_q_range, pivot_q, training_profile,
            validation_profile, reference_fit, slope_models, transfers,
            predictions, metrics,
        )
    if matched_event_count < min_validation_events:
        return _fold_result(
            fold, "skipped", "insufficient_matched_validation_events", edges,
            accepted_inv_q_range, accepted_q_range, pivot_q, training_profile,
            validation_profile, reference_fit, slope_models, transfers,
            predictions, metrics,
        )
    if len(validation_profile) < min_validation_bins:
        metrics["matched_validation_bin_count"] = int(len(validation_profile))
        metrics["matched_validation_bin_status"] = "insufficient_bins"
        return _fold_result(
            fold, "skipped", "insufficient_validation_bins", edges,
            accepted_inv_q_range, accepted_q_range, pivot_q, training_profile,
            validation_profile, reference_fit, slope_models, transfers,
            predictions, metrics,
        )

    profile_q = validation_profile["energy_mean"].to_numpy(dtype=float)
    profile_common = np.ones(len(validation_profile), dtype=bool)
    try:
        for model in ("lowess", "quadratic"):
            predicted_slope = np.asarray(
                slope_models.evaluate(profile_q, model=model), dtype=float
            )
            if (
                np.any(~np.isfinite(predicted_slope))
                or np.any(predicted_slope <= 0.0)
            ):
                raise FloatingPointError(
                    "{} profile slope prediction is invalid".format(model)
                )
            inverse_slope = 1.0 / predicted_slope
            profile_common &= _transfer_domain_mask(
                transfers[model], inverse_slope
            )
    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ) as error:
        return _fold_result(
            fold, "failed", "profile_prediction_failed: {}".format(error), edges,
            accepted_inv_q_range, accepted_q_range, pivot_q, training_profile,
            validation_profile, reference_fit, slope_models, transfers,
            predictions, metrics,
        )

    validation_profile = validation_profile.loc[profile_common].reset_index(drop=True)
    metrics["matched_validation_bin_count"] = int(len(validation_profile))
    metrics["matched_validation_bin_status"] = (
        "ok"
        if len(validation_profile) >= min_validation_bins
        else "insufficient_bins"
    )
    if len(validation_profile) < min_validation_bins:
        return _fold_result(
            fold, "skipped", "insufficient_matched_validation_bins", edges,
            accepted_inv_q_range, accepted_q_range, pivot_q, training_profile,
            validation_profile, reference_fit, slope_models, transfers,
            predictions, metrics,
        )

    shape_q = validation_profile["energy_mean"].to_numpy(dtype=float)
    observed_timing = validation_profile[
        "delta_t_ref_mean_ps"
    ].to_numpy(dtype=float)
    observed_slope = validation_profile[
        "slope_mean_ua_per_ns"
    ].to_numpy(dtype=float)
    sem = validation_profile["delta_t_ref_sem_ps"].to_numpy(dtype=float)
    chi_valid = np.isfinite(sem) & (sem > 0.0)
    for model in ("lowess", "quadratic"):
        try:
            predicted_slope = np.asarray(
                slope_models.evaluate(shape_q, model=model), dtype=float
            )
            proxy_curve = 1.0 / predicted_slope
            try:
                rmse, maximum = _shape_metrics(observed_timing, proxy_curve)
                shape_status = "ok"
            except ValueError:
                rmse, maximum = np.nan, np.nan
                shape_status = "invalid_shape"
            _set_metric(
                metrics, "shape_rmse_{}".format(model), rmse, shape_status
            )
            _set_metric(
                metrics,
                "shape_max_deviation_{}".format(model),
                maximum,
                shape_status,
            )
            _set_metric(
                metrics,
                "slope_validation_rmse_{}_ua_per_ns".format(model),
                np.sqrt(np.mean((predicted_slope - observed_slope) ** 2)),
            )
            transferred = np.asarray(
                transfers[model].evaluate(1.0 / predicted_slope), dtype=float
            )
            correction_residual = transferred - observed_timing
            _set_metric(
                metrics,
                "correction_rmse_{}_ps".format(model),
                np.sqrt(np.mean(correction_residual**2)),
            )
            _set_metric(
                metrics,
                "correction_mad_{}_ps".format(model),
                np.mean(np.abs(correction_residual)),
            )
            if np.any(chi_valid):
                chi2 = float(
                    np.sum((correction_residual[chi_valid] / sem[chi_valid]) ** 2)
                )
                dof = int(np.sum(chi_valid))
                _set_metric(metrics, "weighted_chi2_{}".format(model), chi2)
                _set_metric(
                    metrics, "weighted_chi2_dof_{}".format(model), dof
                )
                _set_metric(
                    metrics,
                    "weighted_chi2_per_dof_{}".format(model),
                    chi2 / dof,
                )
            else:
                for metric_name in (
                    "weighted_chi2",
                    "weighted_chi2_dof",
                    "weighted_chi2_per_dof",
                ):
                    _set_metric(
                        metrics,
                        "{}_{}".format(metric_name, model),
                        np.nan,
                        "invalid_profile_sem",
                    )
        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            OverflowError,
        ) as error:
            return _fold_result(
                fold, "failed",
                "{}_profile_prediction_failed: {}".format(model, error), edges,
                accepted_inv_q_range, accepted_q_range, pivot_q,
                training_profile, validation_profile, reference_fit,
                slope_models, transfers, predictions, metrics,
            )

    matched = predictions.loc[model_common]
    matched_inv_q = matched["inv_q"].to_numpy(dtype=float)
    residual_columns = {
        "raw": "raw_residual_ps",
        "reference": "reference_corrected_residual_ps",
        "slope_lowess": "slope_lowess_corrected_residual_ps",
        "slope_quadratic": "slope_quadratic_corrected_residual_ps",
    }
    for width_name, column in residual_columns.items():
        values = matched[column].to_numpy(dtype=float)
        residual_name = width_name.replace("slope_", "")
        residual_value = _residual_slope(matched_inv_q, values)
        _set_metric(
            metrics,
            "residual_slope_{}_ps_per_inv_q".format(residual_name),
            residual_value,
            "ok" if np.isfinite(residual_value) else "insufficient_events",
        )
        _record_width_metrics(metrics, width_name, values)
        metrics[
            "width_{}_matched_event_count".format(width_name)
        ] = matched_event_count

    unexplained = [
        key
        for key in STATUS_TRACKED_METRIC_KEYS
        if metrics["{}_status".format(key)] == "not_evaluated"
    ]
    if unexplained:
        return _fold_result(
            fold, "failed",
            "unexplained_metrics: {}".format(",".join(sorted(unexplained))),
            edges, accepted_inv_q_range, accepted_q_range, pivot_q,
            training_profile, validation_profile, reference_fit, slope_models,
            transfers, predictions, metrics,
        )

    return _fold_result(
        fold, "ok", "", edges, accepted_inv_q_range, accepted_q_range,
        pivot_q, training_profile, validation_profile, reference_fit,
        slope_models, transfers, predictions, metrics,
    )


def evaluate_channel_folds(
    frame,
    channel_column="channelID",
    fold_strategy="leave_one_group_out",
    **kwargs
):
    """Evaluate every segment fold independently for each channel."""
    fold_builders = {
        "leave_one_group_out": build_segment_folds,
        "cross_file_segment": build_cross_file_segment_folds,
    }
    if fold_strategy not in fold_builders:
        raise ValueError(
            "fold_strategy must be 'leave_one_group_out' or 'cross_file_segment'"
        )
    if channel_column in frame:
        channel_frames = frame.groupby(channel_column, sort=True, dropna=False)
    else:
        channel_frames = [(None, frame)]
    results = []
    for channel, channel_frame in channel_frames:
        for fold in fold_builders[fold_strategy](channel_frame):
            result = evaluate_segment_fold(channel_frame, fold, **kwargs)
            result.metrics["channel"] = channel
            result.predictions["channel"] = channel
            results.append(result)
    return results


__all__ = [
    "BootstrapInterval",
    "BootstrapResult",
    "CurveFit",
    "FoldEvaluation",
    "REQUIRED_METRIC_KEYS",
    "SegmentFold",
    "SlopeModels",
    "TimingWidth",
    "TransferFit",
    "build_comparison_profile",
    "build_cross_file_segment_folds",
    "build_segment_folds",
    "evaluate_channel_folds",
    "evaluate_segment_fold",
    "evaluate_slope_correction",
    "fit_affine_transfer",
    "fit_reference_quadratic",
    "fit_slope_models",
    "fit_slope_quadratic",
    "gaussian_core_width",
    "normalize_endpoints",
    "robust_sigma",
    "stratified_bootstrap",
]
