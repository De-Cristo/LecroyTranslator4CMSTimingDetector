import numpy as np
import pandas as pd
import pytest

import Q_Walk.qwalk_t0_models as models_module
from Q_Walk.qwalk_t0_models import (
    REQUIRED_METRIC_KEYS,
    CurveFit,
    BootstrapResult,
    FoldEvaluation,
    SegmentFold,
    SlopeModels,
    TimingWidth,
    TransferFit,
    build_comparison_profile,
    build_cross_file_segment_folds,
    build_segment_folds,
    evaluate_channel_folds,
    evaluate_segment_fold,
    evaluate_slope_correction,
    fit_affine_transfer,
    fit_reference_quadratic,
    fit_slope_models,
    fit_slope_quadratic,
    gaussian_core_width,
    normalize_endpoints,
    robust_sigma,
    stratified_bootstrap,
)


def _frame(inv_q, timing, slope):
    return pd.DataFrame(
        {
            "energy": 1.0 / np.asarray(inv_q, dtype=float),
            "delta_t_ref_ps": np.asarray(timing, dtype=float),
            "slope_ua_per_ns": np.asarray(slope, dtype=float),
        }
    )


def test_profile_uses_inverse_q_bins_and_per_bin_mad_clip():
    inv_q = np.repeat(np.linspace(0.002, 0.005, 12), 10)
    timing = 100.0 + 20_000.0 * inv_q
    timing[5] = 10_000.0
    frame = _frame(inv_q, timing, 10.0 + 500.0 * inv_q)

    profile = build_comparison_profile(
        frame, n_bins=12, min_entries=5, mad_nsigma=5.0
    )

    assert len(profile) == 12
    assert profile["count_before_clip"].sum() == 120
    assert profile["count_after_clip"].sum() == 119
    assert np.all(profile["delta_t_ref_sem_ps"] == 0.0)


def test_default_profile_window_ignores_sparse_inverse_q_tail():
    normal_inv_q = np.repeat(np.linspace(0.0025, 0.0055, 12), 10)
    inv_q = np.concatenate([normal_inv_q, [1.0]])
    frame = _frame(
        inv_q,
        timing=100.0 + 20_000.0 * inv_q,
        slope=10.0 + 500.0 * inv_q,
    )

    profile = build_comparison_profile(frame, n_bins=12, min_entries=5)

    assert len(profile) == 12
    assert profile["inv_q_bin_high"].max() < 0.01
    assert profile["count_before_clip"].sum() == 120


def test_profile_uses_explicit_edges_and_includes_final_edge():
    inv_q = np.array([0.10, 0.15, 0.20, 0.20, 0.25, 0.30])
    frame = _frame(
        inv_q,
        timing=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        slope=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
    )
    edges = np.array([0.10, 0.20, 0.30])

    profile = build_comparison_profile(
        frame, min_entries=1, inv_q_edges=edges
    )

    assert profile["inv_q_bin_low"].tolist() == [0.10, 0.20]
    assert profile["inv_q_bin_high"].tolist() == [0.20, 0.30]
    assert profile["count_before_clip"].tolist() == [2, 4]
    assert profile["count_after_clip"].tolist() == [2, 4]


def test_profile_clamps_reciprocal_round_trips_within_edge_ulps():
    lower = 0.1
    upper = 0.2
    just_below = np.nextafter(lower, -np.inf)
    just_above = np.nextafter(upper, np.inf)
    frame = _frame(
        [just_below, just_above],
        timing=[1.0, 2.0],
        slope=[10.0, 11.0],
    )

    profile = build_comparison_profile(
        frame, inv_q_edges=[lower, upper], min_entries=1
    )

    assert profile["count_before_clip"].tolist() == [2]
    assert profile["inv_q_mean"].iloc[0] == pytest.approx(0.15)


def test_profile_snaps_reciprocal_round_trips_to_interior_edge_upper_bin():
    interior = 0.2
    just_below = np.nextafter(
        np.nextafter(interior, -np.inf), -np.inf
    )
    frame = _frame(
        [
            0.15,
            just_below,
            np.nextafter(interior, np.inf),
            0.25,
        ],
        timing=[1.0, 2.0, 3.0, 4.0],
        slope=[10.0, 11.0, 12.0, 13.0],
    )

    profile = build_comparison_profile(
        frame, inv_q_edges=[0.1, interior, 0.3], min_entries=1
    )

    assert profile["count_before_clip"].tolist() == [1, 3]
    assert profile["inv_q_mean"].iloc[1] == pytest.approx(
        np.mean([interior, interior, 0.25])
    )


def test_profile_filters_nonfinite_and_nonpositive_measurements():
    frame = pd.DataFrame(
        {
            "energy": [10.0, 0.0, -1.0, np.nan, 20.0, 25.0, 30.0],
            "delta_t_ref_ps": [1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0],
            "slope_ua_per_ns": [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, np.nan],
        }
    )

    profile = build_comparison_profile(
        frame, inv_q_edges=[0.09, 0.11], min_entries=1
    )

    assert profile["count_before_clip"].tolist() == [1]
    assert profile["energy_mean"].tolist() == [10.0]


def test_zero_mad_clip_rejects_only_values_away_from_median():
    inv_q = np.full(6, 0.1)
    frame = _frame(
        inv_q,
        timing=np.array([20.0, 20.0, 20.0, 20.0, 20.0, 99.0]),
        slope=np.array([2.0, 3.0, 4.0, 5.0, 6.0, 100.0]),
    )

    profile = build_comparison_profile(
        frame, inv_q_edges=[0.09, 0.11], min_entries=5
    )

    assert profile["count_before_clip"].tolist() == [6]
    assert profile["count_after_clip"].tolist() == [5]
    assert profile["slope_mean_ua_per_ns"].iloc[0] == pytest.approx(4.0)


def test_profile_reports_sample_standard_deviation_sem_and_jensen_diagnostic():
    slopes = np.array([1.0, 2.0, 4.0, 8.0])
    timing = np.array([10.0, 12.0, 14.0, 16.0])
    profile = build_comparison_profile(
        _frame(np.full(4, 0.1), timing, slopes),
        inv_q_edges=[0.09, 0.11],
        min_entries=4,
    )

    row = profile.iloc[0]
    assert row["delta_t_ref_std_ps"] == pytest.approx(np.std(timing, ddof=1))
    assert row["delta_t_ref_sem_ps"] == pytest.approx(
        np.std(timing, ddof=1) / np.sqrt(4)
    )
    assert row["slope_std_ua_per_ns"] == pytest.approx(np.std(slopes, ddof=1))
    assert row["slope_sem_ua_per_ns"] == pytest.approx(
        np.std(slopes, ddof=1) / np.sqrt(4)
    )
    assert row["inv_slope_mean_ns_per_ua"] == pytest.approx(
        np.mean(1.0 / slopes)
    )
    assert row["inverse_mean_slope_ns_per_ua"] == pytest.approx(
        1.0 / np.mean(slopes)
    )
    assert (
        row["inv_slope_mean_ns_per_ua"]
        > row["inverse_mean_slope_ns_per_ua"]
    )


def test_profile_drops_bins_that_are_sparse_after_clipping():
    frame = _frame(
        np.full(5, 0.1),
        timing=np.array([3.0, 3.0, 3.0, 3.0, 30.0]),
        slope=np.arange(1.0, 6.0),
    )

    profile = build_comparison_profile(
        frame, inv_q_edges=[0.09, 0.11], min_entries=5
    )

    assert profile.empty
    assert "count_after_clip" in profile.columns


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bins": 0}, "n_bins"),
        ({"min_entries": 0}, "min_entries"),
        ({"mad_nsigma": 0.0}, "mad_nsigma"),
        ({"inv_q_edges": [0.1, 0.1, 0.2]}, "inv_q_edges"),
        ({"inv_q_edges": [0.1, np.inf]}, "inv_q_edges"),
    ],
)
def test_profile_rejects_invalid_configuration(kwargs, message):
    frame = _frame([0.1], [1.0], [2.0])
    config = {"min_entries": 1}
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_comparison_profile(frame, **config)


def test_weighted_reference_fit_recovers_quadratic_in_inverse_q():
    x = np.linspace(0.002, 0.006, 12)
    y = 30.0 + 200_000.0 * x - 10_000_000.0 * x**2
    profile = pd.DataFrame(
        {
            "inv_q_mean": x,
            "delta_t_ref_mean_ps": y,
            "delta_t_ref_sem_ps": np.full(12, 2.0),
        }
    )

    fit = fit_reference_quadratic(profile)

    assert isinstance(fit, CurveFit)
    assert np.allclose(
        fit.coefficients, [-10_000_000.0, 200_000.0, 30.0]
    )
    assert np.allclose(fit.evaluate(x), y)


def test_reference_fit_uses_inverse_sem_as_polyfit_weights():
    x = np.linspace(0.01, 0.05, 6)
    y = 5.0 + 20.0 * x - 30.0 * x**2
    y[-1] += 2.0
    sem = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 20.0])
    profile = pd.DataFrame(
        {
            "inv_q_mean": x,
            "delta_t_ref_mean_ps": y,
            "delta_t_ref_sem_ps": sem,
        }
    )

    fit = fit_reference_quadratic(profile)

    expected = np.polyfit(x, y, 2, w=1.0 / sem)
    assert np.allclose(fit.coefficients, expected)


def test_curve_fit_rejects_extrapolation_unless_explicitly_allowed():
    x = np.linspace(0.01, 0.05, 6)
    profile = pd.DataFrame(
        {
            "inv_q_mean": x,
            "delta_t_ref_mean_ps": 2.0 + 3.0 * x,
            "delta_t_ref_sem_ps": np.ones(6),
        }
    )
    fit = fit_reference_quadratic(profile)

    with pytest.raises(ValueError, match="training domain"):
        fit.evaluate([0.009])
    assert np.isfinite(fit.evaluate([0.009], allow_extrapolation=True)).all()


def test_slope_quadratic_is_weighted_in_q_and_uses_polyval_order():
    q = np.linspace(100.0, 600.0, 8)
    slope = 7.0 + 0.02 * q + 1.0e-5 * q**2
    slope[-1] += 3.0
    sem = np.array([0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.7, 30.0])
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": slope,
            "slope_sem_ua_per_ns": sem,
        }
    )

    fit = fit_slope_quadratic(profile)

    assert np.allclose(fit.coefficients, np.polyfit(q, slope, 2, w=1.0 / sem))
    assert np.allclose(fit.evaluate(q), np.polyval(fit.coefficients, q))


def test_quadratic_fit_rejects_numerically_rank_deficient_weights():
    profile = pd.DataFrame(
        {
            "inv_q_mean": [0.1, 0.2, 0.3, 0.4],
            "delta_t_ref_mean_ps": [1.0, 2.0, 3.0, 4.0],
            "delta_t_ref_sem_ps": [1.0, 1.0e200, 1.0e200, 1.0e200],
        }
    )

    with pytest.raises(ValueError, match="rank|ill-conditioned"):
        fit_reference_quadratic(profile)


def test_quadratic_covariance_rescales_tiny_finite_sem_without_underflow():
    x = 1.0e-100 * np.arange(1.0, 7.0)
    z = x / 1.0e-100
    profile = pd.DataFrame(
        {
            "inv_q_mean": x,
            "delta_t_ref_mean_ps": 1.0 + 2.0 * z + 3.0 * z**2,
            "delta_t_ref_sem_ps": np.full(x.shape, 1.0e-200),
        }
    )

    fit = fit_reference_quadratic(profile)

    assert np.isfinite(fit.covariance).all()
    assert fit.covariance[0, 0] > 0.0


@pytest.mark.parametrize(
    "profile",
    [
        pd.DataFrame(
            {
                "inv_q_mean": [0.1, 0.2, 0.3],
                "delta_t_ref_mean_ps": [1.0, 2.0, 3.0],
                "delta_t_ref_sem_ps": [1.0, 0.0, 1.0],
            }
        ),
        pd.DataFrame(
            {
                "inv_q_mean": [0.1, 0.2, 0.3],
                "delta_t_ref_mean_ps": [1.0, np.nan, 3.0],
                "delta_t_ref_sem_ps": [1.0, 1.0, 1.0],
            }
        ),
    ],
)
def test_weighted_reference_fit_rejects_nonpositive_or_nonfinite_inputs(profile):
    with pytest.raises(ValueError, match="finite.*positive|positive.*finite"):
        fit_reference_quadratic(profile)


def test_slope_models_expose_lowess_like_and_quadratic_evaluators():
    q = np.linspace(100.0, 600.0, 12)
    slope = 8.0 + 0.02 * q
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": slope,
            "slope_sem_ua_per_ns": np.full(q.shape, 0.2),
        }
    )

    models = fit_slope_models(profile, bandwidth_q=120.0)

    assert isinstance(models, SlopeModels)
    assert np.allclose(models.evaluate_lowess(q), slope, atol=1e-10)
    assert np.allclose(models.evaluate_quadratic(q), slope, atol=1e-10)
    assert np.allclose(models.evaluate(q, model="lowess"), slope, atol=1e-10)
    assert np.allclose(models.evaluate(q, model="quadratic"), slope, atol=1e-10)


def test_local_linear_rejects_underflowed_effective_support():
    profile = pd.DataFrame(
        {
            "energy_mean": [1.0, 2.0, 4.0],
            "slope_mean_ua_per_ns": [2.0, 3.0, 5.0],
            "slope_sem_ua_per_ns": [0.1, 0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="effective support|rank"):
        fit_slope_models(profile, bandwidth_q=0.0258785)


def test_local_linear_uses_vectorized_closed_form_not_pointwise_lstsq(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("local-linear evaluation must not call lstsq per point")

    monkeypatch.setattr(np.linalg, "lstsq", fail_if_called)
    q = np.linspace(100.0, 600.0, 12)
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": 8.0 + 0.02 * q,
            "slope_sem_ua_per_ns": np.full(q.shape, 0.2),
        }
    )

    models = fit_slope_models(profile, bandwidth_q=120.0)

    assert np.isfinite(models.evaluate_lowess(q)).all()


def test_slope_models_reject_curves_nonpositive_between_profile_points():
    q = np.array([1.0, 2.0, 4.0])
    slope = np.array([1.0, 1.0, 100.0])
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": slope,
            "slope_sem_ua_per_ns": np.ones(3),
        }
    )

    with pytest.raises(ValueError, match="positive"):
        fit_slope_models(profile, bandwidth_q=1.0)


def test_slope_models_reject_nonpositive_bandwidth_and_extrapolation():
    profile = pd.DataFrame(
        {
            "energy_mean": [1.0, 2.0, 3.0],
            "slope_mean_ua_per_ns": [2.0, 3.0, 4.0],
            "slope_sem_ua_per_ns": [0.1, 0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="bandwidth_q"):
        fit_slope_models(profile, bandwidth_q=0.0)
    models = fit_slope_models(profile, bandwidth_q=1.0)
    with pytest.raises(ValueError, match="domain"):
        models.evaluate_lowess([0.5])


def test_endpoint_normalization_removes_offset_and_scale():
    values = np.array([4.0, 2.0, 1.0])

    assert np.allclose(normalize_endpoints(values), [1.0, 1.0 / 3.0, 0.0])
    assert np.allclose(
        normalize_endpoints(values), normalize_endpoints(9.0 + 3.0 * values)
    )


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 1.0],
        [1.0, np.nan],
        [1.0, np.inf],
        [1.0],
    ],
)
def test_endpoint_normalization_rejects_flat_nonfinite_or_short_curves(values):
    with pytest.raises(ValueError, match="flat|non-finite|two"):
        normalize_endpoints(values)


def test_affine_transfer_recovers_known_scale():
    u = np.linspace(0.04, 0.10, 12)
    y = -300.0 + 8_000.0 * u

    fit = fit_affine_transfer(u, y, np.full(12, 2.0))

    assert isinstance(fit, TransferFit)
    assert np.allclose(
        [fit.offset_ps, fit.scale_ps_ua_per_ns], [-300.0, 8_000.0]
    )
    assert fit.scale_ua == fit.scale_ps_ua_per_ns
    assert np.allclose(fit.evaluate(u), y)


def test_affine_transfer_uses_sem_weights():
    u = np.linspace(0.04, 0.10, 8)
    y = -300.0 + 8_000.0 * u
    y[-1] += 500.0
    sem = np.ones(8)
    sem[-1] = 1_000.0

    fit = fit_affine_transfer(u, y, sem)
    expected = np.polyfit(u, y, 1, w=1.0 / sem)

    assert np.allclose([fit.scale_ps_ua_per_ns, fit.offset_ps], expected)


def test_affine_transfer_rejects_numerically_rank_deficient_weights():
    with pytest.raises(ValueError, match="rank|ill-conditioned"):
        fit_affine_transfer(
            [0.1, 0.2, 0.3],
            [1.0, 2.0, 3.0],
            [1.0, 1.0e200, 1.0e200],
        )


def test_affine_covariance_rescales_tiny_finite_sem_without_underflow():
    u = 1.0e-200 * np.arange(1.0, 5.0)
    y = -3.0 + 4.0 * (u / 1.0e-200)

    fit = fit_affine_transfer(u, y, np.full(u.shape, 1.0e-200))

    assert np.isfinite(fit.covariance).all()
    assert fit.covariance[1, 1] > 0.0


def test_transfer_fit_rejects_extrapolation_unless_explicitly_allowed():
    u = np.linspace(0.04, 0.10, 8)
    fit = fit_affine_transfer(
        u, -300.0 + 8_000.0 * u, np.full(u.shape, 2.0)
    )

    with pytest.raises(ValueError, match="training domain"):
        fit.evaluate([0.039])
    assert np.isfinite(
        fit.evaluate([0.039], allow_extrapolation=True)
    ).all()


@pytest.mark.parametrize(
    ("u", "y", "sem"),
    [
        ([0.1, 0.1], [1.0, 2.0], [1.0, 1.0]),
        ([0.1, np.nan], [1.0, 2.0], [1.0, 1.0]),
        ([0.1, 0.2], [1.0, 2.0], [1.0, 0.0]),
    ],
)
def test_affine_transfer_rejects_degenerate_or_invalid_inputs(u, y, sem):
    with pytest.raises(ValueError, match="finite|positive|distinct"):
        fit_affine_transfer(u, y, sem)


def test_correction_uses_model_value_at_training_high_q_pivot():
    q = np.linspace(100.0, 500.0, 10)
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": 10.0 + 0.02 * q,
            "slope_sem_ua_per_ns": np.full(q.shape, 0.2),
        }
    )
    models = fit_slope_models(profile, bandwidth_q=100.0)
    transfer = fit_affine_transfer(
        np.linspace(0.04, 0.10, 10),
        -300.0 + 8_000.0 * np.linspace(0.04, 0.10, 10),
        np.full(10, 2.0),
    )

    correction = evaluate_slope_correction(
        models, transfer, [100.0, 500.0], model="quadratic"
    )
    expected = transfer.scale_ps_ua_per_ns * (
        1.0 / models.evaluate_quadratic([100.0, 500.0])
        - 1.0 / models.evaluate_quadratic([500.0])
    )

    assert np.allclose(correction, expected)
    assert correction[-1] == pytest.approx(0.0, abs=1e-12)


def test_correction_rejects_nonfinite_transfer_parameters():
    q = np.array([100.0, 200.0, 300.0])
    profile = pd.DataFrame(
        {
            "energy_mean": q,
            "slope_mean_ua_per_ns": [10.0, 12.0, 14.0],
            "slope_sem_ua_per_ns": [0.2, 0.2, 0.2],
        }
    )
    models = fit_slope_models(profile, bandwidth_q=100.0)
    transfer = TransferFit(
        offset_ps=0.0,
        scale_ps_ua_per_ns=np.nan,
        covariance=np.eye(2),
        inv_slope_min_ns_per_ua=0.05,
        inv_slope_max_ns_per_ua=0.10,
    )

    with pytest.raises(ValueError, match="finite"):
        evaluate_slope_correction(models, transfer, [100.0])


def _segment_frame(groups=("a", "b", "c"), events_per_group=72, seed=1234):
    rng = np.random.default_rng(seed)
    records = []
    for group_number, file_id in enumerate(groups):
        q = np.linspace(120.0, 520.0, events_per_group)
        q += rng.normal(0.0, 0.8, q.size)
        slope = 9.0 + 0.025 * q + rng.normal(0.0, 0.03, q.size)
        timing = -250.0 + 7200.0 / slope + rng.normal(0.0, 3.0, q.size)
        for event_number in range(events_per_group):
            records.append(
                {
                    "file_id": file_id,
                    "segment": group_number % 2,
                    "energy": q[event_number],
                    "slope_ua_per_ns": slope[event_number],
                    "delta_t_ref_ps": timing[event_number],
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.index = pd.Index(
        ["event-{:04d}".format(number) for number in range(len(frame))],
        name="source_row",
    )
    return frame


def _fold_for(frame, held_out_group):
    return next(
        fold
        for fold in build_segment_folds(frame)
        if fold.held_out_group == held_out_group
    )


def _assert_training_artifacts_equal(first, second):
    assert np.array_equal(first.inv_q_edges, second.inv_q_edges)
    assert first.accepted_q_range == second.accepted_q_range
    assert first.accepted_inv_q_range == second.accepted_inv_q_range
    assert first.pivot_q == second.pivot_q
    assert np.array_equal(
        first.reference_fit.coefficients, second.reference_fit.coefficients
    )
    assert np.array_equal(
        first.slope_models.quadratic_fit.coefficients,
        second.slope_models.quadratic_fit.coefficients,
    )
    for model in ("lowess", "quadratic"):
        assert first.transfers[model].offset_ps == second.transfers[model].offset_ps
        assert (
            first.transfers[model].scale_ps_ua_per_ns
            == second.transfers[model].scale_ps_ua_per_ns
        )


def test_validation_segment_never_appears_in_training_for_non_range_index():
    frame = pd.DataFrame(
        {
            "file_id": ["a"] * 4 + ["b"] * 4,
            "segment": [0, 0, 1, 1, 0, 0, 1, 1],
        },
        index=[101, 103, 107, 109, 211, 223, 227, 229],
    )

    folds = build_segment_folds(frame)

    assert len(folds) == 4
    assert all(isinstance(fold, SegmentFold) for fold in folds)
    for fold in folds:
        assert set(fold.train_indices).isdisjoint(fold.validation_indices)
        held_out = set(
            map(
                tuple,
                frame.loc[
                    fold.validation_indices, ["file_id", "segment"]
                ]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        trained = set(
            map(
                tuple,
                frame.loc[fold.train_indices, ["file_id", "segment"]]
                .drop_duplicates()
                .to_numpy(),
            )
        )
        assert held_out == {fold.held_out_group}
        assert held_out.isdisjoint(trained)


def test_cross_file_segment_folds_hold_out_full_energy_scan_by_segment():
    frame = pd.DataFrame(
        {
            "file_id": ["low", "low", "high", "high"],
            "segment": [0, 1, 0, 1],
            "energy": [100.0, 100.0, 400.0, 400.0],
        },
        index=[10, 11, 20, 21],
    )

    folds = build_cross_file_segment_folds(frame)

    assert len(folds) == 2
    for fold in folds:
        validation = frame.loc[fold.validation_indices]
        training = frame.loc[fold.train_indices]
        assert validation["segment"].nunique() == 1
        assert set(validation["file_id"]) == {"low", "high"}
        assert set(validation["segment"]).isdisjoint(training["segment"])
        assert set(fold.held_out_groups) == set(
            map(tuple, validation[["file_id", "segment"]].to_numpy())
        )


def test_fold_predictions_preserve_validation_event_identity():
    frame = _segment_frame(events_per_group=72, seed=59).assign(channelID=137)
    frame["event_id"] = ["event-{}".format(index) for index in frame.index]
    fold = _fold_for(frame, ("c", 0))

    result = evaluate_segment_fold(
        frame, fold, n_bins=6, min_entries=5, bandwidth_q=100.0
    )

    expected = frame.loc[fold.validation_indices, ["event_id", "channelID"]]
    assert result.predictions["event_id"].tolist() == expected["event_id"].tolist()
    assert result.predictions["channelID"].tolist() == expected["channelID"].tolist()
    assert result.predictions["event_id"].notna().all()
    assert not result.predictions.duplicated(["event_id", "channelID"]).any()


def test_evaluate_segment_fold_rejects_malformed_fold_indices_and_groups():
    frame = _segment_frame(events_per_group=12, seed=60)
    valid = _fold_for(frame, ("c", 0))
    train = valid.train_indices
    validation = valid.validation_indices
    swapped_train = train.delete(0).append(validation[:1])
    swapped_validation = validation.delete(0).append(train[:1])
    malformed = [
        (
            SegmentFold(
                "duplicate",
                valid.held_out_group,
                train.append(train[:1]),
                validation,
            ),
            "unique",
        ),
        (
            SegmentFold(
                "overlap",
                valid.held_out_group,
                train.append(validation[:1]),
                validation,
            ),
            "disjoint",
        ),
        (
            SegmentFold(
                "incomplete",
                valid.held_out_group,
                train.delete(0),
                validation,
            ),
            "complete partition",
        ),
        (
            SegmentFold(
                "wrong-group",
                ("missing", 99),
                train,
                validation,
            ),
            "held_out_group",
        ),
        (
            SegmentFold(
                "mixed-groups",
                valid.held_out_group,
                swapped_train,
                swapped_validation,
            ),
            "validation rows.*held_out_group|training.*held_out_group",
        ),
    ]

    for fold, message in malformed:
        with pytest.raises(ValueError, match=message):
            evaluate_segment_fold(frame, fold, n_bins=4, min_entries=2)


def test_evaluate_segment_fold_rejects_non_fold_before_attribute_access():
    frame = _segment_frame(events_per_group=12, seed=67)

    with pytest.raises(ValueError, match="^fold must be a SegmentFold$"):
        evaluate_segment_fold(frame, None, n_bins=4, min_entries=2)


def test_evaluate_segment_fold_rejects_null_group_values_in_frame_or_fold():
    frame = _segment_frame(events_per_group=24, seed=64)
    fold = _fold_for(frame, ("c", 0))
    null_training = frame.copy()
    null_training.loc[fold.train_indices[0], "file_id"] = np.nan

    with pytest.raises(ValueError, match="file_id.*segment.*missing"):
        evaluate_segment_fold(
            null_training, fold, n_bins=4, min_entries=2
        )

    null_held_out = SegmentFold(
        fold_id="null-held-out",
        held_out_group=("c", np.nan),
        train_indices=fold.train_indices,
        validation_indices=fold.validation_indices,
    )
    with pytest.raises(ValueError, match="held_out_group.*missing"):
        evaluate_segment_fold(frame, null_held_out, n_bins=4, min_entries=2)


def test_evaluate_segment_fold_validates_configuration_before_data_skip():
    frame = _segment_frame(groups=("a",), events_per_group=4, seed=61)
    fold = _fold_for(frame, ("a", 0))

    with pytest.raises(ValueError, match="n_bins"):
        evaluate_segment_fold(frame, fold, n_bins=0)
    with pytest.raises(ValueError, match="mad_nsigma"):
        evaluate_segment_fold(frame, fold, mad_nsigma=0.0)
    with pytest.raises(ValueError, match="bandwidth_q"):
        evaluate_segment_fold(frame, fold, bandwidth_q=0.0)


def test_fold_edges_and_pivot_are_unaffected_by_extreme_validation_values():
    frame = _segment_frame()
    fold = _fold_for(frame, ("c", 0))
    baseline = evaluate_segment_fold(
        frame, fold, n_bins=6, min_entries=5, bandwidth_q=100.0
    )
    extreme = frame.copy()
    extreme.loc[fold.validation_indices, "energy"] = np.geomspace(
        1.0, 1.0e9, len(fold.validation_indices)
    )

    changed = evaluate_segment_fold(
        extreme, fold, n_bins=6, min_entries=5, bandwidth_q=100.0
    )

    assert np.array_equal(changed.inv_q_edges, baseline.inv_q_edges)
    assert changed.pivot_q == baseline.pivot_q
    assert changed.accepted_q_range == baseline.accepted_q_range


def test_model_prediction_status_excludes_values_outside_transfer_domain(
    monkeypatch,
):
    frame = _segment_frame(events_per_group=120, seed=62)
    fold = _fold_for(frame, ("c", 0))
    original_fit = models_module.fit_affine_transfer

    def narrowed_transfer(*args, **kwargs):
        fit = original_fit(*args, **kwargs)
        lower = fit.inv_slope_min_ns_per_ua
        upper = fit.inv_slope_max_ns_per_ua
        return TransferFit(
            offset_ps=fit.offset_ps,
            scale_ps_ua_per_ns=fit.scale_ps_ua_per_ns,
            covariance=fit.covariance,
            inv_slope_min_ns_per_ua=lower + 0.25 * (upper - lower),
            inv_slope_max_ns_per_ua=upper - 0.25 * (upper - lower),
        )

    monkeypatch.setattr(models_module, "fit_affine_transfer", narrowed_transfer)

    result = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert result.status == "ok"
    for model in ("lowess", "quadratic"):
        status_column = "slope_{}_prediction_status".format(model)
        assert {
            "accepted",
            "outside_transfer_domain",
        } <= set(result.predictions[status_column])
        excluded = result.predictions[status_column].eq(
            "outside_transfer_domain"
        )
        assert result.predictions.loc[
            excluded, "slope_{}_prediction_ps".format(model)
        ].isna().all()


def test_comparison_metrics_use_one_common_model_valid_mask(monkeypatch):
    frame = _segment_frame(events_per_group=160, seed=65)
    fold = _fold_for(frame, ("c", 0))
    original_fit = models_module.fit_affine_transfer
    calls = {"count": 0}

    def asymmetric_transfer(*args, **kwargs):
        fit = original_fit(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] != 1:
            return fit
        lower = fit.inv_slope_min_ns_per_ua
        upper = fit.inv_slope_max_ns_per_ua
        return TransferFit(
            offset_ps=fit.offset_ps,
            scale_ps_ua_per_ns=fit.scale_ps_ua_per_ns,
            covariance=fit.covariance,
            inv_slope_min_ns_per_ua=lower + 0.20 * (upper - lower),
            inv_slope_max_ns_per_ua=upper - 0.20 * (upper - lower),
        )

    monkeypatch.setattr(models_module, "fit_affine_transfer", asymmetric_transfer)

    result = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert result.status == "ok"
    lowess_valid = result.predictions[
        "slope_lowess_prediction_status"
    ].eq("accepted")
    quadratic_valid = result.predictions[
        "slope_quadratic_prediction_status"
    ].eq("accepted")
    common = lowess_valid & quadratic_valid
    assert lowess_valid.sum() < quadratic_valid.sum()
    assert result.metrics["matched_validation_event_count"] == common.sum()
    assert result.predictions["comparison_status"].eq("accepted").equals(common)

    residual_columns = {
        "raw": "raw_residual_ps",
        "reference": "reference_corrected_residual_ps",
        "slope_lowess": "slope_lowess_corrected_residual_ps",
        "slope_quadratic": "slope_quadratic_corrected_residual_ps",
    }
    for name, column in residual_columns.items():
        assert result.metrics[
            "robust_width_{}_ps".format(name)
        ] == pytest.approx(
            robust_sigma(result.predictions.loc[common, column])
        )
        assert result.metrics[
            "width_{}_matched_event_count".format(name)
        ] == common.sum()

    profile = result.validation_profile.sort_values("inv_q_mean")
    profile_q = profile["energy_mean"].to_numpy(dtype=float)
    observed = profile["delta_t_ref_mean_ps"].to_numpy(dtype=float)
    profile_common = np.ones(len(profile), dtype=bool)
    for model in ("lowess", "quadratic"):
        inverse_slope = 1.0 / result.slope_models.evaluate(
            profile_q, model=model
        )
        transfer = result.transfers[model]
        valid = (
            (inverse_slope >= transfer.inv_slope_min_ns_per_ua)
            & (inverse_slope <= transfer.inv_slope_max_ns_per_ua)
        )
        profile_common &= valid
    assert result.metrics["matched_validation_bin_count"] == profile_common.sum()
    for model in ("lowess", "quadratic"):
        inverse_slope = 1.0 / result.slope_models.evaluate(
            profile_q[profile_common], model=model
        )
        transferred = result.transfers[model].evaluate(inverse_slope)
        residual = transferred - observed[profile_common]
        assert result.metrics[
            "correction_rmse_{}_ps".format(model)
        ] == pytest.approx(np.sqrt(np.mean(residual**2)))


def test_fold_skips_when_models_have_no_common_transfer_valid_events(monkeypatch):
    frame = _segment_frame(events_per_group=120, seed=66)
    fold = _fold_for(frame, ("c", 0))
    original_fit = models_module.fit_affine_transfer
    calls = {"count": 0}

    def disjoint_transfers(*args, **kwargs):
        fit = original_fit(*args, **kwargs)
        calls["count"] += 1
        lower = fit.inv_slope_min_ns_per_ua
        upper = fit.inv_slope_max_ns_per_ua
        if calls["count"] == 1:
            domain = (lower, lower + 0.1 * (upper - lower))
        else:
            domain = (upper - 0.1 * (upper - lower), upper)
        return TransferFit(
            offset_ps=fit.offset_ps,
            scale_ps_ua_per_ns=fit.scale_ps_ua_per_ns,
            covariance=fit.covariance,
            inv_slope_min_ns_per_ua=domain[0],
            inv_slope_max_ns_per_ua=domain[1],
        )

    monkeypatch.setattr(models_module, "fit_affine_transfer", disjoint_transfers)

    result = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert result.status == "skipped"
    assert result.reason == "insufficient_matched_validation_events"
    assert result.metrics["matched_validation_event_count"] == 0
    assert set(result.predictions["comparison_status"]) >= {
        "excluded_by_model_domain"
    }


def test_numerical_prediction_failure_marks_folds_failed_without_aborting_channel(
    monkeypatch,
):
    frame = _segment_frame(events_per_group=72, seed=63).assign(channelID=137)

    def fail_correction(*args, **kwargs):
        raise FloatingPointError("synthetic prediction failure")

    monkeypatch.setattr(
        models_module, "evaluate_slope_correction", fail_correction
    )

    results = evaluate_channel_folds(
        frame, n_bins=6, min_entries=5, bandwidth_q=100.0
    )

    assert len(results) == 3
    assert all(result.status == "failed" for result in results)
    assert all("prediction" in result.reason for result in results)


def test_held_out_timing_perturbation_changes_shape_and_correction_metrics_only():
    frame = _segment_frame(events_per_group=120, seed=71)
    fold = _fold_for(frame, ("c", 0))
    baseline = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )
    perturbed = frame.copy()
    validation_q = perturbed.loc[fold.validation_indices, "energy"]
    perturbed.loc[fold.validation_indices, "delta_t_ref_ps"] += (
        0.0015 * (validation_q - validation_q.median()) ** 2
    )

    changed = evaluate_segment_fold(
        perturbed, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert baseline.status == changed.status == "ok"
    _assert_training_artifacts_equal(baseline, changed)
    for model in ("lowess", "quadratic"):
        assert changed.metrics["shape_rmse_{}".format(model)] != pytest.approx(
            baseline.metrics["shape_rmse_{}".format(model)]
        )
        assert changed.metrics[
            "shape_max_deviation_{}".format(model)
        ] != pytest.approx(
            baseline.metrics["shape_max_deviation_{}".format(model)]
        )
        assert changed.metrics[
            "correction_rmse_{}_ps".format(model)
        ] != pytest.approx(
            baseline.metrics["correction_rmse_{}_ps".format(model)]
        )
        assert changed.metrics[
            "correction_mad_{}_ps".format(model)
        ] != pytest.approx(
            baseline.metrics["correction_mad_{}_ps".format(model)]
        )
        prediction_column = "slope_{}_prediction_ps".format(model)
        assert np.allclose(
            baseline.predictions[prediction_column],
            changed.predictions[prediction_column],
            equal_nan=True,
        )
        residual_column = "slope_{}_transfer_residual_ps".format(model)
        assert not np.allclose(
            baseline.predictions[residual_column],
            changed.predictions[residual_column],
            equal_nan=True,
        )

        profile = changed.validation_profile.sort_values("inv_q_mean")
        profile_q = profile["energy_mean"].to_numpy(dtype=float)
        observed_timing = profile["delta_t_ref_mean_ps"].to_numpy(dtype=float)
        predicted_slope = changed.slope_models.evaluate(profile_q, model=model)
        shape_residual = normalize_endpoints(
            1.0 / predicted_slope
        ) - normalize_endpoints(observed_timing)
        assert changed.metrics["shape_rmse_{}".format(model)] == pytest.approx(
            np.sqrt(np.mean(shape_residual**2))
        )
        transferred = changed.transfers[model].evaluate(1.0 / predicted_slope)
        correction_residual = transferred - observed_timing
        assert changed.metrics[
            "correction_rmse_{}_ps".format(model)
        ] == pytest.approx(np.sqrt(np.mean(correction_residual**2)))
        assert changed.metrics[
            "correction_mad_{}_ps".format(model)
        ] == pytest.approx(np.mean(np.abs(correction_residual)))
        sem = profile["delta_t_ref_sem_ps"].to_numpy(dtype=float)
        assert changed.metrics[
            "weighted_chi2_{}".format(model)
        ] == pytest.approx(np.sum((correction_residual / sem) ** 2))
        assert changed.metrics[
            "weighted_chi2_dof_{}".format(model)
        ] == len(profile)
        assert changed.metrics[
            "weighted_chi2_per_dof_{}".format(model)
        ] == pytest.approx(
            changed.metrics["weighted_chi2_{}".format(model)] / len(profile)
        )
    assert not np.allclose(
        baseline.validation_profile["delta_t_ref_mean_ps"],
        changed.validation_profile["delta_t_ref_mean_ps"],
    )


def test_held_out_slope_perturbation_changes_validation_metrics_not_training():
    frame = _segment_frame(events_per_group=120, seed=72)
    fold = _fold_for(frame, ("c", 0))
    baseline = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )
    perturbed = frame.copy()
    validation_q = perturbed.loc[fold.validation_indices, "energy"]
    high_q = validation_q > validation_q.median()
    high_q_indices = validation_q.index[high_q]
    perturbed.loc[high_q_indices, "slope_ua_per_ns"] *= 1.35

    changed = evaluate_segment_fold(
        perturbed, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert baseline.status == changed.status == "ok"
    _assert_training_artifacts_equal(baseline, changed)
    assert not np.allclose(
        baseline.validation_profile["slope_mean_ua_per_ns"],
        changed.validation_profile["slope_mean_ua_per_ns"],
    )
    for model in ("lowess", "quadratic"):
        key = "slope_validation_rmse_{}_ua_per_ns".format(model)
        assert changed.metrics[key] != pytest.approx(baseline.metrics[key])
    assert not np.allclose(
        baseline.predictions["slope_ua_per_ns"],
        changed.predictions["slope_ua_per_ns"],
    )


@pytest.mark.parametrize(
    ("frame", "held_out", "expected_reason"),
    [
        (
            pd.DataFrame(
                {
                    "file_id": ["train"] * 12 + ["validation"] * 8,
                    "segment": 0,
                    "energy": [100.0] * 6 + [200.0] * 6 + [150.0] * 8,
                    "slope_ua_per_ns": np.linspace(10.0, 12.0, 20),
                    "delta_t_ref_ps": np.linspace(20.0, 40.0, 20),
                }
            ),
            ("validation", 0),
            "insufficient_training_bins",
        ),
        (
            pd.concat(
                [
                    _segment_frame(groups=("train",), events_per_group=72),
                    _segment_frame(
                        groups=("validation",), events_per_group=4, seed=99
                    ),
                ]
            ),
            ("validation", 0),
            "insufficient_validation_events",
        ),
    ],
)
def test_fold_skip_reasons_are_explicit(frame, held_out, expected_reason):
    frame = frame.copy()
    frame.index = pd.Index(range(1000, 1000 + len(frame)))
    fold = _fold_for(frame, held_out)

    result = evaluate_segment_fold(
        frame, fold, n_bins=6, min_entries=2, bandwidth_q=100.0
    )

    assert result.status == "skipped"
    assert result.reason == expected_reason
    assert result.metrics["fold_status"] == "skipped"
    assert result.metrics["fold_reason"] == expected_reason


def test_fold_skips_when_only_one_validation_profile_bin_is_retained():
    frame = _segment_frame(events_per_group=72, seed=81)
    fold = _fold_for(frame, ("c", 0))
    baseline = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=5, bandwidth_q=100.0
    )
    concentrated = frame.copy()
    bin_index = len(baseline.inv_q_edges) // 2
    inv_q_center = np.mean(
        baseline.inv_q_edges[bin_index - 1 : bin_index + 1]
    )
    q_center = 1.0 / inv_q_center
    concentrated.loc[fold.validation_indices, "energy"] = np.linspace(
        q_center - 0.1, q_center + 0.1, len(fold.validation_indices)
    )

    result = evaluate_segment_fold(
        concentrated, fold, n_bins=8, min_entries=5, bandwidth_q=100.0
    )

    assert result.status == "skipped"
    assert result.reason == "insufficient_validation_bins"
    assert result.metrics["retained_validation_bin_count"] == 1
    assert len(result.predictions) == len(fold.validation_indices)
    assert set(result.predictions["prediction_status"]) == {"accepted"}
    assert np.isfinite(result.predictions["reference_prediction_ps"]).all()
    assert np.isfinite(result.predictions["slope_lowess_prediction_ps"]).all()
    assert np.isnan(result.metrics["shape_rmse_lowess"])
    assert result.metrics["shape_rmse_lowess_status"] == "not_evaluated"
    assert np.isnan(result.metrics["correction_rmse_quadratic_ps"])
    assert (
        result.metrics["correction_rmse_quadratic_ps_status"]
        == "not_evaluated"
    )


def test_fold_skips_when_accepted_validation_profile_is_empty():
    frame = _segment_frame(events_per_group=72, seed=82)
    fold = _fold_for(frame, ("c", 0))
    baseline = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=5, bandwidth_q=100.0
    )
    sparse = frame.copy()
    sparse.loc[fold.validation_indices, "energy"] = 1.0e6
    inv_q_centers = 0.5 * (
        baseline.inv_q_edges[:-1] + baseline.inv_q_edges[1:]
    )
    q_centers = 1.0 / inv_q_centers
    q_min, q_max = baseline.accepted_q_range
    accepted_centers = q_centers[
        (q_centers >= q_min) & (q_centers <= q_max)
    ][:5]
    assert len(accepted_centers) == 5
    sparse.loc[fold.validation_indices[:5], "energy"] = accepted_centers

    result = evaluate_segment_fold(
        sparse, fold, n_bins=8, min_entries=5, bandwidth_q=100.0
    )

    assert result.metrics["accepted_validation_event_count"] == 5
    assert result.validation_profile.empty
    assert result.status == "skipped"
    assert result.reason == "insufficient_validation_bins"
    assert len(result.predictions) == len(fold.validation_indices)
    assert set(result.predictions["prediction_status"]) == {
        "accepted",
        "outside_training_range",
    }
    accepted_predictions = result.predictions.query(
        "prediction_status == 'accepted'"
    )
    assert np.isfinite(
        accepted_predictions["slope_quadratic_prediction_ps"]
    ).all()
    assert np.isnan(result.metrics["shape_max_deviation_quadratic"])
    assert np.isnan(result.metrics["correction_mad_lowess_ps"])


def test_validation_bin_threshold_cannot_be_configured_below_two():
    frame = _segment_frame(events_per_group=24, seed=83)
    fold = _fold_for(frame, ("c", 0))

    with pytest.raises(ValueError, match="min_validation_bins.*at least two"):
        evaluate_segment_fold(frame, fold, min_validation_bins=1)


def test_synthetic_affine_transfer_recovers_and_improves_held_out_width():
    frame = _segment_frame(events_per_group=120, seed=7)
    fold = _fold_for(frame, ("c", 0))

    result = evaluate_segment_fold(
        frame, fold, n_bins=8, min_entries=8, bandwidth_q=100.0
    )

    assert isinstance(result, FoldEvaluation)
    assert result.status == "ok"
    assert result.transfers["lowess"].scale_ps_ua_per_ns == pytest.approx(
        7200.0, rel=0.08
    )
    assert result.transfers["quadratic"].scale_ps_ua_per_ns == pytest.approx(
        7200.0, rel=0.08
    )
    assert (
        result.metrics["robust_width_slope_lowess_ps"]
        < result.metrics["robust_width_raw_ps"]
    )
    assert (
        result.metrics["robust_width_slope_quadratic_ps"]
        < result.metrics["robust_width_raw_ps"]
    )


def test_gaussian_core_width_has_success_and_no_rms_fallback_on_failure():
    rng = np.random.default_rng(18)
    values = rng.normal(40.0, 6.0, 400)

    width = gaussian_core_width(values)
    failed = gaussian_core_width(np.ones(20))

    assert isinstance(width, TimingWidth)
    assert width.status == "ok"
    assert width.count >= 350
    assert width.sigma_ps == pytest.approx(6.0, rel=0.12)
    assert width.uncertainty_ps > 0.0
    assert robust_sigma(values) == pytest.approx(6.0, rel=0.15)
    assert failed.status == "zero_robust_sigma"
    assert np.isnan(failed.sigma_ps)
    assert np.isnan(failed.uncertainty_ps)
    assert np.isnan(robust_sigma([12.0]))
    one_event = gaussian_core_width([12.0])
    assert one_event.status == "insufficient_events"
    assert np.isnan(one_event.sigma_ps)


def test_gaussian_core_truncated_likelihood_is_accurate_for_large_normal_sample():
    rng = np.random.default_rng(2025)
    values = rng.normal(35.0, 7.0, 20_000)

    width = gaussian_core_width(values)

    assert width.status == "ok"
    assert width.sigma_ps == pytest.approx(7.0, rel=0.015)
    assert 0.0 < width.uncertainty_ps < 0.15


def test_segment_stratified_bootstrap_is_deterministic_and_preserves_groups():
    frame = pd.DataFrame(
        {
            "file_id": ["train"] * 5 + ["train"] * 7,
            "segment": [0] * 5 + [1] * 7,
            "value": np.arange(12, dtype=float),
        },
        index=np.arange(50, 62),
    )
    expected_sizes = {("train", 0): 5, ("train", 1): 7}

    def statistic(sample):
        sizes = sample.groupby(["file_id", "segment"]).size().to_dict()
        assert sizes == expected_sizes
        assert sample.index.is_unique
        assert sample.index.equals(pd.RangeIndex(len(sample)))
        assert set(sample["bootstrap_source_index"]).issubset(frame.index)
        return {
            "mean": sample["value"].mean(),
            "curve": np.array([sample["value"].min(), sample["value"].max()]),
        }

    first = stratified_bootstrap(frame, statistic, n_replicates=20, seed=4399)
    second = stratified_bootstrap(frame, statistic, n_replicates=20, seed=4399)

    assert isinstance(first, BootstrapResult)
    assert first.n_replicates == 20
    assert np.array_equal(first.samples["curve"], second.samples["curve"])
    assert first.intervals["mean"].p16 == second.intervals["mean"].p16
    assert np.shape(first.intervals["curve"].p50) == (2,)
    assert np.all(first.intervals["curve"].p16 <= first.intervals["curve"].p84)


def test_bootstrap_uses_finite_replicates_and_reports_insufficient_fraction():
    frame = pd.DataFrame(
        {
            "file_id": ["a"] * 5 + ["b"] * 5,
            "segment": [0] * 5 + [1] * 5,
            "value": np.arange(10, dtype=float),
        },
        index=np.arange(100, 110),
    )
    calls = {"count": 0}

    def statistic(sample):
        calls["count"] += 1
        replicate = calls["count"]
        return {
            "enough": sample["value"].mean() if replicate <= 6 else np.nan,
            "too_sparse": sample["value"].mean() if replicate <= 5 else np.nan,
        }

    result = stratified_bootstrap(
        frame,
        statistic,
        n_replicates=10,
        seed=4399,
        min_success_fraction=0.6,
    )

    assert result.minimum_success_fraction == 0.6
    assert result.intervals["enough"].finite_count == 6
    assert result.intervals["enough"].status == "ok"
    assert np.isfinite(result.intervals["enough"].p50)
    assert result.intervals["too_sparse"].finite_count == 5
    assert result.intervals["too_sparse"].status == (
        "insufficient_finite_replicates"
    )
    assert np.isnan(result.intervals["too_sparse"].p50)


def test_fold_exports_required_metrics_predictions_and_source_indices():
    frame = _segment_frame(events_per_group=80, seed=42)
    fold = _fold_for(frame, ("b", 1))
    outside_index = fold.validation_indices[0]
    frame.loc[outside_index, "energy"] = 1.0e6

    result = evaluate_segment_fold(
        frame, fold, n_bins=6, min_entries=5, bandwidth_q=100.0
    )

    assert result.status == "ok"
    assert REQUIRED_METRIC_KEYS <= set(result.metrics)
    assert set(result.predictions["source_index"]) == set(fold.validation_indices)
    assert set(result.predictions["prediction_status"]) == {
        "accepted",
        "outside_training_range",
    }
    accepted = result.predictions.query("prediction_status == 'accepted'")
    required_predictions = {
        "reference_correction_ps",
        "slope_lowess_correction_ps",
        "slope_quadratic_correction_ps",
        "raw_residual_ps",
        "reference_corrected_residual_ps",
        "slope_lowess_corrected_residual_ps",
        "slope_quadratic_corrected_residual_ps",
        "slope_lowess_transfer_residual_ps",
        "slope_quadratic_transfer_residual_ps",
    }
    assert required_predictions <= set(accepted.columns)
    assert np.isfinite(accepted[list(required_predictions)].to_numpy()).all()
    status_tracked_metrics = (
        "shape_rmse_lowess",
        "shape_max_deviation_quadratic",
        "slope_validation_rmse_lowess_ua_per_ns",
        "correction_rmse_quadratic_ps",
        "weighted_chi2_lowess",
        "weighted_chi2_per_dof_quadratic",
        "residual_slope_reference_ps_per_inv_q",
        "robust_width_slope_lowess_ps",
        "gaussian_width_slope_quadratic_ps",
    )
    for metric in status_tracked_metrics:
        assert np.isfinite(result.metrics[metric])
        assert result.metrics["{}_status".format(metric)] == "ok"
    expected_validation_profile = build_comparison_profile(
        frame.loc[accepted["source_index"]],
        min_entries=5,
        mad_nsigma=5.0,
        inv_q_edges=result.inv_q_edges,
    )
    pd.testing.assert_frame_equal(
        result.validation_profile.reset_index(drop=True),
        expected_validation_profile.reset_index(drop=True),
    )
