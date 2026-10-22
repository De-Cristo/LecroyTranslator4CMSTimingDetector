import numpy as np

from Q_Walk.qwalk_two_hit_pair_profile import rise_time_mask
from Q_Walk.qwalk_run4076_safe_time2_scatter import _fit_landau_gauss


def test_rise_time_mask_rejects_short_and_nonfinite_rises():
    values = np.array([-1.0, 12.0, 88.0, 99.999, 100.0, 101.0, np.nan])

    np.testing.assert_array_equal(
        rise_time_mask(values, 100.0),
        [False, False, False, False, True, True, False],
    )


def test_landau_gauss_fit_reports_normalized_uniform_background_mixture():
    generator = np.random.default_rng(4076)
    values = np.concatenate(
        (
            generator.normal(145.0, 14.0, 1_000),
            generator.uniform(115.0, 265.0, 200),
        )
    )
    peak = {
        "peak_q_adc": 145.0,
        "half_max_q_low_adc": 130.0,
        "half_max_q_high_adc": 170.0,
    }

    result = _fit_landau_gauss(values, peak)
    in_fit_domain = (
        (result["grid"] >= result["fit_q_low_adc"])
        & (result["grid"] <= result["fit_q_high_adc"])
    )

    assert 0.0 <= result["signal_fraction"] <= 1.0
    assert 0.0 <= result["background_fraction"] <= 1.0
    assert result["background_density_per_adc"] >= 0.0
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert np.isclose(
        trapezoid(
            result["conditional_density"][in_fit_domain],
            result["grid"][in_fit_domain],
        ),
        1.0,
        atol=2.0e-3,
    )
