import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study import align_inverse, evaluate, fit_profile, minimum_on_support, support_mask


def test_quadratic_fit_uses_bin_averaged_second_moment():
    q = np.concatenate([np.linspace(i, i + .9, 21) for i in range(10)])
    s = 3 + 2 * q + .4 * q**2
    fit, table = fit_profile(q, s, bins=10, degree=2)
    np.testing.assert_allclose(evaluate(fit, q), s, atol=1e-10)
    np.testing.assert_allclose(table.residual, 0, atol=1e-10)
    assert np.max(abs(evaluate(fit, table.q_mean_adc) - table.prediction)) > .01


def test_support_excludes_gaps_and_outer_mean_tails():
    profile = pd.DataFrame(dict(q_low_adc=[0, 4], q_high_adc=[2, 6], q_mean_adc=[1, 5]))
    np.testing.assert_array_equal(support_mask(profile, np.arange(7.)), [False, True, True, False, True, True, False])


def test_alignment_recovers_awc_offset_and_keeps_negative_awc():
    s = 2 + np.linspace(0, 3, 20)
    for a in (18., -3.):
        actual = align_inverse(s, a / s - 26)
        np.testing.assert_allclose(actual, [a, -26], atol=1e-11)
    with pytest.raises(ValueError, match="Flat"):
        align_inverse(np.ones(20), np.arange(20.))
    with pytest.raises(ValueError, match="Non-positive"):
        align_inverse(np.array([-1., 2., 3.]), np.arange(3.))


def test_quadratic_minimum_between_grid_points_is_checked():
    model = dict(degree=2, center=0, scale=1, coefficients=[-.01, 0, 1])
    assert np.all(evaluate(model, [-1., 1.]) > 0)
    assert minimum_on_support(model, np.array([-1., 1.])) == -.01
