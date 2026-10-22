import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri

spec = importlib.util.spec_from_file_location("n2_4_study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
alignment_map = study.alignment_map
density_weights = study.density_weights
mixture_percentiles = study.mixture_percentiles
propagated_covariance = study.propagated_covariance


def test_alignment_map_matches_direct_weighted_fit():
    q = np.linspace(180, 500, 100)
    s = 10 + .02 * q
    w = 1 / q**2
    beta = np.array([-26., 180., -12000.])
    matrix = alignment_map(s, q, w)
    design = np.column_stack((1 / s, np.ones(len(q))))
    y = np.polynomial.polynomial.polyval(1 / q, beta)
    expected = np.linalg.lstsq(design * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose(matrix @ beta, expected, rtol=1e-10)
    np.testing.assert_allclose(alignment_map(s, q, 31 * w), matrix, atol=1e-10)


def test_reference_offset_uncertainty_cannot_change_awc():
    q = np.linspace(200, 500, 100)
    matrix = alignment_map(10 + .02 * q, q, np.ones(len(q)))
    np.testing.assert_allclose(matrix[:, 0], [0., 1.], rtol=0, atol=1e-14)
    result = propagated_covariance(matrix, np.diag([.04, 0, 0]))
    np.testing.assert_allclose(result, [[0, 0], [0, .04]], rtol=0, atol=1e-14)


def test_correlated_covariance_matches_coefficient_sampling():
    q = np.linspace(200, 500, 100)
    matrix = alignment_map(10 + .02 * q, q, np.ones(len(q)))
    factor = np.array([[.01, 0, 0], [-3., 1., 0], [200., -100., 20.]])
    cov = factor @ factor.T
    draws = np.random.default_rng(137).normal(size=(200000, 3)) @ factor.T @ matrix.T
    np.testing.assert_allclose(np.cov(draws, rowvar=False), propagated_covariance(matrix, cov), rtol=.02)


def test_mixture_single_gaussian_and_zero_covariance():
    np.testing.assert_allclose(mixture_percentiles(np.full(10, 18.), np.full(10, 2.)),
                               18 + 2 * ndtri([.16, .5, .84]), atol=1e-9)
    np.testing.assert_allclose(mixture_percentiles(np.arange(10.), np.zeros(10)),
                               np.percentile(np.arange(10.), [16, 50, 84]))


def test_density_converts_counts_to_density_and_rejects_gaps():
    np.testing.assert_allclose(density_weights(np.array([1., 2., 4.]), [0, 2], [2, 4], [10, 30]), [5, 15, 15])
    with pytest.raises(ValueError, match="outside"):
        density_weights(np.array([3.]), [0, 4], [2, 6], [10, 30])


def test_nonpositive_and_flat_slopes_are_rejected():
    q = np.arange(1., 5.)
    with pytest.raises(ValueError, match="Invalid"):
        alignment_map(-q, q, np.ones(4))
    with pytest.raises(ValueError, match="Flat"):
        alignment_map(np.ones(4), q, np.ones(4))
