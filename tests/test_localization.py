"""Covariance localization 검증 + EnKF localization 연동."""

from __future__ import annotations

import numpy as np

from lorenz_da.assimilation.enkf import initialize_ensemble, stochastic_enkf_analysis
from lorenz_da.assimilation.localization import (
    gaspari_cohn,
    localization_matrix,
    periodic_distance_matrix,
)
from lorenz_da.observations.operator import identity_observation_operator


def test_gaspari_cohn_endpoints_and_range():
    assert np.isclose(gaspari_cohn(0.0), 1.0)
    assert np.isclose(gaspari_cohn(2.0), 0.0, atol=1e-12)
    assert np.isclose(gaspari_cohn(3.0), 0.0)

    z = np.linspace(0, 3, 200)
    vals = gaspari_cohn(z)
    assert np.all(vals >= -1e-12)
    assert np.all(vals <= 1.0 + 1e-12)


def test_gaspari_cohn_monotone_decreasing():
    z = np.linspace(0, 2, 100)
    vals = gaspari_cohn(z)
    assert np.all(np.diff(vals) <= 1e-12)


def test_periodic_distance_matrix():
    d = periodic_distance_matrix(40)
    assert d.shape == (40, 40)
    np.testing.assert_allclose(np.diag(d), 0.0)
    np.testing.assert_allclose(d, d.T)
    assert d.max() == 20  # 40개 주기격자의 최대 거리


def test_localization_matrix_properties():
    L = localization_matrix(40, length_scale=4.0)
    np.testing.assert_allclose(np.diag(L), 1.0)
    np.testing.assert_allclose(L, L.T)
    assert np.all(L >= -1e-12)
    assert np.all(L <= 1.0 + 1e-12)
    # 멀리 떨어진 점은 0
    assert L[0, 20] == 0.0


def test_enkf_accepts_localization():
    """localization을 주면 동작하고, 안 줄 때와 결과가 달라진다."""
    rng = np.random.default_rng(0)
    K = 40
    x0 = np.zeros(K)
    ens = initialize_ensemble(x0, ensemble_size=15, spread=1.0, rng=rng)

    H = identity_observation_operator(K)
    R = np.eye(K)
    y = rng.normal(size=K)

    L = localization_matrix(K, length_scale=3.0)

    a_loc = stochastic_enkf_analysis(
        ens, y, H, R, rng=np.random.default_rng(1), localization=L
    )
    a_noloc = stochastic_enkf_analysis(
        ens, y, H, R, rng=np.random.default_rng(1), localization=None
    )

    assert a_loc.shape == (15, K)
    assert not np.allclose(a_loc, a_noloc)
