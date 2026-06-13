"""
EnKF analysis 모듈 검증 테스트.

확인하는 내용:
1. sample_covariance가 대칭이고 positive semi-definite이다.
2. multiplicative_inflation이 평균은 보존하고 표준편차를 factor배로 키운다.
3. H = I, 큰 N에서 stochastic EnKF analysis 평균이 동일한 P^f를 background
   covariance로 쓴 3D-Var analysis와 일치한다 (Kalman gain 일관성).
"""

from __future__ import annotations

import numpy as np

from lorenz_da.assimilation.enkf import (
    ensemble_mean,
    initialize_ensemble,
    multiplicative_inflation,
    sample_covariance,
    stochastic_enkf_analysis,
)
from lorenz_da.assimilation.three_dvar import analysis_3dvar
from lorenz_da.observations.operator import identity_observation_operator


def test_initialize_ensemble_shape_and_center():
    rng = np.random.default_rng(0)
    x0 = np.array([1.0, -2.0, 3.0])

    ensemble = initialize_ensemble(x0, ensemble_size=5000, spread=0.5, rng=rng)

    assert ensemble.shape == (5000, 3)
    # 큰 앙상블에서는 평균이 x0에 가깝다.
    np.testing.assert_allclose(ensemble_mean(ensemble), x0, atol=0.05)


def test_sample_covariance_symmetric_and_psd():
    rng = np.random.default_rng(1)
    x0 = np.array([1.0, 1.0, 1.0])
    ensemble = initialize_ensemble(x0, ensemble_size=200, spread=2.0, rng=rng)

    P = sample_covariance(ensemble)

    # 대칭성
    np.testing.assert_allclose(P, P.T, atol=1e-12)

    # positive semi-definite: 고유값이 (수치오차 범위에서) 음수가 아니어야 한다.
    eigvals = np.linalg.eigvalsh(P)
    assert np.all(eigvals > -1e-10)


def test_sample_covariance_requires_two_members():
    single = np.array([[1.0, 2.0, 3.0]])
    try:
        sample_covariance(single)
    except ValueError:
        pass
    else:
        raise AssertionError("멤버가 1개이면 ValueError가 발생해야 한다.")


def test_multiplicative_inflation_preserves_mean_and_scales_spread():
    rng = np.random.default_rng(2)
    x0 = np.array([0.0, 5.0, -3.0])
    ensemble = initialize_ensemble(x0, ensemble_size=300, spread=1.5, rng=rng)

    factor = 1.4
    inflated = multiplicative_inflation(ensemble, factor=factor)

    # 평균은 그대로 유지된다.
    np.testing.assert_allclose(
        ensemble_mean(inflated), ensemble_mean(ensemble), atol=1e-12
    )

    # 표준편차는 정확히 factor배가 된다 (perturbation의 선형 스케일링).
    std_before = ensemble.std(axis=0, ddof=1)
    std_after = inflated.std(axis=0, ddof=1)
    np.testing.assert_allclose(std_after, factor * std_before, rtol=1e-12)


def test_enkf_mean_matches_3dvar_with_same_covariance():
    """
    H = I, 큰 앙상블에서 stochastic EnKF analysis 평균은
    P^f를 background covariance B로 사용한 3D-Var analysis와 일치한다.
    """
    rng = np.random.default_rng(42)

    x_truth = np.array([2.0, -1.0, 0.5])
    x0_background = np.array([5.0, 4.0, -2.0])

    # 큰 앙상블이라야 perturbed observation noise의 표본 평균이 0에 가까워진다.
    ensemble_f = initialize_ensemble(
        x0_background, ensemble_size=60000, spread=3.0, rng=rng
    )

    H = identity_observation_operator(state_dim=3)
    obs_std = 1.5
    R = (obs_std**2) * np.eye(3)
    y = x_truth.copy()

    # EnKF analysis
    ensemble_a = stochastic_enkf_analysis(
        ensemble_f=ensemble_f, y=y, H=H, R=R, rng=rng
    )
    enkf_mean = ensemble_mean(ensemble_a)

    # 동일한 P^f를 B로 사용한 3D-Var analysis (xb = forecast 앙상블 평균)
    P = sample_covariance(ensemble_f)
    xb = ensemble_mean(ensemble_f)
    dvar = analysis_3dvar(xb=xb, y=y, H=H, B=P, R=R)

    np.testing.assert_allclose(enkf_mean, dvar, atol=0.05)
