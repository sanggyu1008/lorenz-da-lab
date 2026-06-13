"""
EnOI 모듈 검증.

1. enoi_covariance가 대칭이고 positive semi-definite이다.
2. enoi_analysis가 analysis_3dvar와 정확히 같은 결과를 준다(같은 B를 줄 때).
3. enoi_analysis가 innovation을 줄인다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.assimilation.enoi import (
    climatological_anomalies,
    enoi_analysis,
    enoi_covariance,
)
from lorenz_da.assimilation.three_dvar import analysis_3dvar
from lorenz_da.models.lorenz63 import (
    lorenz63_default_initial_condition,
    lorenz63_rhs,
)
from lorenz_da.numerics.euler import integrate_euler
from lorenz_da.observations.operator import identity_observation_operator


def _long_run():
    x0 = lorenz63_default_initial_condition()
    return integrate_euler(x0=x0, dt=0.01, nsteps=3000, rhs_func=lorenz63_rhs)


def test_enoi_covariance_symmetric_and_psd():
    rng = np.random.default_rng(0)
    long_run = _long_run()
    anomalies = climatological_anomalies(long_run, n_members=100, rng=rng)
    B = enoi_covariance(anomalies, alpha=0.5)

    np.testing.assert_allclose(B, B.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(B)
    assert np.all(eigvals > -1e-8)


def test_enoi_analysis_matches_3dvar():
    rng = np.random.default_rng(1)
    long_run = _long_run()
    anomalies = climatological_anomalies(long_run, n_members=100, rng=rng)
    B = enoi_covariance(anomalies, alpha=0.5)

    H = identity_observation_operator(3)
    R = np.eye(3)
    xb = np.array([1.0, 2.0, 3.0])
    y = np.array([1.5, 1.8, 3.2])

    xa_enoi = enoi_analysis(xb=xb, y=y, H=H, B_enoi=B, R=R)
    xa_3dvar = analysis_3dvar(xb=xb, y=y, H=H, B=B, R=R)

    np.testing.assert_allclose(xa_enoi, xa_3dvar, atol=1e-12)


def test_enoi_analysis_reduces_innovation():
    rng = np.random.default_rng(2)
    long_run = _long_run()
    anomalies = climatological_anomalies(long_run, n_members=200, rng=rng)
    B = enoi_covariance(anomalies, alpha=1.0)

    H = identity_observation_operator(3)
    R = 0.01 * np.eye(3)  # 관측을 매우 신뢰
    xb = long_run[100] + np.array([3.0, -2.0, 1.0])
    y = long_run[100]  # noise 없는 관측

    xa = enoi_analysis(xb=xb, y=y, H=H, B_enoi=B, R=R)

    innov_before = np.linalg.norm(y - H @ xb)
    innov_after = np.linalg.norm(y - H @ xa)
    assert innov_after < innov_before
