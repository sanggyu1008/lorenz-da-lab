"""Lorenz-96 모델 검증."""

from __future__ import annotations

import numpy as np

from lorenz_da.models.lorenz96 import (
    lorenz96_default_initial_condition,
    lorenz96_rhs,
)
from lorenz_da.numerics.rk4 import integrate_rk4


def test_rhs_shape():
    x = np.ones(40)
    assert lorenz96_rhs(x).shape == (40,)


def test_constant_forcing_is_fixed_point():
    """x = F * ones는 고정점이므로 rhs가 0이다."""
    F = 8.0
    x = F * np.ones(40)
    rhs = lorenz96_rhs(x, F=F)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_requires_at_least_four_variables():
    try:
        lorenz96_rhs(np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("K < 4이면 ValueError가 발생해야 한다.")


def test_integration_stays_bounded():
    """RK4로 적분해도 발산하지 않고 유한한 attractor 안에 머문다."""
    x0 = lorenz96_default_initial_condition(K=40, F=8.0)
    traj = integrate_rk4(x0, dt=0.01, nsteps=2000, rhs_func=lorenz96_rhs)
    assert np.all(np.isfinite(traj))
    # Lorenz-96 (F=8) attractor는 대략 [-12, 18] 범위에 머문다.
    assert traj[1000:].max() < 25
    assert traj[1000:].min() > -20
