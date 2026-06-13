"""RK4 적분기 검증."""

from __future__ import annotations

import numpy as np

from lorenz_da.numerics.euler import integrate_euler
from lorenz_da.numerics.rk4 import integrate_rk4, rk4_step


def test_rk4_more_accurate_than_euler_on_exponential():
    """dx/dt = x, x(0)=1 의 해는 e^t. 같은 dt에서 RK4가 Euler보다 정확하다."""

    def rhs(x):
        return x

    x0 = np.array([1.0])
    dt = 0.1
    nsteps = 10  # t = 1

    x_euler = integrate_euler(x0, dt, nsteps, rhs)[-1, 0]
    x_rk4 = integrate_rk4(x0, dt, nsteps, rhs)[-1, 0]
    exact = np.exp(1.0)

    assert abs(x_rk4 - exact) < abs(x_euler - exact)
    assert abs(x_rk4 - exact) < 1e-4


def test_rk4_fourth_order_convergence():
    """dt를 절반으로 줄이면 오차가 약 1/16로 줄어든다(4차 정확도)."""

    def rhs(x):
        return x

    x0 = np.array([1.0])
    exact = np.exp(1.0)

    err_coarse = abs(integrate_rk4(x0, 0.1, 10, rhs)[-1, 0] - exact)
    err_fine = abs(integrate_rk4(x0, 0.05, 20, rhs)[-1, 0] - exact)

    ratio = err_coarse / err_fine
    assert 10 < ratio < 20  # 이론적으로 16배


def test_rk4_step_shape():
    def rhs(x):
        return -x

    out = rk4_step(np.array([1.0, 2.0, 3.0]), 0.01, rhs)
    assert out.shape == (3,)
