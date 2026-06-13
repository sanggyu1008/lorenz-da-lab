"""
4th-order Runge-Kutta time integration.

이 파일은 explicit RK4 시간적분 함수를 정의한다.
euler.py와 마찬가지로 특정 모델에 의존하지 않도록 작성한다.

RK4 method:

    k1 = f(x_n)
    k2 = f(x_n + dt/2 * k1)
    k3 = f(x_n + dt/2 * k2)
    k4 = f(x_n + dt * k3)
    x_{n+1} = x_n + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

Euler보다 한 step 비용은 크지만, 같은 dt에서 정확도가 훨씬 높다(4차 정확도).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def rk4_step(
    state: np.ndarray,
    dt: float,
    rhs_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    RK4 방법으로 한 time step 전진한다.

    Parameters
    ----------
    state : np.ndarray
        현재 상태벡터.
    dt : float
        시간간격.
    rhs_func : callable
        RHS 계산 함수. 입력은 state, 출력은 dstate/dt.

    Returns
    -------
    next_state : np.ndarray
        다음 시점의 상태벡터.
    """
    state = np.asarray(state, dtype=float)

    if dt <= 0:
        raise ValueError(f"dt는 양수여야 한다. 현재 dt: {dt}")

    k1 = np.asarray(rhs_func(state), dtype=float)
    k2 = np.asarray(rhs_func(state + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(rhs_func(state + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(rhs_func(state + dt * k3), dtype=float)

    if k1.shape != state.shape:
        raise ValueError(
            "rhs_func(state)의 shape이 state와 같아야 한다. "
            f"state shape: {state.shape}, rhs shape: {k1.shape}"
        )

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_rk4(
    x0: np.ndarray,
    dt: float,
    nsteps: int,
    rhs_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    RK4 방법으로 모델을 적분한다.

    Parameters
    ----------
    x0 : np.ndarray
        초기조건.
    dt : float
        시간간격.
    nsteps : int
        시간적분 step 수.
    rhs_func : callable
        RHS 계산 함수.

    Returns
    -------
    states : np.ndarray, shape (nsteps + 1, state_dim)
        전체 시간에 대한 상태벡터 배열.
    """
    x0 = np.asarray(x0, dtype=float)

    if dt <= 0:
        raise ValueError(f"dt는 양수여야 한다. 현재 dt: {dt}")

    if nsteps < 0:
        raise ValueError(f"nsteps는 0 이상이어야 한다. 현재 nsteps: {nsteps}")

    states = np.zeros((nsteps + 1, *x0.shape), dtype=float)
    states[0] = x0

    for n in range(nsteps):
        states[n + 1] = rk4_step(states[n], dt, rhs_func)

    return states
