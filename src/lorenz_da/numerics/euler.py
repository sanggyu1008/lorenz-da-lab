"""
Euler time integration.

이 파일은 explicit Euler 시간적분 함수를 정의한다.
특정 모델에 의존하지 않도록 작성한다.

Euler method:

    x_{n+1} = x_n + dt * f(x_n)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def euler_step(
    state: np.ndarray,
    dt: float,
    rhs_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Euler 방법으로 한 time step 전진한다.

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

    rhs = rhs_func(state)
    rhs = np.asarray(rhs, dtype=float)

    if rhs.shape != state.shape:
        raise ValueError(
            "rhs_func(state)의 shape이 state와 같아야 한다. "
            f"state shape: {state.shape}, rhs shape: {rhs.shape}"
        )

    return state + dt * rhs


def integrate_euler(
    x0: np.ndarray,
    dt: float,
    nsteps: int,
    rhs_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Euler 방법으로 모델을 적분한다.

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
        states[n + 1] = euler_step(states[n], dt, rhs_func)

    return states