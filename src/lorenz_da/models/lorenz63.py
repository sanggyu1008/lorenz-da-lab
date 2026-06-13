"""
Lorenz-63 model.

이 파일은 Lorenz-63 방정식 자체를 정의한다.
시간적분 방법은 이 파일에 넣지 않는다.

Lorenz-63 system:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
"""

from __future__ import annotations

import numpy as np


def lorenz63_rhs(
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Lorenz-63 방정식의 우변을 계산한다.

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        현재 상태벡터 [x, y, z].
    sigma : float
        Lorenz-63 sigma 매개변수.
    rho : float
        Lorenz-63 rho 매개변수.
    beta : float
        Lorenz-63 beta 매개변수.

    Returns
    -------
    rhs : np.ndarray, shape (3,)
        시간미분값 [dx/dt, dy/dt, dz/dt].
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (3,):
        raise ValueError(
            f"Lorenz-63 state는 shape (3,)이어야 한다. 현재 shape: {state.shape}"
        )

    x, y, z = state

    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.array([dxdt, dydt, dzdt], dtype=float)


def lorenz63_default_initial_condition() -> np.ndarray:
    """
    Lorenz-63 기본 초기조건을 반환한다.

    Returns
    -------
    x0 : np.ndarray, shape (3,)
        기본 초기조건 [1, 1, 1].
    """
    return np.array([1.0, 1.0, 1.0], dtype=float)