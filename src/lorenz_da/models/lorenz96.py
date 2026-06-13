"""
Lorenz-96 model.

이 파일은 Lorenz-96 방정식을 정의한다. 시간적분 방법은 넣지 않는다.

Lorenz-96 system (cyclic boundary, k = 0 .. K-1):

    dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F

Lorenz-63(3변수)와 달리 변수가 K개(보통 40개)인 공간 확장계로,
앙상블 자료동화에서 localization과 inflation의 필요성을 보여주는 표준 테스트베드이다.

forcing F = 8이면 chaotic 거동을 보인다.
"""

from __future__ import annotations

import numpy as np


def lorenz96_rhs(state: np.ndarray, F: float = 8.0) -> np.ndarray:
    """
    Lorenz-96 방정식의 우변을 계산한다.

        dx_k/dt = (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + F

    cyclic boundary이므로 인덱스는 modulo K로 순환한다.
    np.roll을 사용하면 전체 격자에 대해 한 번에 계산할 수 있다.

        x_{k+1} = np.roll(x, -1)
        x_{k-1} = np.roll(x,  1)
        x_{k-2} = np.roll(x,  2)

    Parameters
    ----------
    state : np.ndarray, shape (K,)
        현재 상태벡터. K >= 4 이어야 한다.
    F : float
        forcing 매개변수.

    Returns
    -------
    rhs : np.ndarray, shape (K,)
        시간미분값.
    """
    state = np.asarray(state, dtype=float)

    if state.ndim != 1:
        raise ValueError(f"state는 1차원이어야 한다. 현재 shape: {state.shape}")

    if state.shape[0] < 4:
        raise ValueError(
            f"Lorenz-96는 K >= 4가 필요하다. 현재 K: {state.shape[0]}"
        )

    return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F


def lorenz96_default_initial_condition(K: int = 40, F: float = 8.0) -> np.ndarray:
    """
    Lorenz-96 기본 초기조건을 반환한다.

    모든 변수를 forcing F로 두고, 한 변수에 작은 perturbation을 준다.
    이 perturbation이 chaotic 역학을 통해 전체로 퍼져 attractor로 들어간다.

    Parameters
    ----------
    K : int
        변수 개수.
    F : float
        forcing 매개변수.

    Returns
    -------
    x0 : np.ndarray, shape (K,)
        기본 초기조건.
    """
    if K < 4:
        raise ValueError(f"K는 4 이상이어야 한다. 현재 K: {K}")

    x0 = F * np.ones(K, dtype=float)
    x0[0] += 0.01
    return x0
