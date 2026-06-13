"""
Observation operator utilities.

이 파일은 Lorenz-63 상태벡터에서 어떤 변수를 관측할지 정의하는
observation operator H를 만든다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def identity_observation_operator(state_dim: int = 3) -> np.ndarray:
    """
    모든 상태변수를 관측하는 observation operator를 만든다.

    Lorenz-63에서 state = [x, y, z]이고 모든 변수를 관측하면

        H = I

    이다.

    Parameters
    ----------
    state_dim : int
        상태변수 개수.

    Returns
    -------
    H : np.ndarray, shape (state_dim, state_dim)
        단위행렬 observation operator.
    """
    if state_dim <= 0:
        raise ValueError(f"state_dim은 양수여야 한다. 현재 state_dim: {state_dim}")

    return np.eye(state_dim, dtype=float)


def partial_observation_operator(
    observed_indices: Sequence[int],
    state_dim: int = 3,
) -> np.ndarray:
    """
    일부 상태변수만 관측하는 observation operator를 만든다.

    예를 들어 Lorenz-63 state = [x, y, z]에서 x만 관측하려면
    observed_indices = [0] 이다.

    이 경우

        H = [[1, 0, 0]]

    이 된다.

    Parameters
    ----------
    observed_indices : sequence of int
        관측할 상태변수 index 목록.
        예: [0]은 x만 관측, [0, 2]는 x와 z를 관측.
    state_dim : int
        전체 상태변수 개수.

    Returns
    -------
    H : np.ndarray, shape (nobs, state_dim)
        부분 관측용 observation operator.
    """
    if state_dim <= 0:
        raise ValueError(f"state_dim은 양수여야 한다. 현재 state_dim: {state_dim}")

    observed_indices = list(observed_indices)

    if len(observed_indices) == 0:
        raise ValueError("observed_indices는 비어 있을 수 없다.")

    for idx in observed_indices:
        if idx < 0 or idx >= state_dim:
            raise ValueError(
                f"관측 index가 범위를 벗어났다. index: {idx}, state_dim: {state_dim}"
            )

    H = np.zeros((len(observed_indices), state_dim), dtype=float)

    for row, idx in enumerate(observed_indices):
        H[row, idx] = 1.0

    return H