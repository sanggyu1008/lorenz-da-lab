"""
Covariance localization.

앙상블 크기 N이 상태차원보다 작으면, 멀리 떨어진 변수들 사이의 표본 공분산이
실제로는 0이어야 하는데도 sampling error 때문에 0이 아닌 잡음으로 나타난다.
이 잡음은 EnKF 분석을 망가뜨린다(spurious correlation).

Covariance localization은 거리에 따라 공분산을 줄여(taper) 이런 잡음을 억제한다.
보통 Gaspari-Cohn 함수를 거리에 적용해 만든 localization 행렬 L을
표본 공분산과 원소별 곱(Schur product)한다.

    P_localized = L ∘ P
"""

from __future__ import annotations

import numpy as np


def gaspari_cohn(z: np.ndarray | float) -> np.ndarray:
    """
    Gaspari-Cohn 5차 piecewise 다항식 상관함수를 계산한다.

    z = (거리) / c 이고, 함수는 z = 0에서 1, z >= 2에서 0이 되는 매끄러운 함수이다.
    (c는 localization length scale, 실제 support는 2c이다.)

    Parameters
    ----------
    z : array_like
        정규화된 거리 (>= 0).

    Returns
    -------
    value : np.ndarray
        Gaspari-Cohn 가중치, [0, 1] 범위.
    """
    z = np.abs(np.asarray(z, dtype=float))
    out = np.zeros_like(z)

    # 0 <= z <= 1
    m1 = z <= 1.0
    z1 = z[m1]
    out[m1] = (
        -0.25 * z1**5
        + 0.5 * z1**4
        + 0.625 * z1**3
        - (5.0 / 3.0) * z1**2
        + 1.0
    )

    # 1 < z <= 2
    m2 = (z > 1.0) & (z <= 2.0)
    z2 = z[m2]
    out[m2] = (
        (1.0 / 12.0) * z2**5
        - 0.5 * z2**4
        + 0.625 * z2**3
        + (5.0 / 3.0) * z2**2
        - 5.0 * z2
        + 4.0
        - (2.0 / 3.0) / z2
    )

    # z > 2 는 0 (이미 out 초기값)
    return out


def periodic_distance_matrix(n: int) -> np.ndarray:
    """
    주기경계 격자에서 점들 사이의 최단 거리 행렬을 만든다.

    격자 간격은 1로 둔다. 점 i와 j 사이 거리는 min(|i-j|, n-|i-j|)이다.

    Parameters
    ----------
    n : int
        격자점 개수.

    Returns
    -------
    dist : np.ndarray, shape (n, n)
        거리 행렬.
    """
    if n <= 0:
        raise ValueError(f"n은 양수여야 한다. 현재 n: {n}")

    idx = np.arange(n)
    raw = np.abs(idx[:, None] - idx[None, :])
    return np.minimum(raw, n - raw).astype(float)


def localization_matrix(n: int, length_scale: float) -> np.ndarray:
    """
    주기경계 격자에 대한 Gaspari-Cohn localization 행렬을 만든다.

    Parameters
    ----------
    n : int
        격자점 개수.
    length_scale : float
        localization length scale c. 클수록 멀리까지 상관을 허용한다.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        localization 행렬. 대각은 1, 거리가 멀수록 0에 가까워진다.
    """
    if length_scale <= 0:
        raise ValueError(
            f"length_scale은 양수여야 한다. 현재 length_scale: {length_scale}"
        )

    dist = periodic_distance_matrix(n)
    return gaspari_cohn(dist / length_scale)
