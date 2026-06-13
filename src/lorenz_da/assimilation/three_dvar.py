"""
Simple 3D-Var analysis.

이 파일은 기본적인 3D-Var analysis update를 구현한다.
"""

from __future__ import annotations

import numpy as np


def compute_kalman_gain_like_matrix(
    H: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    3D-Var analysis에서 사용하는 gain matrix를 계산한다.

        K = B H^T (H B H^T + R)^(-1)

    Parameters
    ----------
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    B : np.ndarray, shape (state_dim, state_dim)
        background error covariance.
    R : np.ndarray, shape (nobs, nobs)
        observation error covariance.

    Returns
    -------
    K : np.ndarray, shape (state_dim, nobs)
        gain matrix.
    """
    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)

    S = H @ B @ H.T + R

    # inv(S)를 직접 계산하기보다 solve를 사용하는 편이 수치적으로 더 안정적이다.
    K = B @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

    return K


def analysis_3dvar(
    xb: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    3D-Var analysis state를 계산한다.

    기본 식은 다음과 같다.

        xa = xb + K (y - H xb)

    Parameters
    ----------
    xb : np.ndarray, shape (state_dim,)
        background state 또는 forecast state.
    y : np.ndarray, shape (nobs,)
        observation vector.
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    B : np.ndarray, shape (state_dim, state_dim)
        background error covariance.
    R : np.ndarray, shape (nobs, nobs)
        observation error covariance.

    Returns
    -------
    xa : np.ndarray, shape (state_dim,)
        analysis state.
    """
    xb = np.asarray(xb, dtype=float)
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)

    innovation = y - H @ xb
    K = compute_kalman_gain_like_matrix(H=H, B=B, R=R)

    xa = xb + K @ innovation

    return xa


def innovation(
    xb: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    innovation, 즉 observation-minus-background를 계산한다.

        d = y - H xb

    Parameters
    ----------
    xb : np.ndarray
        background state.
    y : np.ndarray
        observation vector.
    H : np.ndarray
        observation operator.

    Returns
    -------
    d : np.ndarray
        innovation vector.
    """
    return np.asarray(y, dtype=float) - np.asarray(H, dtype=float) @ np.asarray(
        xb, dtype=float
    )