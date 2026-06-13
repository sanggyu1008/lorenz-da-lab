"""
Diagnostic utilities.

이 파일은 RMSE 등 실험 결과를 진단하는 함수를 정의한다.
"""

from __future__ import annotations

import numpy as np


def rmse_time_series(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """
    각 시간별 RMSE를 계산한다.

    Lorenz-63의 경우 각 시간에서 x, y, z 세 변수에 대해 RMSE를 계산한다.

    Parameters
    ----------
    estimate : np.ndarray, shape (nt, state_dim)
        예측값 또는 분석값.
    truth : np.ndarray, shape (nt, state_dim)
        참값.

    Returns
    -------
    rmse : np.ndarray, shape (nt,)
        시간별 RMSE.
    """
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)

    if estimate.shape != truth.shape:
        raise ValueError(
            "estimate와 truth의 shape이 같아야 한다. "
            f"estimate shape: {estimate.shape}, truth shape: {truth.shape}"
        )

    return np.sqrt(np.mean((estimate - truth) ** 2, axis=1))


def mean_rmse(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> float:
    """
    전체 기간 평균 RMSE를 계산한다.
    """
    return float(np.mean(rmse_time_series(estimate, truth)))