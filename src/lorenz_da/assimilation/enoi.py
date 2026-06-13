"""
Ensemble Optimal Interpolation (EnOI).

EnKF(07번)는 앙상블을 모델로 함께 적분하며 매 순간 흐름 의존 공분산 P^f를 새로 추정한다.
반면 EnOI는 **미리 한 번만 만든 고정(static) 앙상블** 로부터 background error covariance를
한 번 계산해 두고, 그 값을 모든 분석 시각에 재사용한다.

고정 앙상블은 보통 긴 free run에서 뽑은 상태들의 anomaly(평균 편차)로 만든다.
이는 모델의 "기후학적(climatological) 변동"을 나타낸다.

장점: 앙상블을 적분할 필요가 없어 매우 싸다.
단점: 공분산이 그 순간의 흐름을 반영하지 못한다(정적).

분석 식 자체는 3D-Var와 같다.

    x^a = x^b + K (y - H x^b),   K = B H^T (H B H^T + R)^(-1)

차이는 B를 climatological 앙상블에서 만든다는 점뿐이다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.assimilation.three_dvar import analysis_3dvar


def climatological_anomalies(
    long_run: np.ndarray,
    n_members: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    긴 free run에서 상태들을 뽑아 climatological anomaly 행렬을 만든다.

    anomaly는 전체 run의 시간 평균을 뺀 값이다.

        a_i = x_{t_i} - x_mean

    Parameters
    ----------
    long_run : np.ndarray, shape (nt, state_dim)
        긴 모델 적분 결과 (climatology를 대표).
    n_members : int
        뽑을 anomaly 개수 (정적 앙상블 크기).
    rng : np.random.Generator or None
        난수 생성기.

    Returns
    -------
    anomalies : np.ndarray, shape (n_members, state_dim)
        평균을 뺀 anomaly 행렬.
    """
    long_run = np.asarray(long_run, dtype=float)

    if long_run.ndim != 2:
        raise ValueError(
            f"long_run은 2차원 (nt, state_dim)이어야 한다. 현재 shape: {long_run.shape}"
        )

    nt = long_run.shape[0]

    if n_members <= 1:
        raise ValueError(f"n_members는 2 이상이어야 한다. 현재: {n_members}")

    if n_members > nt:
        raise ValueError(
            f"n_members({n_members})가 long_run 길이({nt})보다 클 수 없다."
        )

    if rng is None:
        rng = np.random.default_rng()

    mean_state = long_run.mean(axis=0)
    idx = rng.choice(nt, size=n_members, replace=False)
    sample = long_run[idx]

    return sample - mean_state[np.newaxis, :]


def enoi_covariance(anomalies: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    climatological anomaly로부터 background error covariance를 만든다.

        B = alpha * (1 / (N - 1)) A^T A

    alpha는 climatological 변동을 background error 크기에 맞춰 줄이는 scaling 계수이다.
    (보통 1보다 작은 값을 쓴다.)

    Parameters
    ----------
    anomalies : np.ndarray, shape (N, state_dim)
        anomaly 행렬.
    alpha : float
        scaling 계수.

    Returns
    -------
    B : np.ndarray, shape (state_dim, state_dim)
        EnOI background error covariance.
    """
    anomalies = np.asarray(anomalies, dtype=float)

    if anomalies.ndim != 2:
        raise ValueError(
            f"anomalies는 2차원 (N, state_dim)이어야 한다. 현재 shape: {anomalies.shape}"
        )

    n_members = anomalies.shape[0]

    if n_members < 2:
        raise ValueError(f"공분산 계산에는 anomaly가 2개 이상 필요하다. 현재 N: {n_members}")

    if alpha <= 0:
        raise ValueError(f"alpha는 양수여야 한다. 현재 alpha: {alpha}")

    return alpha * (anomalies.T @ anomalies) / (n_members - 1)


def enoi_analysis(
    xb: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    B_enoi: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    고정된 climatological covariance B_enoi를 사용한 EnOI analysis를 수행한다.

    분석 식은 3D-Var와 동일하므로 analysis_3dvar를 그대로 재사용한다.
    EnOI의 본질은 "B를 어떻게 만드는가"에 있다(climatological 앙상블).

    Parameters
    ----------
    xb : np.ndarray, shape (state_dim,)
        background 상태.
    y : np.ndarray, shape (nobs,)
        관측.
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    B_enoi : np.ndarray, shape (state_dim, state_dim)
        EnOI background error covariance (enoi_covariance로 생성).
    R : np.ndarray, shape (nobs, nobs)
        관측오차 공분산.

    Returns
    -------
    xa : np.ndarray, shape (state_dim,)
        analysis 상태.
    """
    return analysis_3dvar(xb=xb, y=y, H=H, B=B_enoi, R=R)
