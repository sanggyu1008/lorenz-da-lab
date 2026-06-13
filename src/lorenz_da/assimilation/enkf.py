"""
Stochastic Ensemble Kalman Filter (perturbed observation EnKF).

이 파일은 perturbed-observation 방식의 stochastic EnKF analysis update를 구현한다.

3D-Var가 정적(static) background error covariance B를 사용하는 것과 달리,
EnKF는 앙상블에서 추정한 흐름 의존(flow-dependent) covariance P^f를 사용한다.

    P^f = (1 / (N - 1)) X' X'^T

여기서 X'는 앙상블 평균을 뺀 perturbation 행렬이다.
"""

from __future__ import annotations

import numpy as np


def initialize_ensemble(
    x0: np.ndarray,
    ensemble_size: int,
    spread: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    초기 상태 x0 주변에 Gaussian noise를 더해 초기 앙상블을 만든다.

        x_i = x0 + eta_i,   eta_i ~ N(0, spread^2 I)

    Parameters
    ----------
    x0 : np.ndarray, shape (state_dim,)
        앙상블의 중심이 되는 초기 상태.
    ensemble_size : int
        앙상블 멤버 수 N.
    spread : float
        초기 앙상블 perturbation의 표준편차.
    rng : np.random.Generator or None
        난수 생성기.

    Returns
    -------
    ensemble : np.ndarray, shape (ensemble_size, state_dim)
        초기 앙상블.
    """
    x0 = np.asarray(x0, dtype=float)

    if x0.ndim != 1:
        raise ValueError(f"x0는 1차원이어야 한다. 현재 shape: {x0.shape}")

    if ensemble_size <= 0:
        raise ValueError(
            f"ensemble_size는 양수여야 한다. 현재 ensemble_size: {ensemble_size}"
        )

    if spread <= 0:
        raise ValueError(f"spread는 양수여야 한다. 현재 spread: {spread}")

    if rng is None:
        rng = np.random.default_rng()

    state_dim = x0.shape[0]
    noise = rng.normal(loc=0.0, scale=spread, size=(ensemble_size, state_dim))

    return x0[np.newaxis, :] + noise


def ensemble_mean(ensemble: np.ndarray) -> np.ndarray:
    """
    앙상블 평균 상태를 계산한다.

    Parameters
    ----------
    ensemble : np.ndarray, shape (N, state_dim)
        앙상블 멤버들.

    Returns
    -------
    mean : np.ndarray, shape (state_dim,)
        앙상블 평균.
    """
    ensemble = np.asarray(ensemble, dtype=float)

    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble은 2차원 (N, state_dim)이어야 한다. 현재 shape: {ensemble.shape}"
        )

    return ensemble.mean(axis=0)


def ensemble_perturbations(ensemble: np.ndarray) -> np.ndarray:
    """
    앙상블 perturbation 행렬을 계산한다.

        X' = X - x_bar

    각 멤버에서 앙상블 평균을 뺀 값이다.

    Parameters
    ----------
    ensemble : np.ndarray, shape (N, state_dim)
        앙상블 멤버들.

    Returns
    -------
    perturbations : np.ndarray, shape (N, state_dim)
        평균을 뺀 perturbation 행렬.
    """
    ensemble = np.asarray(ensemble, dtype=float)

    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble은 2차원 (N, state_dim)이어야 한다. 현재 shape: {ensemble.shape}"
        )

    return ensemble - ensemble_mean(ensemble)[np.newaxis, :]


def sample_covariance(ensemble: np.ndarray) -> np.ndarray:
    """
    앙상블로부터 표본 공분산 P^f를 계산한다.

        P^f = (1 / (N - 1)) X'^T X'

    Parameters
    ----------
    ensemble : np.ndarray, shape (N, state_dim)
        앙상블 멤버들. N >= 2 이어야 한다.

    Returns
    -------
    P : np.ndarray, shape (state_dim, state_dim)
        앙상블 표본 공분산.
    """
    ensemble = np.asarray(ensemble, dtype=float)

    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble은 2차원 (N, state_dim)이어야 한다. 현재 shape: {ensemble.shape}"
        )

    n_members = ensemble.shape[0]

    if n_members < 2:
        raise ValueError(
            f"표본 공분산 계산에는 멤버가 2개 이상 필요하다. 현재 N: {n_members}"
        )

    perturbations = ensemble_perturbations(ensemble)

    return (perturbations.T @ perturbations) / (n_members - 1)


def multiplicative_inflation(
    ensemble: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    앙상블 평균은 유지한 채 perturbation을 factor배로 키우는 곱셈 inflation을 적용한다.

        x_i_new = x_bar + factor * (x_i - x_bar)

    앙상블 EnKF에서는 표본 공분산이 시간이 지나며 과소평가되어 filter가
    관측을 무시하게 되는 현상(filter divergence)이 생긴다.
    Inflation은 이를 완화하기 위해 앙상블 spread를 인위적으로 키운다.

    Parameters
    ----------
    ensemble : np.ndarray, shape (N, state_dim)
        앙상블 멤버들.
    factor : float
        inflation 계수. 보통 1.0보다 약간 큰 값을 사용한다.

    Returns
    -------
    inflated : np.ndarray, shape (N, state_dim)
        inflation이 적용된 앙상블.
    """
    ensemble = np.asarray(ensemble, dtype=float)

    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble은 2차원 (N, state_dim)이어야 한다. 현재 shape: {ensemble.shape}"
        )

    if factor <= 0:
        raise ValueError(f"factor는 양수여야 한다. 현재 factor: {factor}")

    mean = ensemble_mean(ensemble)[np.newaxis, :]
    perturbations = ensemble - mean

    return mean + factor * perturbations


def stochastic_enkf_analysis(
    ensemble_f: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    rng: np.random.Generator | None = None,
    localization: np.ndarray | None = None,
) -> np.ndarray:
    """
    perturbed-observation 방식의 stochastic EnKF analysis를 수행한다.

    각 멤버에 대해 다음과 같이 업데이트한다.

        x_i^a = x_i^f + K (y + eps_i - H x_i^f),   eps_i ~ N(0, R)

    여기서 gain matrix는 앙상블 표본 공분산 P^f를 이용해 계산한다.

        K = P^f H^T (H P^f H^T + R)^(-1)

    3D-Var와 비교하면 정적 B 대신 P^f를 쓴다는 점만 다르다.
    관측에 noise eps_i를 더해 주는 것이 stochastic EnKF의 핵심으로,
    이를 통해 analysis 앙상블의 spread가 이론적으로 올바른 크기를 유지한다.

    Parameters
    ----------
    ensemble_f : np.ndarray, shape (N, state_dim)
        forecast 앙상블.
    y : np.ndarray, shape (nobs,)
        관측 벡터.
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    R : np.ndarray, shape (nobs, nobs)
        관측오차 공분산.
    rng : np.random.Generator or None
        난수 생성기.
    localization : np.ndarray or None, shape (state_dim, state_dim)
        localization 행렬. None이면 적용하지 않는다. 주면 P^f에 원소별 곱(Schur
        product)을 적용해 멀리 떨어진 변수의 spurious correlation을 억제한다.

    Returns
    -------
    ensemble_a : np.ndarray, shape (N, state_dim)
        analysis 앙상블.
    """
    ensemble_f = np.asarray(ensemble_f, dtype=float)
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)
    R = np.asarray(R, dtype=float)

    if ensemble_f.ndim != 2:
        raise ValueError(
            "ensemble_f는 2차원 (N, state_dim)이어야 한다. "
            f"현재 shape: {ensemble_f.shape}"
        )

    if H.ndim != 2:
        raise ValueError(f"H는 2차원이어야 한다. 현재 shape: {H.shape}")

    n_members, state_dim = ensemble_f.shape
    nobs = H.shape[0]

    if H.shape[1] != state_dim:
        raise ValueError(
            "H의 state dimension과 ensemble의 state dimension이 맞지 않는다. "
            f"H shape: {H.shape}, ensemble shape: {ensemble_f.shape}"
        )

    if y.shape != (nobs,):
        raise ValueError(
            f"y는 shape ({nobs},)이어야 한다. 현재 shape: {y.shape}"
        )

    if R.shape != (nobs, nobs):
        raise ValueError(
            f"R은 shape ({nobs}, {nobs})이어야 한다. 현재 shape: {R.shape}"
        )

    if rng is None:
        rng = np.random.default_rng()

    # 앙상블 표본 공분산으로 gain matrix를 계산한다.
    P = sample_covariance(ensemble_f)

    if localization is not None:
        localization = np.asarray(localization, dtype=float)
        if localization.shape != (state_dim, state_dim):
            raise ValueError(
                "localization은 shape (state_dim, state_dim)이어야 한다. "
                f"현재 shape: {localization.shape}, state_dim: {state_dim}"
            )
        P = localization * P

    S = H @ P @ H.T + R

    # three_dvar와 동일하게 직접 역행렬을 구하지 않고 solve를 사용한다.
    K = P @ H.T @ np.linalg.solve(S, np.eye(nobs))

    # 각 멤버마다 관측에 noise를 더하는 perturbed observation 방식.
    obs_noise = rng.normal(loc=0.0, scale=1.0, size=(n_members, nobs))
    # R = L L^T (Cholesky)로 상관 있는 noise도 올바르게 생성한다.
    chol_R = np.linalg.cholesky(R)
    perturbed_obs = y[np.newaxis, :] + obs_noise @ chol_R.T

    innovations = perturbed_obs - ensemble_f @ H.T
    ensemble_a = ensemble_f + innovations @ K.T

    return ensemble_a
