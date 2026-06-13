"""
Synthetic observation utilities.

이 파일은 truth trajectory에서 synthetic observation을 생성하는 함수를 정의한다.
"""

from __future__ import annotations

import numpy as np


def make_observation_indices(
    nsteps: int,
    obs_interval: int,
) -> np.ndarray:
    """
    관측이 존재하는 time step index를 만든다.

    Parameters
    ----------
    nsteps : int
        전체 시간적분 step 수.
    obs_interval : int
        몇 model step마다 관측을 둘 것인지 나타내는 간격.

    Returns
    -------
    obs_indices : np.ndarray
        관측 시점의 time step index 배열.
    """
    if nsteps <= 0:
        raise ValueError(f"nsteps는 양수여야 한다. 현재 nsteps: {nsteps}")

    if obs_interval <= 0:
        raise ValueError(
            f"obs_interval은 양수여야 한다. 현재 obs_interval: {obs_interval}"
        )

    return np.arange(obs_interval, nsteps + 1, obs_interval)


def generate_synthetic_observations(
    truth: np.ndarray,
    obs_indices: np.ndarray,
    H: np.ndarray,
    obs_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    truth trajectory에서 synthetic observation을 생성한다.

    관측식은 다음과 같다.

        y = H x_true + epsilon

    여기서 epsilon은 평균 0, 표준편차 obs_std인 Gaussian noise이다.

    Parameters
    ----------
    truth : np.ndarray, shape (nt, state_dim)
        참값 trajectory.
    obs_indices : np.ndarray
        관측 시점 index.
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    obs_std : float
        관측오차 표준편차.
    rng : np.random.Generator or None
        난수 생성기.

    Returns
    -------
    observations : np.ndarray, shape (n_obstimes, nobs)
        synthetic observation 배열.
    """
    truth = np.asarray(truth, dtype=float)
    obs_indices = np.asarray(obs_indices, dtype=int)
    H = np.asarray(H, dtype=float)

    if truth.ndim != 2:
        raise ValueError(f"truth는 2차원이어야 한다. 현재 shape: {truth.shape}")

    if H.ndim != 2:
        raise ValueError(f"H는 2차원이어야 한다. 현재 shape: {H.shape}")

    if H.shape[1] != truth.shape[1]:
        raise ValueError(
            "H의 state dimension과 truth의 state dimension이 맞지 않는다. "
            f"H shape: {H.shape}, truth shape: {truth.shape}"
        )

    if obs_std <= 0:
        raise ValueError(f"obs_std는 양수여야 한다. 현재 obs_std: {obs_std}")

    if rng is None:
        rng = np.random.default_rng()

    nobs = H.shape[0]
    observations = np.zeros((len(obs_indices), nobs), dtype=float)

    for i, idx in enumerate(obs_indices):
        if idx < 0 or idx >= len(truth):
            raise ValueError(
                f"obs index가 truth 범위를 벗어났다. index: {idx}, len: {len(truth)}"
            )

        noise = rng.normal(loc=0.0, scale=obs_std, size=nobs)
        observations[i] = H @ truth[idx] + noise

    return observations