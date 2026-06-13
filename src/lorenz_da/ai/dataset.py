"""
AI surrogate model을 위한 dataset 준비.

수치모델 trajectory에서 "현재 상태 -> 다음 상태" 또는 "현재 상태 -> 상태 변화량"
형태의 학습 데이터를 만든다.

증분(increment)을 예측하도록 두는 것이 보통 더 잘 학습된다.

    dy = x_{n+1} - x_n

이렇게 하면 model이 큰 절대값 대신 작은 변화량만 배우면 되기 때문이다.
"""

from __future__ import annotations

import numpy as np


def make_one_step_dataset(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    한 step 예측 데이터를 만든다.

        X = x_n,   Y = x_{n+1}

    Parameters
    ----------
    trajectory : np.ndarray, shape (nt, state_dim)
        모델 trajectory.

    Returns
    -------
    X : np.ndarray, shape (nt-1, state_dim)
    Y : np.ndarray, shape (nt-1, state_dim)
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory는 2차원이어야 한다. 현재 shape: {trajectory.shape}"
        )
    return trajectory[:-1].copy(), trajectory[1:].copy()


def make_increment_dataset(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    증분 예측 데이터를 만든다.

        X = x_n,   dY = x_{n+1} - x_n

    Parameters
    ----------
    trajectory : np.ndarray, shape (nt, state_dim)
        모델 trajectory.

    Returns
    -------
    X : np.ndarray, shape (nt-1, state_dim)
    dY : np.ndarray, shape (nt-1, state_dim)
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError(
            f"trajectory는 2차원이어야 한다. 현재 shape: {trajectory.shape}"
        )
    X = trajectory[:-1]
    dY = trajectory[1:] - trajectory[:-1]
    return X.copy(), dY.copy()


class Standardizer:
    """
    각 변수(열)를 평균 0, 표준편차 1로 표준화하는 도구.

    신경망은 입력/출력의 스케일이 비슷할 때 잘 학습되므로,
    학습 전에 입력과 출력을 각각 표준화한다.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        # 표준편차가 0인 열은 1로 두어 0으로 나누는 것을 막는다.
        self.std = np.where(self.std == 0.0, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("fit을 먼저 호출해야 한다.")
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("fit을 먼저 호출해야 한다.")
        return np.asarray(Xs, dtype=float) * self.std + self.mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
