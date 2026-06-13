"""
MLP surrogate model 학습과 예측.

학습은 mean squared error(MSE) loss를 backpropagation으로 미분하고,
Adam optimizer로 가중치를 갱신하는 방식으로 한다. 모두 numpy로 직접 구현한다.

surrogate는 "현재 상태 -> 상태 증분"을 표준화된 공간에서 예측하도록 학습한다.
예측 시에는 표준화를 역변환해 실제 증분을 얻고, 현재 상태에 더해 다음 상태를 만든다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.ai.dataset import Standardizer
from lorenz_da.ai.models import MLP


def mse_loss_and_grads(
    mlp: MLP,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    """
    MSE loss와 각 층 가중치에 대한 gradient를 backpropagation으로 계산한다.

    loss = mean over all elements of (pred - Y)^2

    Returns
    -------
    loss : float
    grads : list of (dW, db)
    """
    pred, caches = mlp.forward(X, return_cache=True)
    diff = pred - Y
    n_batch, n_out = Y.shape
    loss = float(np.mean(diff**2))

    # dLoss/dpred
    dout = (2.0 / (n_batch * n_out)) * diff

    grads: list[tuple[np.ndarray, np.ndarray]] = []
    da = dout
    n_layers = len(mlp.params)

    for i in reversed(range(n_layers)):
        a_in, z = caches[i]
        W, _ = mlp.params[i]
        if i < n_layers - 1:
            dz = da * (1.0 - np.tanh(z) ** 2)  # tanh 미분
        else:
            dz = da  # 선형 출력
        dW = a_in.T @ dz
        db = dz.sum(axis=0)
        grads.append((dW, db))
        da = dz @ W.T

    grads.reverse()
    return loss, grads


def train(
    mlp: MLP,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 256,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """
    Adam optimizer로 MLP를 학습한다.

    Parameters
    ----------
    mlp : MLP
        학습할 모델 (in-place로 갱신된다).
    X, Y : np.ndarray
        표준화된 입력과 목표값.
    epochs : int
        전체 데이터 반복 횟수.
    lr : float
        learning rate.
    batch_size : int
        mini-batch 크기.
    rng : np.random.Generator or None
        섞기에 쓸 난수 생성기.

    Returns
    -------
    history : list of float
        epoch마다의 전체 데이터 loss.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if rng is None:
        rng = np.random.default_rng()

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    mW = [np.zeros_like(W) for W, b in mlp.params]
    vW = [np.zeros_like(W) for W, b in mlp.params]
    mb = [np.zeros_like(b) for W, b in mlp.params]
    vb = [np.zeros_like(b) for W, b in mlp.params]

    n = X.shape[0]
    history: list[float] = []
    t = 0

    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            _, grads = mse_loss_and_grads(mlp, X[idx], Y[idx])
            t += 1
            for i, (dW, db) in enumerate(grads):
                mW[i] = beta1 * mW[i] + (1 - beta1) * dW
                vW[i] = beta2 * vW[i] + (1 - beta2) * dW**2
                mb[i] = beta1 * mb[i] + (1 - beta1) * db
                vb[i] = beta2 * vb[i] + (1 - beta2) * db**2

                mW_hat = mW[i] / (1 - beta1**t)
                vW_hat = vW[i] / (1 - beta2**t)
                mb_hat = mb[i] / (1 - beta1**t)
                vb_hat = vb[i] / (1 - beta2**t)

                mlp.params[i][0] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
                mlp.params[i][1] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        full_loss, _ = mse_loss_and_grads(mlp, X, Y)
        history.append(full_loss)

    return history


def surrogate_step(
    mlp: MLP,
    x: np.ndarray,
    in_scaler: Standardizer,
    out_scaler: Standardizer,
) -> np.ndarray:
    """
    학습된 surrogate로 한 step 예측한다.

    표준화된 입력 -> 표준화된 증분 -> 실제 증분 -> 다음 상태 순으로 계산한다.

        x_{n+1} = x_n + dy,   dy = out_scaler.inverse(mlp(in_scaler.transform(x_n)))
    """
    x = np.asarray(x, dtype=float)
    xs = in_scaler.transform(x[np.newaxis, :])
    dys = mlp.forward(xs)
    dy = out_scaler.inverse_transform(dys)[0]
    return x + dy


def surrogate_rollout(
    mlp: MLP,
    x0: np.ndarray,
    nsteps: int,
    in_scaler: Standardizer,
    out_scaler: Standardizer,
) -> np.ndarray:
    """
    surrogate를 반복 적용해 다중 step 예측(rollout)을 수행한다.

    한 step씩 예측한 결과를 다시 입력으로 넣으므로, 오차가 누적될 수 있다.

    Returns
    -------
    traj : np.ndarray, shape (nsteps + 1, state_dim)
    """
    x0 = np.asarray(x0, dtype=float)
    traj = np.zeros((nsteps + 1, x0.shape[0]), dtype=float)
    traj[0] = x0
    for n in range(nsteps):
        traj[n + 1] = surrogate_step(mlp, traj[n], in_scaler, out_scaler)
    return traj
