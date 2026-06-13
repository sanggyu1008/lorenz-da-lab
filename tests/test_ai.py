"""
AI surrogate (numpy MLP) 검증.

1. Standardizer round-trip.
2. MLP forward shape.
3. backprop gradient가 유한차분과 일치(가장 중요한 정확성 검증).
4. 학습이 loss를 줄인다.
5. 학습된 surrogate가 Lorenz-63 한 step을 잘 예측한다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.ai.dataset import (
    Standardizer,
    make_increment_dataset,
)
from lorenz_da.ai.models import MLP
from lorenz_da.ai.train import (
    mse_loss_and_grads,
    surrogate_step,
    train,
)
from lorenz_da.models.lorenz63 import lorenz63_default_initial_condition, lorenz63_rhs
from lorenz_da.numerics.euler import integrate_euler


def test_standardizer_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(100, 3))
    s = Standardizer().fit(X)
    Xs = s.transform(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(s.inverse_transform(Xs), X, atol=1e-10)


def test_mlp_forward_shape():
    rng = np.random.default_rng(0)
    mlp = MLP([3, 8, 8, 3], rng=rng)
    out = mlp.forward(np.zeros((5, 3)))
    assert out.shape == (5, 3)


def test_mlp_backprop_matches_finite_difference():
    rng = np.random.default_rng(1)
    mlp = MLP([3, 5, 3], rng=rng)
    X = rng.normal(size=(4, 3))
    Y = rng.normal(size=(4, 3))

    _, grads = mse_loss_and_grads(mlp, X, Y)

    eps = 1e-6
    # 첫 번째 층 W의 몇 개 원소만 검사
    W = mlp.params[0][0]
    dW_analytic = grads[0][0]
    for (i, j) in [(0, 0), (1, 2), (2, 4)]:
        orig = W[i, j]
        W[i, j] = orig + eps
        loss_plus, _ = mse_loss_and_grads(mlp, X, Y)
        W[i, j] = orig - eps
        loss_minus, _ = mse_loss_and_grads(mlp, X, Y)
        W[i, j] = orig
        fd = (loss_plus - loss_minus) / (2 * eps)
        assert abs(fd - dW_analytic[i, j]) < 1e-7


def test_training_reduces_loss_and_predicts_lorenz63():
    rng = np.random.default_rng(42)

    # Lorenz-63 trajectory로 학습 데이터 생성
    dt = 0.01
    x0 = lorenz63_default_initial_condition()
    traj = integrate_euler(x0, dt=dt, nsteps=4000, rhs_func=lorenz63_rhs)

    X, dY = make_increment_dataset(traj)
    in_scaler = Standardizer().fit(X)
    out_scaler = Standardizer().fit(dY)
    Xs = in_scaler.transform(X)
    dYs = out_scaler.transform(dY)

    mlp = MLP([3, 32, 32, 3], rng=rng)
    history = train(mlp, Xs, dYs, epochs=200, lr=5e-3, batch_size=256, rng=rng)

    assert history[-1] < history[0]  # loss 감소
    assert history[-1] < 0.05  # 충분히 작은 loss

    # 한 step 예측이 실제 Euler step과 가깝다
    x_test = traj[100]
    x_next_true = traj[101]
    x_next_pred = surrogate_step(mlp, x_test, in_scaler, out_scaler)
    assert np.linalg.norm(x_next_pred - x_next_true) < 0.05
