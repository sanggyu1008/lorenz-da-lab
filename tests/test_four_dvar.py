"""
4D-Var cost/gradient 검증.

핵심: adjoint로 계산한 gradient가 cost function의 유한차분(finite difference)
gradient와 일치하는지 확인한다. 이것이 4D-Var 구현의 정확성을 보장하는 표준 검증이다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.assimilation.four_dvar import (
    four_dvar_cost,
    four_dvar_cost_and_grad,
)
from lorenz_da.models.lorenz63 import lorenz63_default_initial_condition
from lorenz_da.numerics.euler import integrate_euler
from lorenz_da.observations.operator import identity_observation_operator
from lorenz_da.observations.synthetic import (
    generate_synthetic_observations,
    make_observation_indices,
)
from lorenz_da.models.lorenz63 import lorenz63_rhs


def _setup_window(seed=0):
    rng = np.random.default_rng(seed)

    dt = 0.01
    nsteps = 100
    obs_interval = 10

    x0_true = lorenz63_default_initial_condition()
    truth = integrate_euler(x0=x0_true, dt=dt, nsteps=nsteps, rhs_func=lorenz63_rhs)

    obs_indices = make_observation_indices(nsteps=nsteps, obs_interval=obs_interval)
    H = identity_observation_operator(3)
    obs_std = 1.0
    R = (obs_std**2) * np.eye(3)
    observations = generate_synthetic_observations(
        truth=truth, obs_indices=obs_indices, H=H, obs_std=obs_std, rng=rng
    )

    x0_b = x0_true + np.array([0.5, -0.3, 0.4])
    B = (2.0**2) * np.eye(3)

    return dict(
        dt=dt,
        nsteps=nsteps,
        obs_indices=obs_indices,
        H=H,
        R=R,
        B=B,
        observations=observations,
        x0_b=x0_b,
        x0_true=x0_true,
    )


def test_adjoint_gradient_matches_finite_difference():
    cfg = _setup_window()

    # gradient를 평가할 지점 (background에서 약간 벗어난 점)
    x0 = cfg["x0_b"] + np.array([0.1, 0.2, -0.1])

    def cost(z):
        return four_dvar_cost(
            x0=z,
            x0_b=cfg["x0_b"],
            B=cfg["B"],
            R=cfg["R"],
            H=cfg["H"],
            observations=cfg["observations"],
            obs_indices=cfg["obs_indices"],
            dt=cfg["dt"],
            nsteps=cfg["nsteps"],
        )

    _, grad_adjoint = four_dvar_cost_and_grad(
        x0=x0,
        x0_b=cfg["x0_b"],
        B=cfg["B"],
        R=cfg["R"],
        H=cfg["H"],
        observations=cfg["observations"],
        obs_indices=cfg["obs_indices"],
        dt=cfg["dt"],
        nsteps=cfg["nsteps"],
    )

    # 중심 차분으로 수치 gradient 계산
    eps = 1e-6
    grad_fd = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        grad_fd[i] = (cost(x0 + e) - cost(x0 - e)) / (2 * eps)

    rel_error = np.linalg.norm(grad_adjoint - grad_fd) / np.linalg.norm(grad_fd)
    assert rel_error < 1e-6, f"gradient 상대오차가 너무 크다: {rel_error}"


def test_cost_is_nonnegative_and_grad_zero_at_perfect_obs():
    """관측이 truth 그대로이고 x0 = truth라면 관측항은 0이고, grad는 background항만 남는다."""
    cfg = _setup_window()
    x0_true = cfg["x0_true"]

    # 관측을 noise 없이 truth로 다시 생성
    truth = integrate_euler(
        x0=x0_true, dt=cfg["dt"], nsteps=cfg["nsteps"], rhs_func=lorenz63_rhs
    )
    obs_clean = np.array([truth[idx] for idx in cfg["obs_indices"]])

    J, grad = four_dvar_cost_and_grad(
        x0=x0_true,
        x0_b=cfg["x0_b"],
        B=cfg["B"],
        R=cfg["R"],
        H=cfg["H"],
        observations=obs_clean,
        obs_indices=cfg["obs_indices"],
        dt=cfg["dt"],
        nsteps=cfg["nsteps"],
    )

    # 관측항은 0이므로 cost는 background항뿐
    dx = x0_true - cfg["x0_b"]
    expected_J = 0.5 * dx @ np.linalg.solve(cfg["B"], dx)
    np.testing.assert_allclose(J, expected_J, rtol=1e-10)

    # grad도 background항만 남아야 한다
    expected_grad = np.linalg.solve(cfg["B"], dx)
    np.testing.assert_allclose(grad, expected_grad, atol=1e-10)
