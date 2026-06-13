"""
Strong-constraint 4D-Var.

이 파일은 Lorenz-63 모델에 대한 strong-constraint 4D-Var의 cost function과
adjoint 기반 gradient를 구현한다.

control variable은 assimilation window 시작 시각의 상태 x0이다.
모델은 완벽하다고 가정하므로(strong constraint), x0만 정하면 window 전체 궤적이 결정된다.

Cost function:

    J(x0) = 1/2 (x0 - x0_b)^T B^-1 (x0 - x0_b)
            + 1/2 sum_k (H x_k - y_k)^T R^-1 (H x_k - y_k)

여기서 x_k는 x0에서 출발한 forward 모델 궤적의 관측 시각 상태이다.

Gradient는 adjoint model로 계산한다.

    grad J(x0) = B^-1 (x0 - x0_b) + lambda_0

lambda_0는 관측 잔차를 forcing으로 주입한 adjoint 적분의 시작 시각 값이다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lorenz_da.adjoint.lorenz63_adjoint import euler_adjoint_matrix
from lorenz_da.models.lorenz63 import lorenz63_rhs
from lorenz_da.numerics.euler import integrate_euler


def _check_inputs(
    x0: np.ndarray,
    x0_b: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    observations: np.ndarray,
    obs_indices: np.ndarray,
    nsteps: int,
) -> None:
    if x0.shape != x0_b.shape:
        raise ValueError(
            f"x0와 x0_b의 shape이 같아야 한다. {x0.shape} vs {x0_b.shape}"
        )

    if observations.shape[0] != len(obs_indices):
        raise ValueError(
            "observations의 개수와 obs_indices의 개수가 같아야 한다. "
            f"{observations.shape[0]} vs {len(obs_indices)}"
        )

    if np.any(obs_indices < 0) or np.any(obs_indices > nsteps):
        raise ValueError("obs_indices가 [0, nsteps] 범위를 벗어났다.")

    if H.shape[0] != R.shape[0]:
        raise ValueError(
            f"H의 행 수와 R의 크기가 맞지 않는다. H: {H.shape}, R: {R.shape}"
        )


def four_dvar_cost(
    x0: np.ndarray,
    x0_b: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    observations: np.ndarray,
    obs_indices: np.ndarray,
    dt: float,
    nsteps: int,
    rhs_func: Callable[[np.ndarray], np.ndarray] = lorenz63_rhs,
) -> float:
    """
    4D-Var cost function 값을 계산한다.

    Parameters
    ----------
    x0 : np.ndarray, shape (state_dim,)
        control variable (window 시작 상태).
    x0_b : np.ndarray, shape (state_dim,)
        background 상태.
    B : np.ndarray, shape (state_dim, state_dim)
        background error covariance.
    R : np.ndarray, shape (nobs, nobs)
        관측오차 공분산.
    H : np.ndarray, shape (nobs, state_dim)
        observation operator.
    observations : np.ndarray, shape (n_obstimes, nobs)
        관측값. observations[k]는 obs_indices[k] 시각의 관측.
    obs_indices : np.ndarray
        관측이 존재하는 time step index (0..nsteps).
    dt : float
        시간간격.
    nsteps : int
        window 내 time step 수.
    rhs_func : callable
        모델 RHS 함수.

    Returns
    -------
    J : float
        cost function 값.
    """
    x0 = np.asarray(x0, dtype=float)
    x0_b = np.asarray(x0_b, dtype=float)
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)
    H = np.asarray(H, dtype=float)
    observations = np.asarray(observations, dtype=float)
    obs_indices = np.asarray(obs_indices, dtype=int)

    _check_inputs(x0, x0_b, B, R, H, observations, obs_indices, nsteps)

    traj = integrate_euler(x0=x0, dt=dt, nsteps=nsteps, rhs_func=rhs_func)

    dx = x0 - x0_b
    cost_b = 0.5 * float(dx @ np.linalg.solve(B, dx))

    cost_o = 0.0
    for k, idx in enumerate(obs_indices):
        d = H @ traj[idx] - observations[k]
        cost_o += 0.5 * float(d @ np.linalg.solve(R, d))

    return cost_b + cost_o


def four_dvar_cost_and_grad(
    x0: np.ndarray,
    x0_b: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    observations: np.ndarray,
    obs_indices: np.ndarray,
    dt: float,
    nsteps: int,
    rhs_func: Callable[[np.ndarray], np.ndarray] = lorenz63_rhs,
) -> tuple[float, np.ndarray]:
    """
    4D-Var cost function 값과 adjoint gradient를 함께 계산한다.

    forward 적분을 한 번만 수행하고, 그 궤적을 따라 adjoint를 backward로 적분한다.
    scipy.optimize.minimize에 jac=True로 넘기기 편하도록 (J, grad) 튜플을 반환한다.

    Gradient 유도
    -------------
    관측항만 보면, 시각 n에 대한 adjoint variable은

        lambda_n = (obs forcing at n) + M_n^T lambda_{n+1}

    이고, obs forcing은 관측 시각에서만

        H^T R^-1 (H x_n - y)

    이다. M_n = I + dt J(x_n)이고, 그 transpose가 한 step adjoint이다.
    최종적으로

        grad J = B^-1 (x0 - x0_b) + lambda_0

    이다.

    Returns
    -------
    J : float
        cost function 값.
    grad : np.ndarray, shape (state_dim,)
        cost function의 gradient.
    """
    x0 = np.asarray(x0, dtype=float)
    x0_b = np.asarray(x0_b, dtype=float)
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)
    H = np.asarray(H, dtype=float)
    observations = np.asarray(observations, dtype=float)
    obs_indices = np.asarray(obs_indices, dtype=int)

    _check_inputs(x0, x0_b, B, R, H, observations, obs_indices, nsteps)

    state_dim = x0.shape[0]

    # forward 적분 (한 번)
    traj = integrate_euler(x0=x0, dt=dt, nsteps=nsteps, rhs_func=rhs_func)

    # background term
    dx = x0 - x0_b
    cost_b = 0.5 * float(dx @ np.linalg.solve(B, dx))
    grad_b = np.linalg.solve(B, dx)

    # observation term과 adjoint forcing 준비
    obs_forcing: dict[int, np.ndarray] = {}
    cost_o = 0.0
    for k, idx in enumerate(obs_indices):
        d = H @ traj[idx] - observations[k]
        cost_o += 0.5 * float(d @ np.linalg.solve(R, d))
        obs_forcing[int(idx)] = H.T @ np.linalg.solve(R, d)

    # adjoint backward 적분 (관측 잔차를 forcing으로 주입)
    adj = np.zeros(state_dim, dtype=float)
    for n in range(nsteps, -1, -1):
        if n in obs_forcing:
            adj = adj + obs_forcing[n]
        if n > 0:
            # lambda_{n-1} = M_{n-1}^T lambda_n
            MT = euler_adjoint_matrix(state=traj[n - 1], dt=dt)
            adj = MT @ adj

    grad = grad_b + adj

    return cost_b + cost_o, grad


def four_dvar_gradient(
    x0: np.ndarray,
    x0_b: np.ndarray,
    B: np.ndarray,
    R: np.ndarray,
    H: np.ndarray,
    observations: np.ndarray,
    obs_indices: np.ndarray,
    dt: float,
    nsteps: int,
    rhs_func: Callable[[np.ndarray], np.ndarray] = lorenz63_rhs,
) -> np.ndarray:
    """
    4D-Var gradient만 반환하는 편의 함수. 내부적으로 cost_and_grad를 호출한다.
    """
    _, grad = four_dvar_cost_and_grad(
        x0=x0,
        x0_b=x0_b,
        B=B,
        R=R,
        H=H,
        observations=observations,
        obs_indices=obs_indices,
        dt=dt,
        nsteps=nsteps,
        rhs_func=rhs_func,
    )
    return grad
