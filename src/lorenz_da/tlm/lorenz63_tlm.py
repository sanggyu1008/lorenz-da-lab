"""
Lorenz-63 tangent linear model.

이 파일은 Lorenz-63 모델의 Jacobian과 Euler 시간적분에 대응하는
tangent linear model을 정의한다.

Nonlinear model:

    x_{n+1} = x_n + dt * f(x_n)

Euler step에 대한 tangent linear model:

    delta x_{n+1} = (I + dt * J(x_n)) delta x_n

여기서 J(x_n)는 Lorenz-63 RHS f의 Jacobian이다.
"""

from __future__ import annotations

import numpy as np


def lorenz63_jacobian(
    state: np.ndarray,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Lorenz-63 RHS의 Jacobian matrix를 계산한다.

    Lorenz-63 방정식은 다음과 같다.

        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    따라서 Jacobian J = df/dx는 다음과 같다.

        J =
        [ -sigma      sigma      0   ]
        [ rho - z     -1        -x   ]
        [ y            x        -beta]

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        현재 상태벡터 [x, y, z].
    sigma : float
        Lorenz-63 sigma 매개변수.
    rho : float
        Lorenz-63 rho 매개변수.
    beta : float
        Lorenz-63 beta 매개변수.

    Returns
    -------
    J : np.ndarray, shape (3, 3)
        Lorenz-63 RHS의 Jacobian matrix.
    """
    state = np.asarray(state, dtype=float)

    if state.shape != (3,):
        raise ValueError(
            f"Lorenz-63 state는 shape (3,)이어야 한다. 현재 shape: {state.shape}"
        )

    x, y, z = state

    J = np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ],
        dtype=float,
    )

    return J


def euler_tlm_matrix(
    state: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Euler step에 대한 tangent linear matrix를 계산한다.

    Nonlinear Euler step:

        x_{n+1} = x_n + dt * f(x_n)

    이에 대한 tangent linear model은 다음과 같다.

        delta x_{n+1} = (I + dt * J(x_n)) delta x_n

    Parameters
    ----------
    state : np.ndarray, shape (3,)
        기준 trajectory의 현재 상태벡터.
    dt : float
        시간간격.
    sigma, rho, beta : float
        Lorenz-63 매개변수.

    Returns
    -------
    M : np.ndarray, shape (3, 3)
        Euler step에 대한 tangent linear matrix.
    """
    if dt <= 0:
        raise ValueError(f"dt는 양수여야 한다. 현재 dt: {dt}")

    J = lorenz63_jacobian(
        state=state,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    I = np.eye(3, dtype=float)
    M = I + dt * J

    return M


def euler_tlm_step(
    perturbation: np.ndarray,
    state: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Euler TLM으로 perturbation을 한 step 전진시킨다.

    Parameters
    ----------
    perturbation : np.ndarray, shape (3,)
        현재 perturbation 벡터.
    state : np.ndarray, shape (3,)
        기준 trajectory의 현재 상태벡터.
    dt : float
        시간간격.
    sigma, rho, beta : float
        Lorenz-63 매개변수.

    Returns
    -------
    next_perturbation : np.ndarray, shape (3,)
        다음 시점의 perturbation 벡터.
    """
    perturbation = np.asarray(perturbation, dtype=float)

    if perturbation.shape != (3,):
        raise ValueError(
            "perturbation은 shape (3,)이어야 한다. "
            f"현재 shape: {perturbation.shape}"
        )

    M = euler_tlm_matrix(
        state=state,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    return M @ perturbation


def integrate_euler_tlm(
    perturbation0: np.ndarray,
    reference_trajectory: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    기준 trajectory를 따라 Euler TLM을 적분한다.

    Parameters
    ----------
    perturbation0 : np.ndarray, shape (3,)
        초기 perturbation.
    reference_trajectory : np.ndarray, shape (nt, 3)
        nonlinear reference trajectory.
    dt : float
        시간간격.
    sigma, rho, beta : float
        Lorenz-63 매개변수.

    Returns
    -------
    perturbations : np.ndarray, shape (nt, 3)
        각 시간의 TLM perturbation.
    """
    perturbation0 = np.asarray(perturbation0, dtype=float)
    reference_trajectory = np.asarray(reference_trajectory, dtype=float)

    if perturbation0.shape != (3,):
        raise ValueError(
            f"perturbation0은 shape (3,)이어야 한다. 현재 shape: {perturbation0.shape}"
        )

    if reference_trajectory.ndim != 2 or reference_trajectory.shape[1] != 3:
        raise ValueError(
            "reference_trajectory는 shape (nt, 3)이어야 한다. "
            f"현재 shape: {reference_trajectory.shape}"
        )

    nt = reference_trajectory.shape[0]
    perturbations = np.zeros((nt, 3), dtype=float)
    perturbations[0] = perturbation0

    for n in range(nt - 1):
        perturbations[n + 1] = euler_tlm_step(
            perturbation=perturbations[n],
            state=reference_trajectory[n],
            dt=dt,
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

    return perturbations