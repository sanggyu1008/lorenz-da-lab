"""
Lorenz-63 adjoint model.

이 파일은 Euler 시간적분에 대응하는 Lorenz-63 tangent linear model의
adjoint model을 정의한다.

Nonlinear Euler model:

    x_{n+1} = x_n + dt * f(x_n)

Tangent linear model:

    delta x_{n+1} = M_n delta x_n

where

    M_n = I + dt * J(x_n)

Adjoint model:

    lambda_n = M_n^T lambda_{n+1}

여기서 adjoint variable lambda는 시간에 대해 backward 방향으로 전파된다.
"""

from __future__ import annotations

import numpy as np

from lorenz_da.tlm.lorenz63_tlm import euler_tlm_matrix


def euler_adjoint_matrix(
    state: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Euler TLM matrix의 adjoint matrix를 계산한다.

    Euler TLM matrix는 다음과 같다.

        M_n = I + dt * J(x_n)

    이에 대응하는 adjoint matrix는 다음과 같다.

        M_n^T

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
    MT : np.ndarray, shape (3, 3)
        Euler TLM matrix의 transpose.
    """
    M = euler_tlm_matrix(
        state=state,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    return M.T


def euler_adjoint_step(
    adjoint_next: np.ndarray,
    state: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    Euler adjoint model을 한 step backward로 전파한다.

    TLM:

        delta x_{n+1} = M_n delta x_n

    Adjoint:

        lambda_n = M_n^T lambda_{n+1}

    Parameters
    ----------
    adjoint_next : np.ndarray, shape (3,)
        다음 시점의 adjoint variable lambda_{n+1}.
    state : np.ndarray, shape (3,)
        기준 trajectory의 현재 상태 x_n.
    dt : float
        시간간격.
    sigma, rho, beta : float
        Lorenz-63 매개변수.

    Returns
    -------
    adjoint_current : np.ndarray, shape (3,)
        현재 시점의 adjoint variable lambda_n.
    """
    adjoint_next = np.asarray(adjoint_next, dtype=float)

    if adjoint_next.shape != (3,):
        raise ValueError(
            "adjoint_next는 shape (3,)이어야 한다. "
            f"현재 shape: {adjoint_next.shape}"
        )

    MT = euler_adjoint_matrix(
        state=state,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    return MT @ adjoint_next


def integrate_euler_adjoint(
    terminal_adjoint: np.ndarray,
    reference_trajectory: np.ndarray,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    """
    기준 trajectory를 따라 Euler adjoint model을 backward 적분한다.

    Parameters
    ----------
    terminal_adjoint : np.ndarray, shape (3,)
        마지막 시점의 adjoint variable lambda_N.
    reference_trajectory : np.ndarray, shape (nt, 3)
        nonlinear reference trajectory.
        TLM matrix M_n은 reference_trajectory[n]에서 계산된다.
    dt : float
        시간간격.
    sigma, rho, beta : float
        Lorenz-63 매개변수.

    Returns
    -------
    adjoints : np.ndarray, shape (nt, 3)
        각 시점의 adjoint variable.
        adjoints[-1] = terminal_adjoint.
    """
    terminal_adjoint = np.asarray(terminal_adjoint, dtype=float)
    reference_trajectory = np.asarray(reference_trajectory, dtype=float)

    if terminal_adjoint.shape != (3,):
        raise ValueError(
            "terminal_adjoint는 shape (3,)이어야 한다. "
            f"현재 shape: {terminal_adjoint.shape}"
        )

    if reference_trajectory.ndim != 2 or reference_trajectory.shape[1] != 3:
        raise ValueError(
            "reference_trajectory는 shape (nt, 3)이어야 한다. "
            f"현재 shape: {reference_trajectory.shape}"
        )

    nt = reference_trajectory.shape[0]

    adjoints = np.zeros((nt, 3), dtype=float)
    adjoints[-1] = terminal_adjoint

    for n in range(nt - 2, -1, -1):
        adjoints[n] = euler_adjoint_step(
            adjoint_next=adjoints[n + 1],
            state=reference_trajectory[n],
            dt=dt,
            sigma=sigma,
            rho=rho,
            beta=beta,
        )

    return adjoints


def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터의 inner product를 계산한다.

    Parameters
    ----------
    a, b : np.ndarray
        같은 shape의 벡터.

    Returns
    -------
    value : float
        a dot b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"a와 b의 shape이 같아야 한다. a shape: {a.shape}, b shape: {b.shape}"
        )

    return float(np.dot(a.ravel(), b.ravel()))


def relative_inner_product_error(lhs: float, rhs: float) -> float:
    """
    inner product test의 상대오차를 계산한다.

    Parameters
    ----------
    lhs : float
        왼쪽 inner product.
    rhs : float
        오른쪽 inner product.

    Returns
    -------
    rel_error : float
        상대오차.
    """
    denom = max(abs(lhs), abs(rhs), 1.0e-30)
    return abs(lhs - rhs) / denom