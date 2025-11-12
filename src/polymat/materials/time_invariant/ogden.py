from numpy import diag, dot, eye, sqrt
from numpy.linalg import det, eig, multi_dot

from polymat.types import Tensor, Vector


def Ogden(F: Tensor, param: list[float]) -> Tensor:
    """
    Ogden hyperelastic material model.

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor of shape (3,3)

    params: list[float]
        Material parameters [mu1, m2, ..., alpha1, alpha2, ..., kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor.
    """
    N: int = int((len(param) - 1) / 2)
    mu: list[float] = param[0:N]
    alpha: list[float] = param[N : 2 * N]
    kappa: float = param[-1]

    J: float = det(F)

    C = dot(F.T, F)

    lam2, Q = eig(C)

    lam: Vector = J ** (-1.0 / 3.0) * sqrt(lam2)

    StressP: Tensor = kappa * (J - 1) * eye(3)

    for i in range(N):
        fact: float = (2 / J) * mu[i] / alpha[i]

        p: float = (lam[0] ** alpha[i] + lam[1] ** alpha[i] + lam[2] ** alpha[i]) / 3

        StressP[0, 0] += fact * (lam[0] ** alpha[i] - p)

        StressP[1, 1] += fact * (lam[1] ** alpha[i] - p)

        StressP[2, 2] += fact * (lam[2] ** alpha[i] - p)

    Stress: Tensor = multi_dot((Q.T, StressP, Q))

    return Stress


def Ogden_Marc(F: Tensor, params: list[float]) -> Tensor:
    """
    Marc version of Ogden hyperelastic material model.

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor of shape (3,3)

    params: list[float]
        Material parameters [mu1, m2, ..., alpha1, alpha2, ..., kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor.
    """
    N: int = int((len(params) - 1) / 2)
    mu: list[float] = params[0:N]
    alpha: list[float] = params[N : 2 * N]
    kappa: float = params[-1]

    J: float = det(F)

    lam: Vector = diag(F)

    Stress: Tensor = 3 * kappa * (J ** (1 / 3) - 1) * J ** (-2 / 3) * eye(3)

    for n in range(N):
        a: float = 1 / J * mu[n] * J ** (-alpha[n] / 3)

        b: float = (lam[0] ** alpha[n] + lam[1] ** alpha[n] + lam[2] ** alpha[n]) / 3

        Stress[0, 0] += a * (lam[0] ** alpha[n] - b)

        Stress[1, 1] += a * (lam[1] ** alpha[n] - b)

        Stress[2, 2] += a * (lam[2] ** alpha[n] - b)

    return Stress
