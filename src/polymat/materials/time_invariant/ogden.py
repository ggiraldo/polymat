from numpy import dot, eye, sqrt
from numpy.linalg import det, eig, multi_dot

from polymat.types import Tensor, Vector


def Ogden(F: Tensor, param: Vector) -> Tensor:
    """
    Ogden hyperelastic material model.

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor of shape (3,3)

    params: Vector
        Material parameters [mu1, m2, ..., alpha1, alpha2, ..., kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor.
    """
    N: int = int((len(param) - 1) / 2)
    mu: Vector = param[0:N]
    alpha: Vector = param[N : 2 * N]
    kappa: float = param[-1]

    J: float = det(F)

    b: Tensor = dot(F, F.T)

    lam2, Q = eig(b)

    lamstar: Vector = J ** (-1.0 / 3.0) * sqrt(lam2)

    StressP: Tensor = kappa * (J - 1) * eye(3)

    for i in range(N):
        fact: float = (2.0 / J) * mu[i] / alpha[i]

        p: float = (lamstar[0] ** alpha[i] + lamstar[1] ** alpha[i] + lamstar[2] ** alpha[i]) / 3

        StressP[0, 0] += fact * (lamstar[0] ** alpha[i] - p)

        StressP[1, 1] += fact * (lamstar[1] ** alpha[i] - p)

        StressP[2, 2] += fact * (lamstar[2] ** alpha[i] - p)

    Stress: Tensor = multi_dot((Q, StressP, Q.T))

    return Stress


def Ogden2(F: Tensor, params: Vector) -> Tensor:
    """
    Alternative version of Ogden hyperelastic material model (Ansys, Marc).

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor of shape (3,3)

    params: Vector
        Material parameters [mu1, mu2, ..., alpha1, alpha2, ..., kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor.
    """
    N: int = int((len(params) - 1) / 2)
    mu: Vector = params[0:N]
    alpha: Vector = params[N : 2 * N]
    kappa: float = params[-1]

    J: float = det(F)

    b: Tensor = dot(F, F.T)

    lam2, Q = eig(b)

    lam: Vector = sqrt(lam2)

    StressP: Tensor = 3 * kappa * (J ** (1 / 3) - 1) * J ** (-2 / 3) * eye(3)

    for n in range(N):
        fact: float = 1 / J * mu[n] * J ** (-alpha[n] / 3)

        p: float = (lam[0] ** alpha[n] + lam[1] ** alpha[n] + lam[2] ** alpha[n]) / 3

        StressP[0, 0] += fact * (lam[0] ** alpha[n] - p)

        StressP[1, 1] += fact * (lam[1] ** alpha[n] - p)

        StressP[2, 2] += fact * (lam[2] ** alpha[n] - p)

    Stress: Tensor = multi_dot((Q, StressP, Q.T))

    return Stress
