from numpy import dot, eye, sign, spacing, sqrt, tan, trace
from numpy.linalg import det

from polymat.types import Scalar, Tensor


def invLangevin(x: float) -> float:
    """
    Inverse of the Langevin function defined as L(x) = coth(x) - 1/x.

    Uses the Bergstrom approach.

    Parameters
    ----------
    x: float
        Input to the inverse function

    Returns
    -------
    y: float
        Result of the inverse function
    """
    eps: float = spacing(1.0)

    if x >= 1 - eps:
        x = 1 - eps

    if x <= -1 + eps:
        x = -1 + eps

    if abs(x) < 0.839:
        return 1.31435 * tan(1.59 * x) + 0.911249 * x

    return 1.0 / (sign(x) - x)


def EightChain(F: Tensor, params: list[float]) -> Tensor:
    """
    Arruda-Boyce Eight-Chain hyperelastic material model.

    3D loading specified by deformation gradient tensor.

    Parameters
    ----------
    F : Tensor
        Deformation gradient tensor as np.ndarray of size (3x3)

    params : list[float]
        Material parameters [mu, lambdaL, kappa]

    Returns
    -------
    Tensor
        Cauchy stress tensor as np.ndarray of size (3x3)
    """
    mu: float = params[0]
    lambdaL: float = params[1]
    kappa: float = params[2]

    J: Scalar = det(F)

    bstar: Tensor = J ** (-2.0 / 3.0) * dot(F, F.T)

    dev_bstar: Tensor = bstar - trace(bstar) / 3.0 * eye(3)

    lam_chain: Scalar = sqrt(trace(bstar) / 3.0)

    Stress_dev: Tensor = (
        mu / (J * lam_chain) * invLangevin(lam_chain / lambdaL) / invLangevin(1.0 / lambdaL) * dev_bstar
    )

    Stress_vol: Tensor = kappa * (J - 1.0) * eye(3)

    return Stress_dev + Stress_vol
