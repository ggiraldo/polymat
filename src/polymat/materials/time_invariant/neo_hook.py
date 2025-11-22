from numpy import dot, eye, trace
from numpy.linalg import det

from polymat.types import Tensor, Vector


def NeoHook(F: Tensor, params: Vector) -> Tensor:
    """
    Neo-Hookean hyperelastic material model.

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor of shape (3,3)

    params: Vector
        Material parameters [mu, kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor.
    """
    mu: float = params[0]
    kappa: float = params[1]

    J: float = det(F)

    Fstar: Tensor = J ** (-1.0 / 3.0) * F

    bstar: Tensor = dot(Fstar, Fstar.T)

    dev_bstar: Tensor = bstar - trace(bstar) / 3.0 * eye(3)

    Stress_dev: Tensor = mu / J * dev_bstar

    Stress_vol: Tensor = kappa * (J - 1.0) * eye(3)

    return Stress_dev + Stress_vol
