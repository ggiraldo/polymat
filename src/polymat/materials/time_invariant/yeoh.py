from numpy import dot, eye, trace
from numpy.linalg import det

from polymat.types import Scalar, Tensor, Vector


def Yeoh(F: Tensor, params: Vector) -> Tensor:
    """
    Yeoh hyperelastic material model. 3D loading specified by deformation gradient tensor.

    Parameters
    ----------
    F : Tensor
        Deformation gradient tensor as np.ndarray of size (3x3)

    params : Vector
        Material parameters [C10, C20, C30, kappa]

    Returns
    -------
    Tensor
        Cauchy stress tensor as np.ndarray of size (3x3)
    """
    C10: float = params[0]
    C20: float = params[1]
    C30: float = params[2]
    kappa: float = params[3]

    J: Scalar = det(F)

    bstar: Tensor = J ** (-2.0 / 3.0) * dot(F, F.T)

    dev_bstar: Tensor = bstar - trace(bstar) / 3.0 * eye(3)

    I1star: Scalar = trace(bstar)

    Stress_dev: Tensor = 2.0 / J * (C10 + 2.0 * C20 * (I1star - 3.0) + 3.0 * C30 * (I1star - 3) ** 2.0) * dev_bstar

    Stress_vol: Tensor = kappa * (J - 1.0) * eye(3)

    return Stress_dev + Stress_vol
