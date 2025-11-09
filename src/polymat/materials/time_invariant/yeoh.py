from numpy import dot, eye, trace
from numpy.linalg import det

from polymat.types import Scalar, Tensor


def Yeoh(F: Tensor, params: list[float]) -> Tensor:
    """
    Yeoh hyperelastic material model. 3D loading specified by deformation gradient tensor.

    Parameters
    ----------
    F : Tensor
        Deformation gradient tensor as np.ndarray of size (3x3)

    params : list[float]
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

    bstar: Tensor = J ** (-2 / 3) * dot(F, F.T)

    dev_bstar: Tensor = bstar - trace(bstar) / 3 * eye(3)

    I1s: Scalar = trace(bstar)

    Stress: Tensor = 2 / J * (
        C10 + 2 * C20 * (I1s - 3) + 3 * C30 * (I1s - 3) ** 2
    ) * dev_bstar + kappa * (J - 1) * eye(3)

    return Stress
