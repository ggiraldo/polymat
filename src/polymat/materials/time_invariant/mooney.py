from numpy import dot, eye, trace
from numpy.linalg import det

from polymat.types import Tensor, Vector


def Mooney5(F: Tensor, params: Vector) -> Tensor:
    """
    Mooney 5-terms polynomial hyperelastic material model.

    3D loading specified by deformation gradient F.

    Parameters
    ----------
    F: Tensor
        Deformation gradient tensor as array of size (3x3)

    params: Vector
        Material parameters [C10, C02, C11, C20, C30, kappa]

    Returns
    -------
    Stress: Tensor
        Cauchy stress tensor
    """
    C10: float = params[0]
    C01: float = params[1]
    C11: float = params[2]
    C20: float = params[3]
    C30: float = params[4]
    kappa: float = params[5]

    J: float = det(F)

    Fstar: Tensor = J ** (-1.0 / 3.0) * F

    bstar: Tensor = dot(Fstar, Fstar.T)

    bstar2: Tensor = dot(bstar, bstar)

    I1star: float = trace(bstar)

    I2star: float = 0.5 * (I1star**2.0 - trace(bstar2))

    dPhi_dI1star: float = C10 + C11 * (I2star - 3.0) + 2.0 * C20 * (I1star - 3.0) + 3.0 * C30 * (I1star - 3.0) ** 2.0

    dPhi_dI2star: float = C01 + C11 * (I1star - 3.0)

    Stress_dev: Tensor = 2.0 / J * (dPhi_dI1star + dPhi_dI2star * I1star) * bstar

    Stress_dev += -2.0 / J * dPhi_dI2star * bstar2

    Stress_dev += -2.0 / (3.0 * J) * (I1star * dPhi_dI1star + 2.0 * I2star * dPhi_dI2star) * eye(3)

    Stress_vol: Tensor = 3.0 * kappa * (J - 1.0) * eye(3)

    return Stress_dev + Stress_vol
