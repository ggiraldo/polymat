from numpy import abs, diag, exp, sqrt, zeros_like
from scipy.optimize import OptimizeResult, minimize

from polymat.types import ElasticModel, Scalar, Tensor, Vector


def uniaxial_tension(
    model: ElasticModel, trueStrain: Vector, params: list[float]
) -> Vector:
    """
    Compresssible uniaxial loading.

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Uniaxial true strain history

    params : list[float]
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Uniaxial true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    def S22abs(lam2: list[float]) -> Scalar:
        F: Tensor = diag([lam1, lam2[0], lam2[0]])
        S22: Scalar = model(F, params)[1, 1]
        return abs(S22)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)

        # Search numerically for the transverse strain that makes S22 = 0
        opt: OptimizeResult = minimize(
            fun=S22abs,
            x0=1 / sqrt(lam1),
            method="Nelder-Mead",
            tol=1e-9,
        )

        lam2: Scalar = opt.x[0]

        F: Tensor = diag([lam1, lam2, lam2])

        stress[i] = model(F, params)[0, 0]

    return stress
