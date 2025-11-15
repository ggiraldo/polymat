from numpy import abs, diag, exp, sqrt, zeros_like
from scipy.optimize import OptimizeResult, minimize

from polymat.types import ElasticModel, Scalar, Tensor, Vector


def uniaxial_stress(model: ElasticModel, trueStrain: Vector, params: list[float]) -> Vector:
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

    def S22abs(x: list[float]) -> Scalar:
        F: Tensor = diag([lam1, x[0], x[0]])
        S22: Scalar = model(F, params)[1, 1]
        return abs(S22)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)

        # Search numerically for the transverse stretch that makes S22 = 0
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


def biaxial_stress(model: ElasticModel, trueStrain: Vector, params: list[float]) -> Vector:
    """
    Compresssible equi-biaxial loading.

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Principal true strain history

    params : list[float]
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Principal true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    def S33abs(x: list[float]) -> Scalar:
        F: Tensor = diag([lam1, lam1, x[0]])
        S33: Scalar = model(F, params)[2, 2]
        return abs(S33)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)

        # Search numerically for the transverse strain that makes S33 = 0
        opt: OptimizeResult = minimize(
            fun=S33abs,
            x0=1 / lam1**2,
            method="Nelder-Mead",
            tol=1e-9,
        )

        lam3: Scalar = opt.x[0]

        F: Tensor = diag([lam1, lam1, lam3])

        stress[i] = model(F, params)[0, 0]

    return stress


def planar_stress(model: ElasticModel, trueStrain: Vector, params: list[float]) -> Vector:
    """
    Compresssible planar loading (pure shear).

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Principal true strain history

    params : list[float]
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Principal true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    def S33abs(x: list[float]) -> Scalar:
        F: Tensor = diag([lam1, 1.0, x[0]])
        S33: Scalar = model(F, params)[2, 2]
        return abs(S33)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)

        # Search numerically for the transverse stretch that makes S33 = 0
        opt: OptimizeResult = minimize(
            fun=S33abs,
            x0=1 / lam1,
            method="Nelder-Mead",
            tol=1e-9,
        )

        lam3: Scalar = opt.x[0]

        F: Tensor = diag([lam1, 1.0, lam3])

        stress[i] = model(F, params)[0, 0]

    return stress
