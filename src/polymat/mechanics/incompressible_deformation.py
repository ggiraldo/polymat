from numpy import append, diag, exp, sqrt, zeros_like

from polymat.types import ElasticModel, Scalar, Tensor, Vector


def uniaxial_stress_incompressible(model: ElasticModel, trueStrain: Vector, params: Vector) -> Vector:
    """
    Incompresssible uniaxial loading.

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Uniaxial true strain history

    params : Vector
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Uniaxial true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    _params: Vector = append(params, 0.0)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)
        lam2: Scalar = 1 / sqrt(lam1)

        F: Tensor = diag([lam1, lam2, lam2])

        stress[i] = model(F, _params)[0, 0]

    return stress


def biaxial_stress_incompressible(model: ElasticModel, trueStrain: Vector, params: Vector) -> Vector:
    """
    Icompresssible equi-biaxial loading.

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Principal true strain history

    params : Vector
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Principal true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    _params: Vector = append(params, 0.0)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)
        lam3: Scalar = 1 / lam1**2

        F: Tensor = diag([lam1, lam1, lam3])

        stress[i] = model(F, _params)[0, 0]

    return stress


def planar_stress_incompressible(model: ElasticModel, trueStrain: Vector, params: Vector) -> Vector:
    """
    Incompresssible planar loading (pure shear).

    Parameters
    ----------
    model : ElasticModel
        Hyperelastic material model

    trueStrain : Vector
        Principal true strain history

    params : Vector
        Material parameters to be passed to model

    Returns
    -------
    trueStress : Vector
        Principal true stress history
    """
    stress: Vector = zeros_like(trueStrain)

    _params: Vector = append(params, 0.0)

    for i, strain in enumerate(trueStrain):
        lam1: Scalar = exp(strain)

        lam3: Scalar = 1 / lam1

        F: Tensor = diag([lam1, 1.0, lam3])

        stress[i] = model(F, _params)[0, 0]

    return stress
