from numpy import isclose, linspace

from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.incompressible_deformation import (
    biaxial_stress_incompressible,
    planar_stress_incompressible,
    uniaxial_stress_incompressible,
)
from polymat.types import Vector


def test_uniaxial_stress_incompressible_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = uniaxial_stress_incompressible(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(5.7, trueStress[-1], rtol=0.01)


def test_biaxial_stress_incompressible_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = biaxial_stress_incompressible(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(2.87, trueStress[-1], rtol=0.01)


def test_planar_stress_incompressible_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = planar_stress_incompressible(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(5.45, trueStress[-1], rtol=0.01)
