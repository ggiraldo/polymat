from numpy import isclose, linspace

from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import biaxial_stress, planar_stress, uniaxial_stress
from polymat.types import Vector


def test_uniaxial_stress_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = uniaxial_stress(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(8.1, trueStress[-1], rtol=0.01)


def test_biaxial_stress_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = biaxial_stress(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(7.93, trueStress[-1], rtol=0.01)


def test_planar_stress_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = planar_stress(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], atol=1e-12)

    assert isclose(8.46, trueStress[-1], rtol=0.01)
