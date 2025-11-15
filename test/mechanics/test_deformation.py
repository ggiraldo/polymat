from numpy import isclose, linspace

from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import uniaxial_stress
from polymat.types import Vector


def test_uniaxial_stress_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    trueStrain: Vector = linspace(0, 0.8, 100)

    trueStress: Vector = uniaxial_stress(Yeoh, trueStrain, test_mat)

    assert isclose(0.0, trueStress[0], rtol=0.01)

    assert isclose(8.1, trueStress[-1], rtol=0.01)
