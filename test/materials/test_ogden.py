import numpy as np

from polymat.materials.time_invariant.ogden import Ogden, Ogden_Marc
from polymat.mechanics.elastic_deformation import uniaxial_stress
from polymat.types import Tensor, Vector


def test_ogden_zero_strain() -> None:
    test_mat: list[float] = [1.0, 2.0, 1.1, 0.4, 100.0]

    F: Tensor = np.eye(3)

    Stress: Tensor = Ogden(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0)


def test_ogden_uniaxial_stress() -> None:
    test_mat: list[float] = [1.0, 2.0, 1.1, 0.4, 100.0]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = uniaxial_stress(Ogden, trueStrain, test_mat)

    assert np.isclose(trueStress[-1], 8.13, rtol=0.02)


def test_ogden_marc_zero_strain() -> None:
    test_mat: list[float] = [1.0, 2.0, 1.1, 0.4, 100.0]

    F: Tensor = np.eye(3)

    Stress: Tensor = Ogden_Marc(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0)
