import numpy as np

from polymat.materials.time_invariant.mooney import Mooney5
from polymat.mechanics.elastic_deformation import uniaxial_tension
from polymat.types import Tensor, Vector


def test_mooney_zero_strain() -> None:
    test_mat: list[float] = [0.175, 0.0, 0.0, -1.35e-3, 3.9e-5, 2e3]

    F: Tensor = np.eye(3)

    Stress: Tensor = Mooney5(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0, atol=1e-6)


def test_mooney_uniaxial_tension() -> None:
    test_mat: list[float] = [0.175, 0.0, 0.0, -1.35e-3, 3.9e-5, 1e3]

    trueStrain: Vector = np.linspace(0.0, np.log(7.0), 100)

    trueStress: Vector = uniaxial_tension(Mooney5, trueStrain, test_mat)

    assert np.isclose(trueStress[-1], 29.02, rtol=0.02)
