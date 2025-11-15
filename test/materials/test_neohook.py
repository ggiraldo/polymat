import numpy as np

from polymat.materials.time_invariant.neo_hook import NeoHook
from polymat.mechanics.elastic_deformation import uniaxial_stress
from polymat.types import Tensor, Vector


def test_neo_hook_zero_strain() -> None:
    test_mat: list[float] = [1.0, 100.0]

    F: Tensor = np.eye(3)

    Stress: Tensor = NeoHook(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0, atol=1e-6)


def test_neo_hook_uniaxial_stress() -> None:
    test_mat: list[float] = [0.5838, 1e6]

    trueStrain: Vector = np.linspace(0.0, np.log(7.0), 100)

    trueStress: Vector = uniaxial_stress(NeoHook, trueStrain, test_mat)

    assert np.isclose(trueStress[-1], 29.02, rtol=0.02)
