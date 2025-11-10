import numpy as np

from polymat.materials.time_invariant.eight_chain import EightChain
from polymat.types import Tensor


def test_ec_zero_strain() -> None:
    test_mat: list[float] = [1.0, 3.0, 100.0]

    F: Tensor = np.eye(3)

    Stress: Tensor = EightChain(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0)


def test_ec_uniaxial_tension() -> None:
    test_mat: list[float] = [1.0, 3.0, 100.0]

    lam1: float = np.exp(0.8)
    lam2: float = 1 / np.sqrt(lam1)

    F: Tensor = np.diag([lam1, lam2, lam2])

    Stress: Tensor = EightChain(F, test_mat)

    assert np.isclose(Stress[0, 0], 3.25, rtol=0.01)
