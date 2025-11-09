import numpy as np

from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.types import Tensor


def test_yeoh_zero_strain() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    F: Tensor = np.eye(3)

    Stress: Tensor = Yeoh(F, test_mat)

    assert np.isclose(Stress[0, 0], 0.0)


def test_yeoh_uniaxial_tension() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100.0]

    lam1: float = np.exp(1.0)
    lam2: float = 1 / np.sqrt(lam1)

    F: Tensor = np.diag([lam1, lam2, lam2])

    Stress: Tensor = Yeoh(F, test_mat)

    assert np.isclose(Stress[0, 0], 8.476, rtol=0.01)
