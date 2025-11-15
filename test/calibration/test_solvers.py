from numpy import isclose, linspace
from pytest import mark

from polymat.calibration.solvers import single_test_elastic_simplex
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import uniaxial_stress
from polymat.types import Vector


@mark.xfail
def test_single_test_elastic_simplex() -> None:
    test_material: list[float] = [2.0, -0.016, 3e-4, 333.0]
    trueStrain: Vector = linspace(0.0, 0.5, 20)
    trueStress: Vector = uniaxial_stress(Yeoh, trueStrain, test_material)

    calibrated_params, calibration_error = single_test_elastic_simplex(
        strain=trueStrain,
        stress=trueStress,
        elastic_model=Yeoh,
        deformation_mode=uniaxial_stress,
        initial_guess=[1.0, -0.01, 1e-4, 100.0],
    )

    print(f"{calibrated_params=}")
    print(f"{calibration_error=}")

    assert all(isclose(calibrated_params, test_material, rtol=0.05))
