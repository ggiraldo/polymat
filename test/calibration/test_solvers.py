from numpy import isclose, linspace
from pytest import mark

from polymat.calibration.solvers.single_test import fit_single_test_elastic, fit_single_test_incompressible
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import uniaxial_stress
from polymat.mechanics.incompressible_deformation import uniaxial_stress_incompressible
from polymat.types import Vector


@mark.xfail
def test_fit_single_test_elastic() -> None:
    test_material: list[float] = [2.0, -0.016, 3e-4, 333.0]
    trueStrain: Vector = linspace(0.0, 0.5, 20)
    trueStress: Vector = uniaxial_stress(Yeoh, trueStrain, test_material)

    calibrated_params, calibration_error = fit_single_test_elastic(
        strain=trueStrain,
        stress=trueStress,
        elastic_model=Yeoh,
        deformation_mode=uniaxial_stress,
        initial_guess=[1.0, -0.01, 1e-4, 100.0],
    )

    print(f"{calibrated_params=}")
    print(f"{calibration_error=}")

    assert all(isclose(calibrated_params, test_material, rtol=0.1))


def test_fit_single_test_incompressible() -> None:
    test_material: list[float] = [2.0, -0.016, 3e-4]
    trueStrain: Vector = linspace(0.0, 0.5, 50)
    trueStress: Vector = uniaxial_stress_incompressible(Yeoh, trueStrain, test_material)

    calibrated_params, calibration_error = fit_single_test_incompressible(
        strain=trueStrain,
        stress=trueStress,
        elastic_model=Yeoh,
        deformation_mode=uniaxial_stress_incompressible,
        initial_guess=[1.0, -0.01, 1e-4],
    )

    print(f"{calibrated_params=}")
    print(f"{calibration_error=}")

    assert all(isclose(calibrated_params, test_material, rtol=0.01))
