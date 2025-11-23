import matplotlib.pyplot as plt
import numpy as np

from polymat.calibration.solvers.elastic_fit import fit_elastic_material
from polymat.materials.time_invariant.eight_chain import EightChain
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import biaxial_stress, planar_stress, uniaxial_stress
from polymat.mechanics.incompressible_deformation import uniaxial_stress_incompressible
from polymat.types import Vector


def test_plot_yeoh_uniaxial() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = uniaxial_stress(Yeoh, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)
    plt.title("Yeoh model - Uniaxial stress")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)

    plt.show()


def test_plot_yeoh_biaxial() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = biaxial_stress(Yeoh, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)
    plt.title("Yeoh model - Biaxial stress")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)

    plt.show()


def test_plot_yeoh_planar() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = planar_stress(Yeoh, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)
    plt.title("Yeoh model - Planar stress")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)

    plt.show()


def test_plot_eight_chain() -> None:
    test_mat: list[float] = [1.0, 3.0, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = uniaxial_stress(EightChain, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)
    plt.title("Arruda-Boyce EC model")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)
    plt.show()


def test_fit_plot() -> None:
    test_material: list[float] = [0.5, -1e-2, 1e-4]

    trueStrain: Vector = np.linspace(0.0, 1.0, 20)
    trueStress: Vector = uniaxial_stress_incompressible(Yeoh, trueStrain, test_material)

    calibrated_params, calibration_error = fit_elastic_material(
        strain=[trueStrain],
        stress=[trueStress],
        elastic_model=Yeoh,
        deformation_mode=[uniaxial_stress_incompressible],
        lower_bound=[-1e4, -1e2, -1.0],
        upper_bound=[1e4, 1e2, 1.0],
    )

    predictedStress: Vector = uniaxial_stress_incompressible(Yeoh, trueStrain, calibrated_params)

    np.set_printoptions(precision=4)

    plt.figure()

    plt.plot(trueStrain, trueStress, "go", label="Test")
    plt.plot(trueStrain, predictedStress, "r", label="Prediction")
    plt.title("Yeoh model")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)
    plt.text(0.8, 0.0, f"Params: {calibrated_params} \nError: {calibration_error:.1f}%")
    plt.legend()

    plt.show()
