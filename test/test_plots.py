import matplotlib.pyplot as plt
import numpy as np

from polymat.materials.time_invariant.eight_chain import EightChain
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import biaxial_stress, planar_stress, uniaxial_stress
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
