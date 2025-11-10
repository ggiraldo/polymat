import matplotlib.pyplot as plt
import numpy as np

from polymat.materials.time_invariant.eight_chain import EightChain
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.deformation import uniaxial_tension
from polymat.types import Vector


def test_plot_yeoh() -> None:
    test_mat: list[float] = [1.0, -0.01, 1e-4, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = uniaxial_tension(Yeoh, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)

    plt.title("Yeoh model")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)

    plt.show()


def test_plot_eight_chain() -> None:
    test_mat: list[float] = [1.0, 3.0, 100]

    trueStrain: Vector = np.linspace(0.0, 0.8, 100)

    trueStress: Vector = uniaxial_tension(EightChain, trueStrain, test_mat)

    plt.figure()

    plt.plot(trueStrain, trueStress)

    plt.title("Arruda-Boyce EC model")
    plt.xlabel("True Strain")
    plt.ylabel("True Stress")
    plt.grid(True, which="both", alpha=0.4)

    plt.show()
