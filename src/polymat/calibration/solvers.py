from polymat.calibration.fitness_functions import fitness_single_test_elastic
from polymat.types import Vector


def single_test_elastic() -> tuple[Vector, float]:
    """
    Solves optimization problem to calibrate the material model against the test data.

    Parameters
    ----------
    @ToDo

    Returns
    -------
    calibrated_params: Vector
        Found list of optimal material parameters

    error: float
        Computed error for the found solution
    """

    return [1.0, 2.0], 100.0
