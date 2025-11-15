from numpy import array, isclose

from polymat.calibration.error_measures import error_re
from polymat.calibration.fitness_functions import fitness_single_test_elastic
from polymat.materials.time_invariant.yeoh import Yeoh
from polymat.mechanics.elastic_deformation import uniaxial_stress


def test_fitness_single_test_elastic() -> None:
    err: float = fitness_single_test_elastic(
        params=[1.0, -0.01, 1e-4, 100.0],
        strain=array(
            [
                0.0,
                0.05555556,
                0.11111111,
                0.16666667,
                0.22222222,
                0.27777778,
                0.33333333,
                0.38888889,
                0.44444444,
                0.5,
            ]
        ),
        stress=array(
            [
                0.01,
                0.35,
                0.69,
                1.09,
                1.48,
                1.92,
                2.41,
                2.91,
                3.44,
                4.05,
            ]
        ),
        elastic_model=Yeoh,
        deformation_mode=uniaxial_stress,
        error_measure=error_re,
    )

    assert isclose(err, 0.0, atol=1.0)
