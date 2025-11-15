from polymat.types import ElasticDeformation, ElasticModel, ErrorMeasure, Vector


def fitness_single_test_elastic(
    params: list[float],
    strain: Vector,
    stress: Vector,
    elastic_model: ElasticModel,
    deformation_mode: ElasticDeformation,
    error_measure: ErrorMeasure,
) -> float:
    """
    Fitness fuction to measure error in stress prediction.

    Parameters
    ----------
    params: list[float]
        Test material parameters

    strain: Vector
        Experimental strain

    stress: Vector
        Experimental stress

    elastic_model: ElasticModel
        Hyperelastic stress computation function

    deformation_mode: ElasticDeformation
        Stress curve computation function

    error_measure: ErrorMeasure,
        Error metric for the fitness function

    Retunrs
    -------
    fitness: float
        Target value for minimization algorithm
    """
    # Assuming that the first point is very close to zero
    # let's ignore it.
    _strain = strain[1:]
    _stress = stress[1:]

    predictedStress: Vector = deformation_mode(elastic_model, _strain, params)

    err: float = error_measure(_stress, predictedStress)

    return err
