from polymat.types import ElasticDeformation, ElasticModel, ErrorMeasure, Vector


def fitness_elastic(
    params: Vector,
    strain: list[Vector],
    stress: list[Vector],
    elastic_model: ElasticModel,
    deformation_mode: list[ElasticDeformation],
    error_measure: ErrorMeasure,
) -> float:
    """
    Fitness fuction to measure error in stress prediction of hyperelastic materials.

    Handles multiple tests by averaging the indivual test errors.

    Parameters
    ----------
    params: Vector
        Test material parameters

    strain: list[Vector]
        List of experimental strain curves

    stress: list[Vector]
        List of experimental stress curves

    elastic_model: ElasticModel
        Hyperelastic stress computation function

    deformation_mode: list[ElasticDeformation]
        List of corresponding stress curve computation functions

    error_measure: ErrorMeasure,
        Error metric for the fitness function

    Retunrs
    -------
    fitness: float
        Target value for minimization algorithm
    """
    err: float = 0.0
    n: int = 0

    for _strain, _stress, _deformation in zip(strain, stress, deformation_mode):
        predictedStress: Vector = _deformation(elastic_model, _strain[1:], params)
        err += error_measure(_stress[1:], predictedStress)
        n += 1

    return err / n
