from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, differential_evolution

from polymat.calibration.error_measures import error_re
from polymat.calibration.fitness_functions import fitness_elastic
from polymat.types import ElasticDeformation, ElasticModel, Vector

SEARCH_TOL = 1e-9
SEARCH_MAX_ITER = 500
SEARCH_POP_SIZE = 20


def fit_elastic_material(
    strain: list[Vector],
    stress: list[Vector],
    elastic_model: ElasticModel,
    deformation_mode: list[ElasticDeformation],
    lower_bound: Vector,
    upper_bound: Vector,
    linear_constraint: LinearConstraint | None = None,
    nonlinear_constraint: NonlinearConstraint | None = None,
) -> tuple[Vector, float]:
    """
    Solves optimization problem to calibrate the material model against the test data.

    Uses differential evolution search.

    Parameters
    ----------
    strain: list[Vector]
        List of experimental true strain curves

    stress: list[Vector]
        List of experimental true stress curves

    elastic_model: ElasticModel
        Hyperelastic material model from polymat.materials

    deformation_mode: list[ElasticDeformation]
        List of corresponding stress computation function from polymat.mechanics.elastic_deformation

    lower_bound: Vector | None = None
        Lower bound for the search space

    upper_bound: Vector | None = None
        Upper bound for the search space

    linear_constraint: LinearConstraint | None = None
        Object modeling constraints in the form lb <= dot(A, x) <= ub

    nonlinear_constraint: NonlinearConstraint | None = None
        Object modeling constraints in the form lb <= f(x) <= ub

    Returns
    -------
    calibrated_params: Vector
        Found list of optimal material parameters

    error: float
        Computed error for the found solution
    """
    constraints: list[LinearConstraint | NonlinearConstraint] = []

    if linear_constraint:
        constraints.append(linear_constraint)

    if nonlinear_constraint:
        constraints.append(nonlinear_constraint)

    bounds: Bounds = Bounds(lower_bound, upper_bound)

    opt: OptimizeResult = differential_evolution(
        func=fitness_elastic,
        bounds=bounds,
        args=(strain, stress, elastic_model, deformation_mode, error_re),
        constraints=constraints,
        maxiter=SEARCH_MAX_ITER,
        popsize=SEARCH_POP_SIZE,
        tol=SEARCH_TOL,
    )

    calibration_params: Vector = opt.x
    calibration_error: float = opt.fun

    return calibration_params, calibration_error
