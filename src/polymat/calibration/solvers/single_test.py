from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, differential_evolution

from polymat.calibration.error_measures import error_re
from polymat.calibration.fitness_functions import fitness_single_test_elastic
from polymat.types import ElasticDeformation, ElasticModel, Vector

SEARCH_TOL = 1e-9
SEARCH_MAX_ITER = 500
SEARCH_POP_SIZE = 20


def fit_single_test_elastic(
    strain: Vector,
    stress: Vector,
    elastic_model: ElasticModel,
    deformation_mode: ElasticDeformation,
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
    strain: Vector
        Experimental true strain curve

    stress: Vector
        Experimental true stress curve

    elastic_model: ElasticModel
        Hyperelastic material model from polymat.materials

    deformation_mode: ElasticDeformation
        Stress computation function from polymat.mechanics.elastic_deformation

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
        func=fitness_single_test_elastic,
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
