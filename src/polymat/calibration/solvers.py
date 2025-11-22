from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize

from polymat.calibration.error_measures import error_re
from polymat.calibration.fitness_functions import fitness_single_test_elastic
from polymat.types import ElasticDeformation, ElasticModel, Vector


def fit_single_test_elastic(
    strain: Vector,
    stress: Vector,
    elastic_model: ElasticModel,
    deformation_mode: ElasticDeformation,
    initial_guess: Vector,
    lower_bound: Vector | None = None,
    upper_bound: Vector | None = None,
    linear_constraint: LinearConstraint | None = None,
    nonlinear_constraint: NonlinearConstraint | None = None,
) -> tuple[Vector, float]:
    """
    Solves optimization problem to calibrate the material model against the test data using simplex algorithm.

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

    initial_guess: Vector
        Initial search candidate for the material parameters

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

    bounds = None

    if lower_bound and upper_bound:
        bounds = Bounds(lower_bound, upper_bound)

    opt: OptimizeResult = minimize(
        fun=fitness_single_test_elastic,
        x0=initial_guess,
        args=(strain, stress, elastic_model, deformation_mode, error_re),
        bounds=bounds,
        constraints=constraints,
        tol=1e-9,
    )

    calibration_params: Vector = opt.x
    calibration_error: float = opt.fun

    return calibration_params, calibration_error


def fit_single_test_incompressible(
    strain: Vector,
    stress: Vector,
    elastic_model: ElasticModel,
    deformation_mode: ElasticDeformation,
    initial_guess: Vector,
    lower_bound: Vector | None = None,
    upper_bound: Vector | None = None,
    linear_constraint: LinearConstraint | None = None,
    nonlinear_constraint: NonlinearConstraint | None = None,
) -> tuple[Vector, float]:
    """
    Solves optimization problem to calibrate the material model against the test data using simplex algorithm.

    Assumes incompressibility, thus the bulk modulus is not included.

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

    initial_guess: Vector
        Initial search candidate for the material parameters

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

    bounds = None

    if lower_bound and upper_bound:
        bounds = Bounds(lower_bound, upper_bound)

    opt: OptimizeResult = minimize(
        fun=fitness_single_test_elastic,
        x0=initial_guess,
        args=(strain, stress, elastic_model, deformation_mode, error_re),
        bounds=bounds,
        constraints=constraints,
        tol=1e-9,
    )

    calibration_params: Vector = opt.x
    calibration_error: float = opt.fun

    return calibration_params, calibration_error
