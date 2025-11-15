from numpy import abs, maximum, mean, sqrt

from polymat.types import Vector


def error_nmad(x: Vector, p: Vector) -> float:
    """
    Normalized Mean Absolute Difference (NMAD) error measure

    Parameters
    ----------
    x: Vector
        eXperimental data vector

    p: Vector
        Predicted data vector

    Returns
    -------
    error: float
        Computed NMAD error
    """
    mad: float = mean(abs(x - p))
    arith_mean: float = max((mean(abs(x)), mean(abs(p))))
    nmad: float = mad / arith_mean

    return nmad * 100.0


def error_rms(x: Vector, p: Vector) -> float:
    """
    Normalized Root-Mean-Squared Difference (RMS) error measure

    Parameters
    ----------
    x: Vector
        eXperimental data vector

    p: Vector
        Predicted data vector

    Returns
    -------
    error: float
        Computed NRMS error
    """
    rms: float = sqrt(mean((x - p) ** 2.0))
    geom_mean: float = sqrt(mean(x**2))

    return 100.0 * rms / geom_mean


def error_re(x: Vector, p: Vector) -> float:
    """
    Normalized Relative Error (RE) measure

    Parameters
    ----------
    x: Vector
        eXperimental data vector

    p: Vector
        Predicted data vector

    Returns
    -------
    error: float
        Computed RE error
    """
    abs_res: Vector = abs(x - p)
    max_abs: Vector = maximum(abs(x), abs(p))
    rel_err: Vector = abs_res / max_abs
    mean_re: float = mean(rel_err)

    return 100.0 * mean_re
