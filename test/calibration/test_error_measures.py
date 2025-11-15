from numpy import array, isclose

from polymat.calibration.error_measures import error_nmad, error_rms


def test_error_nmad() -> None:
    x = array([0.1, 0.2, 0.3])
    p = array([0.15, 0.30, 0.45])

    err: float = error_nmad(x, p)

    assert isclose(err, 33.3, rtol=0.01)


def test_error_rms() -> None:
    x = array([0.1, 0.2, 0.3])
    p = array([0.15, 0.30, 0.45])

    err: float = error_rms(x, p)

    assert isclose(err, 50.0, rtol=0.01)
