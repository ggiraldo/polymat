from numpy import array, isclose

from polymat.calibration.error_measures import error_nmad, error_re, error_rms


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


def test_error_re() -> None:
    x = array([0.1, 0.2, 0.3])
    p = array([0.15, 0.30, 0.45])

    err: float = error_re(x, p)

    assert isclose(err, 33.3, rtol=0.01)


def test_error_re_2() -> None:
    x = array([0.1, 0.2, 0.3])
    p = array([0.11, 0.21, 0.60])

    err: float = error_re(x, p)

    assert isclose(err, 21.3, rtol=0.01)


def test_error_nmad_2() -> None:
    x = array([0.1, 0.2, 0.3])
    p = array([0.11, 0.21, 0.60])

    err: float = error_nmad(x, p)

    assert isclose(err, 34.8, rtol=0.01)


def test_error_re_3() -> None:
    x = array([0.01, 0.2, 3.0])
    p = array([0.02, 0.22, 3.3])

    err: float = error_re(x, p)

    assert isclose(err, 22.7, rtol=0.01)


def test_error_nmad_3() -> None:
    x = array([0.01, 0.2, 3.0])
    p = array([0.02, 0.22, 3.3])

    err: float = error_nmad(x, p)

    assert isclose(err, 9.3, rtol=0.01)
