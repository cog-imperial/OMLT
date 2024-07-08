import pyomo.environ as pyo
import pytest
from omlt.base import OmltVar
from omlt.dependencies import julia_available


def _test_scalar_var(lang):
    v = OmltVar(lang=lang, initialize=2, domain=pyo.Integers)
    assert v.is_indexed() is False
    assert v.ctype == pyo.ScalarVar

    v.construct()

    v.value = 3
    assert v.value == 3

    v.bounds = (0, 5)
    assert v.lb == 0
    assert v.ub == 5
    assert v.bounds == (0, 5)


def test_scalar_pyomo():
    _test_scalar_var("pyomo")


@pytest.mark.skipif(
    not julia_available, reason="Test only valid when Julia is available"
)
def test_scalar_jump():
    _test_scalar_var("jump")
