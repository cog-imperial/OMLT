import pyomo.environ as pyo
import pytest
from omlt.base import OmltVarFactory
from omlt.dependencies import julia_available

VAR_VALUE = 3
FIX_VALUE = 2
UPPER_BOUND = 5

var_factory = OmltVarFactory()


def _test_scalar_var(lang):
    v = var_factory.new_var(lang=lang, initialize=2)
    assert v._parent is None
    assert v._constructed is False
    assert v.name is None
    assert v.is_indexed() is False
    assert v.ctype == pyo.ScalarVar
    assert v.is_component_type()
    assert v.valid_model_component()

    v.construct()
    assert v.is_constructed()

    v.value = 3
    assert v.value == VAR_VALUE

    v.fix(2, skip_validation=True)
    v.bounds = (0, 5)
    assert v.lb == 0
    assert v.ub == UPPER_BOUND
    v.lb = 1
    v.ub = 3
    assert v.bounds == (1, 3)

    v.domain = pyo.Integers
    assert v.domain == pyo.Integers


def test_scalar_pyomo():
    _test_scalar_var("pyomo")


def test_scalar_invalid_lang():
    expected_msg = "Variable format %s not recognized. Supported formats are %s"
    with pytest.raises(KeyError, match=expected_msg):
        var_factory.new_var(lang="test")


def _test_indexed_var(lang):
    v = var_factory.new_var(range(4), lang=lang, initialize=2)
    assert v._parent is None
    assert v._constructed is False
    assert v.is_indexed() is True
    assert v.ctype == pyo.Var

    v.construct()
    assert v.is_constructed()

    v.value = 3
    assert v.value == VAR_VALUE

    v.fix(2, skip_validation=True)
    for e in v:
        assert v[e].value == FIX_VALUE

    v.fix()

    v.bounds = (0, 5)
    v.setlb(1)
    v.setub(3)
    assert v.bounds == (1, 3)

    v.domain = pyo.Integers
    assert v.domain == pyo.Integers


def test_indexed_pyomo():
    _test_indexed_var("pyomo")


def test_indexed_invalid_lang():
    expected_msg = (
        "Variable format %s not recognized. Supported formats are %s"
    )
    with pytest.raises(KeyError, match=expected_msg):
        var_factory.new_var(range(3), lang="test")
