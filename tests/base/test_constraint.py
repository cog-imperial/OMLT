import pyomo.environ as pyo
import pytest

from omlt.base import (
    OmltConstraintFactory,
    OmltVarFactory,
)

VAR1_VALUE = 6
VAR2_VALUE = 3
CONST_VALUE = 4

var_factory = OmltVarFactory()
constraint_factory = OmltConstraintFactory()


def test_build_constraint():
    v1 = var_factory.new_var()
    v1.domain = pyo.Integers
    v1.value = VAR1_VALUE
    e1 = v1 + CONST_VALUE

    v2 = var_factory.new_var()
    v2.domain = pyo.Integers
    v2.value = VAR2_VALUE
    e2 = v2 + CONST_VALUE

    c_equal_expressions = e1 == e2

    assert c_equal_expressions.sense == "=="
    assert id(c_equal_expressions.lhs) == id(e1._expression)

    c_equal_var = e1 == v2
    assert c_equal_var.sense == "=="
    assert id(c_equal_var.lhs) == id(e1._expression)

    c_equal_const = e1 == CONST_VALUE
    assert c_equal_const.sense == "=="
    assert id(c_equal_const.lhs) == id(e1._expression)

    c_le_expressions = e1 <= e2

    assert c_le_expressions.sense == "<="
    assert id(c_le_expressions.rhs) == id(e2._expression)

    c_le_var = e1 <= v2

    assert c_le_var.sense == "<="
    assert id(c_le_var.rhs) == id(v2._pyovar)

    c_le_const = e1 <= CONST_VALUE

    assert c_le_const.sense == "<="
    assert c_le_const.rhs == CONST_VALUE


def test_constraint_invalid_lang():
    expected_msg = "Constraint format %s not recognized. Supported formats are %s"

    with pytest.raises(KeyError, match=expected_msg):
        constraint_factory.new_constraint(lang="test")

    with pytest.raises(KeyError, match=expected_msg):
        constraint_factory.new_constraint(range(3), lang="test")


def test_constraint_invalid_index():
    v1 = var_factory.new_var()
    v1.domain = pyo.Integers
    v1.value = VAR1_VALUE
    e1 = v1 + CONST_VALUE

    c = constraint_factory.new_constraint(range(3))
    expected_msg = "Couldn't find index %s in index set %s."
    with pytest.raises(KeyError, match=expected_msg):
        c[4] = e1 >= 0

    with pytest.raises(KeyError, match=expected_msg):
        _ = c[4]
