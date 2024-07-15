import pyomo.environ as pyo
import pytest
from omlt.base import OmltConstraint, OmltExpr, OmltExprIndexed, OmltExprScalar, OmltVar

VAR1_VALUE = 6
VAR2_VALUE = 3
CONST_VALUE = 4

def test_build_constraint():
    v1 = OmltVar()
    v1.domain = pyo.Integers
    v1.value = VAR1_VALUE
    e1 = v1 + CONST_VALUE

    v2 = OmltVar()
    v2.domain = pyo.Integers
    v2.value = VAR2_VALUE
    e2 = v2 + CONST_VALUE

    c_eq = e1 == e2

    assert c_eq.sense == "=="
    assert id(c_eq.lhs) == id(e1._expression)

    c_le = OmltConstraint(lhs=e1, sense="<=", rhs=e2)

    assert c_le.sense == "<="
    assert id(c_le.rhs) == id(e2._expression)

def test_constraint_invalid_lang():
    expected_msg = (
        "Constraint format %s not recognized. Supported formats "
        "are 'pyomo' or 'jump'."
    )

    with pytest.raises(ValueError, match=expected_msg):
        OmltConstraint(lang="test")

    with pytest.raises(ValueError, match=expected_msg):
        OmltConstraint(range(3), lang="test")
