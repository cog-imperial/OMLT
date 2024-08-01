import pyomo.environ as pyo
import pytest
from omlt.base import OmltExpr, OmltVarFactory

VAR1_VALUE = 6
VAR2_VALUE = 3
CONST_VALUE = 4

var_factory = OmltVarFactory()

def _test_build_scalar_expressions(lang):
    v1 = var_factory.new_var(lang=lang)
    v2 = var_factory.new_var(lang=lang)

    v1.domain = pyo.Integers
    v2.domain = pyo.Integers
    v1.value = VAR1_VALUE
    v2.value = VAR2_VALUE

    v_sum = v1 + v2
    assert(isinstance(v_sum, OmltExpr))
    assert(v_sum() == VAR1_VALUE + VAR2_VALUE)

    v_diff = v1 - v2
    assert(isinstance(v_diff, OmltExpr))
    assert(v_diff() == VAR1_VALUE - VAR2_VALUE)

    v_prod = v1 * v2
    assert(isinstance(v_prod, OmltExpr))
    assert(v_prod() == VAR1_VALUE * VAR2_VALUE)

    v_quot = v1 / v2
    assert(isinstance(v_quot, OmltExpr))
    assert(v_quot() == VAR1_VALUE / VAR2_VALUE)

    v_radd = CONST_VALUE + v1
    assert(isinstance(v_radd, OmltExpr))
    assert(v_radd() == CONST_VALUE + VAR1_VALUE)

    v_rsub = CONST_VALUE - v1
    assert(isinstance(v_rsub, OmltExpr))
    assert(v_rsub() == CONST_VALUE - VAR1_VALUE)

    v_rprod = CONST_VALUE * v1
    assert(isinstance(v_rprod, OmltExpr))
    assert(v_rprod() == CONST_VALUE * VAR1_VALUE)

    v_rquot = CONST_VALUE / v1
    assert(isinstance(v_rquot, OmltExpr))
    assert(v_rquot() == CONST_VALUE / VAR1_VALUE)


def test_build_scalar_exp_pyomo():
    _test_build_scalar_expressions("pyomo")

def test_init_scalar_expression():
    v1 = var_factory.new_var()
    v1.domain = pyo.Integers
    v1.value = VAR1_VALUE
    e1 = v1 + CONST_VALUE

    e2 = OmltExpr(expr=e1)

    assert e2.ctype == pyo.Expression
    assert e2.is_component_type()
    assert e2.is_expression_type()
    assert e2.valid_model_component()
    assert e2.is_potentially_variable()
    assert not e2.is_indexed()

    assert e2.nargs() == 2
    assert e2.args[1] == CONST_VALUE
    assert e2.arg(1) == CONST_VALUE
    assert len(e2) == 1
    assert e2() == VAR1_VALUE + CONST_VALUE

    expected_msg = (
        "Expression %s type %s not recognized."
        )

    with pytest.raises(TypeError, match=expected_msg):
        OmltExpr(expr="test")

    expected_msg = (
                "Expression format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'."
            )
    with pytest.raises(ValueError, match=expected_msg):
        OmltExpr(lang="test")

    expected_msg = "Expression middle term was {%s}."
    with pytest.raises(ValueError, match=expected_msg):
        OmltExpr(expr=(v1, "test", CONST_VALUE))

    expected_msg = "Term of expression %s is an unsupported type. %s"

    with pytest.raises(TypeError, match=expected_msg):
        OmltExpr(expr=((e1, "-", "test"), "+", CONST_VALUE))

def test_combine_scalar_expression():
    v1 = var_factory.new_var()
    v1.domain = pyo.Integers
    v1.value = VAR1_VALUE
    e1 = v1 + CONST_VALUE

    v2 = var_factory.new_var()
    v2.domain = pyo.Integers
    v2.value = VAR2_VALUE
    e2 = v2 + CONST_VALUE

    e_sum = e1 + e2
    assert e_sum() == VAR1_VALUE + VAR2_VALUE + 2*CONST_VALUE

    e_diff = e1 - e2
    assert e_diff() == VAR1_VALUE - VAR2_VALUE

    e_prod = e1 * e2
    assert e_prod() == (VAR1_VALUE + CONST_VALUE) * (VAR2_VALUE + CONST_VALUE)


    p_sum = e1 + CONST_VALUE
    assert p_sum() == VAR1_VALUE + 2*CONST_VALUE

    p_diff = e1 - CONST_VALUE
    assert p_diff() == VAR1_VALUE

    p_prod = e1 * CONST_VALUE
    assert p_prod() == (VAR1_VALUE + CONST_VALUE) * CONST_VALUE

    r_sum = CONST_VALUE + e1
    assert r_sum() == VAR1_VALUE + 2*CONST_VALUE

    r_diff = CONST_VALUE - e1
    assert r_diff() == -VAR1_VALUE

    r_prod = CONST_VALUE * e1
    assert r_prod() == (VAR1_VALUE + CONST_VALUE) * CONST_VALUE
