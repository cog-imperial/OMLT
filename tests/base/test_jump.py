import re

import pytest

from omlt.base.julia import (
    OmltBlockJuMP,
    OmltConstraintScalarJuMP,
    OmltExprJuMP,
    OmltScalarJuMP,
)
from omlt.dependencies import julia_available, onnx, onnx_available
from omlt.neuralnet import (
    FullSpaceSmoothNNFormulation,
    ReducedSpaceSmoothNNFormulation,
    ReluBigMFormulation,
    ReluPartitionFormulation,
)

if onnx_available:
    from omlt.io.onnx import load_onnx_neural_network

from omlt import OffsetScaling

TWO = 2
FOUR = 4
FIVE = 5
SIX = 6
ELEVEN = 11


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_variable_jump():
    v = OmltScalarJuMP(bounds=(0, 5), initialize=(3,))

    assert v.name is None
    assert v.lb == 0
    assert v.ub == FIVE

    v.value = 4
    assert v.value == FOUR


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_expression_linear_jump():
    # JumpVar + int
    # OmltExprJuMP/AffExpr + OmltExprJuMP/AffExpr
    # OmltExprJuMP/NonlinearExpr + int
    #
    # int - JumpVar
    # JumpVar - OmltExprJuMP
    # OmltExprJuMP/AffExpr - OmltExprJuMP
    # OmltExprJump/NonlinearExpr - JumpVar
    #
    # int * JumpVar
    # OmltExprJuMP/AffExpr * ParamData
    # OmltExprJuMP/NonlinearExpr * int
    #
    # OmltExprJuMP/AffExpr / int

    jump_block = OmltBlockJuMP()
    jump_block.v1 = OmltScalarJuMP(initialize=2)
    jump_block.v2 = OmltScalarJuMP(initialize=3)
    element2 = jump_block.v2._var

    var_plus_three = jump_block.v1 + 3
    assert isinstance(var_plus_three, OmltExprJuMP)
    assert var_plus_three() == FIVE

    three_minus_var = 3 - jump_block.v1
    assert isinstance(three_minus_var, OmltExprJuMP)
    assert three_minus_var() == 1

    three_times_var = 3 * jump_block.v1
    assert isinstance(three_times_var, OmltExprJuMP)
    assert three_times_var() == SIX

    var_times_three = jump_block.v1 * 3
    assert isinstance(var_times_three, OmltExprJuMP)
    assert var_times_three() == SIX

    expr_sum = var_plus_three + var_times_three
    assert isinstance(expr_sum, OmltExprJuMP)
    assert expr_sum() == ELEVEN

    var_minus_expr = element2 - three_minus_var
    assert isinstance(var_minus_expr, OmltExprJuMP)
    assert var_minus_expr() == TWO

    expr_minus_expr = var_plus_three - three_minus_var
    assert isinstance(expr_minus_expr, OmltExprJuMP)
    assert expr_minus_expr() == FOUR

    expr_div_int = var_plus_three / 5
    assert isinstance(expr_div_int, OmltExprJuMP)
    assert expr_div_int() == 1

    constraint = var_minus_expr == expr_div_int
    assert isinstance(constraint, OmltConstraintScalarJuMP)


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_expression_nonlinear_jump():
    jump_block = OmltBlockJuMP()
    jump_block.v1 = OmltScalarJuMP(initialize=2)
    element1 = jump_block.v1._var

    expe = element1.exp()
    loge = element1.log()
    hypt = element1.tanh()

    assert isinstance(expe, OmltExprJuMP)
    assert isinstance(loge, OmltExprJuMP)
    assert isinstance(hypt, OmltExprJuMP)
    assert isinstance(hypt + 3, OmltExprJuMP)
    assert isinstance(hypt - 3, OmltExprJuMP)
    assert isinstance(hypt - element1, OmltExprJuMP)
    assert isinstance(hypt * 3, OmltExprJuMP)
    assert isinstance(hypt.log(), OmltExprJuMP)
    assert isinstance(hypt.tanh(), OmltExprJuMP)


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_expression_bad_definition_jump():
    expected_msg1 = re.escape(
        "('Tried to create an OmltExprJuMP with an invalid expression. Expressions "
        "must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where "
        "d is exp, log, or tanh. %s was provided', 'invalid')"
    )
    with pytest.raises(ValueError, match=expected_msg1):
        OmltExprJuMP("invalid")
    expected_msg2 = re.escape(
        "('Tried to create an OmltExprJuMP with an invalid expression. Expressions "
        "must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where "
        "d is exp, log, or tanh. %s was provided', ('invalid', 'pair'))"
    )
    with pytest.raises(ValueError, match=expected_msg2):
        OmltExprJuMP(("invalid", "pair"))
    expected_msg3 = re.escape(
        "('Tried to create an OmltExprJuMP with an invalid expression. Expressions "
        "must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where "
        "d is exp, log, or tanh. %s was provided', ('invalid', 'triple', 'expression'))"
    )
    with pytest.raises(ValueError, match=expected_msg3):
        OmltExprJuMP(("invalid", "triple", "expression"))


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_expression_bad_arithmetic_jump():
    v = OmltScalarJuMP()
    expected_msg = (
        "('Unrecognized types for addition, %s, %s', "
        "<class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"
    )
    with pytest.raises(TypeError, match=expected_msg):
        v + "invalid"

    expected_msg = (
        "('Unrecognized types for subtraction, %s, %s', "
        "<class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"
    )
    with pytest.raises(TypeError, match=expected_msg):
        v - "invalid"

    expected_msg = (
        "('Unrecognized types for multiplication, %s, %s', "
        "<class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"
    )
    with pytest.raises(TypeError, match=expected_msg):
        v * "invalid"

    expected_msg = (
        "('Unrecognized types for division, %s, %s', "
        "<class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"
    )
    with pytest.raises(TypeError, match=expected_msg):
        v / "invalid"


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_two_node_relu_big_m_jump(two_node_network_relu):
    m_neural_net_block = OmltBlockJuMP()
    formulation = ReluBigMFormulation(two_node_network_relu)
    m_neural_net_block.build_formulation(formulation)


@pytest.mark.skipif(not julia_available, reason="Need JuMP for this test")
def test_two_node_relu_partition_jump(two_node_network_relu):
    m_neural_net_block = OmltBlockJuMP()
    formulation = ReluPartitionFormulation(two_node_network_relu)
    m_neural_net_block.build_formulation(formulation)


@pytest.mark.skipif(
    not onnx_available or not julia_available, reason="Need JuMP and ONNX for this test"
)
def test_full_space_sigmoid_jump(datadir):
    m_neural_net_block = OmltBlockJuMP()
    neural_net = onnx.load(datadir.file("keras_linear_131_sigmoid.onnx"))

    scale_x = (1, 0.5)
    scale_y = (-0.25, 0.125)

    scaler = OffsetScaling(
        offset_inputs=[scale_x[0]],
        factor_inputs=[scale_x[1]],
        offset_outputs=[scale_y[0]],
        factor_outputs=[scale_y[1]],
    )

    scaled_input_bounds = {0: (-4, 5)}
    net = load_onnx_neural_network(neural_net, scaler, input_bounds=scaled_input_bounds)
    formulation = FullSpaceSmoothNNFormulation(net)
    m_neural_net_block.build_formulation(formulation)


@pytest.mark.skipif(
    not onnx_available or not julia_available, reason="Need JuMP and ONNX for this test"
)
def test_reduced_space_linear_jump(datadir):
    m_neural_net_block = OmltBlockJuMP()
    neural_net = onnx.load(datadir.file("keras_linear_131.onnx"))

    scale_x = (1, 0.5)
    scale_y = (-0.25, 0.125)

    scaler = OffsetScaling(
        offset_inputs=[scale_x[0]],
        factor_inputs=[scale_x[1]],
        offset_outputs=[scale_y[0]],
        factor_outputs=[scale_y[1]],
    )

    scaled_input_bounds = {0: (-4, 5)}
    net = load_onnx_neural_network(neural_net, scaler, input_bounds=scaled_input_bounds)
    formulation = ReducedSpaceSmoothNNFormulation(net)
    m_neural_net_block.build_formulation(formulation)
