import numpy as np
import pyomo.environ as pyo
import pytest

from omlt import OmltBlock
from omlt.dependencies import onnx_available
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    ReluBigMFormulation,
    ReluComplementarityFormulation,
    ReluPartitionFormulation,
)
from omlt.neuralnet.activations import ComplementarityReLUActivation

NEAR_EQUAL = 1e-3


def test_two_node_bigm(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < NEAR_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < NEAR_EQUAL


def test_two_node_relu_big_m_formulation(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReluBigMFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < NEAR_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < NEAR_EQUAL


def test_two_node_complementarity(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(
        two_node_network_relu,
        activation_constraints={"relu": ComplementarityReLUActivation()},
    )
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < NEAR_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < NEAR_EQUAL


def test_two_node_relu_complementarity_formulation(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReluComplementarityFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < NEAR_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < NEAR_EQUAL


def test_two_node_relu_partition_formulation(two_node_network_relu):
    m = pyo.ConcreteModel()

    m.neural_net_block = OmltBlock()
    formulation = ReluPartitionFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=0)

    m.neural_net_block.inputs[0].fix(-2)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < NEAR_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < NEAR_EQUAL


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_conv_relu_big_m_formulation(datadir):
    from omlt.io.onnx import load_onnx_neural_network_with_bounds

    net = load_onnx_neural_network_with_bounds(datadir.file("keras_conv_7x7_relu.onnx"))
    m = pyo.ConcreteModel()

    m.neural_net_block = OmltBlock()
    formulation = ReluBigMFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=0)

    # compute expected output for this input
    x_start = np.eye(7, 7).reshape(1, 7, 7)
    x = x_start
    for layer in net.layers:
        x = layer.eval_single_layer(x)
    output = x

    for i in range(7):
        for j in range(7):
            m.neural_net_block.inputs[0, i, j].fix(x_start[0, i, j])
    pyo.SolverFactory("cbc").solve(m, tee=False)

    d, r, c = output.shape
    for i in range(d):
        for j in range(r):
            for k in range(c):
                expected = output[i, j, k]
                actual = pyo.value(m.neural_net_block.outputs[i, j, k])
                assert abs(actual - expected) < NEAR_EQUAL
