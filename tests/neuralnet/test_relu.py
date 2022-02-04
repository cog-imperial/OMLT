import pyomo.environ as pyo
import numpy as np

from omlt.block import OmltBlock
from omlt.io.onnx import load_onnx_neural_network_with_bounds
from omlt.neuralnet import FullSpaceNNFormulation, ReluBigMFormulation, ReluComplementarityFormulation, ReluPartitionFormulation
from omlt.neuralnet.activations import ComplementarityReLUActivation

# TODO: Add tests for single dimensional outputs as well

def test_two_node_bigm(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 1) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 0) < 1e-3

def test_two_node_ReluBigMFormulation(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReluBigMFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 1) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 0) < 1e-3


def test_two_node_complementarity(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(
        two_node_network_relu,
        activation_constraints={
            "relu": ComplementarityReLUActivation()
        }
    )
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 1) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 0) < 1e-3

def test_two_node_ReluComplementarityFormulation(two_node_network_relu):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReluComplementarityFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 1) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 0) < 1e-3

def test_two_node_ReluPartitionFormulation(two_node_network_relu):
    m = pyo.ConcreteModel()

    m.neural_net_block = OmltBlock()
    formulation = ReluPartitionFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=0)

    m.neural_net_block.inputs[0].fix(-2)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 1) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 0) < 1e-3

def test_conv_ReluBigMFormulation(datadir):
    net = load_onnx_neural_network_with_bounds(datadir.file('keras_conv_7x7_relu.onnx'))
    m = pyo.ConcreteModel()

    m.neural_net_block = OmltBlock()
    formulation = ReluBigMFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=0)

    # compute expected output for this input
    input = np.eye(7, 7).reshape(1, 7, 7)
    x = input
    for layer in net.layers:
        x = layer.eval(x)
    output = x

    for i in range(7):
        for j in range(7):
            m.neural_net_block.inputs[0, i, j].fix(input[0, i, j])
    status = pyo.SolverFactory("cbc").solve(m, tee=False)

    d, r, c = output.shape
    for i in range(d):
        for j in range(r):
            for k in range(c):
                expected = output[i, j, k]
                actual = pyo.value(m.neural_net_block.outputs[i, j, k])
                assert abs(actual - expected) < 1e-3
