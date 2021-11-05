import pyomo.environ as pyo
import numpy as np
import pytest

from omlt.block import OmltBlock
from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.relu import ReLUBigMFormulation, ReLUComplementarityFormulation


@pytest.fixture
def two_node_network():
    """
            1           1
    x0 -------- (1) --------- (3)
     |                   /
     |                  /
     |                 / 5
     |                /
     |               |
     |    -1         |     1
     ---------- (2) --------- (4)
    """
    net = NetworkDefinition(input_bounds=[(-10.0, 10.0)])

    input_layer = InputLayer([1, 1])
    net.add_node(input_layer)

    dense_layer_0 = DenseLayer(
        input_layer.output_size,
        [1, 2],
        activation="relu",
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_node(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_node(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    return net


def test_two_node_bigm(two_node_network):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUBigMFormulation(two_node_network)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0, 0].fix(-2)
    _status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < 1e-8

    m.neural_net_block.inputs[0, 0].fix(1)
    _status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < 1e-8


def test_two_node_complementarity(two_node_network):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUComplementarityFormulation(two_node_network, transform="mpec.simple_nonlinear")
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0, 0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < 1e-6

    m.neural_net_block.inputs[0, 0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < 1e-6
