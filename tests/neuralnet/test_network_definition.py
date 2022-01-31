import pyomo.environ as pyo
import numpy as np
import pytest

from omlt.block import OmltBlock
from omlt.neuralnet.nn_formulation import FullSpaceNNFormulation
from omlt.io.keras_reader import load_keras_sequential
from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import DenseLayer, InputLayer

# TODO: Build more tests with different activations and edge cases
def test_two_node_full_space():
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
    net = NetworkDefinition(scaled_input_bounds=[(-10.0, 10.0)])

    input_layer = InputLayer([1])
    net.add_layer(input_layer)

    dense_layer_0 = DenseLayer(
        input_layer.output_size,
        [1, 2],
        activation="relu",
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_layer(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    # verify that we are seeing the correct values
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=True)
    pyo.assert_optimal_termination(status)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10.0) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2.0) < 1e-8

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    pyo.assert_optimal_termination(status)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1.0) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0.0) < 1e-8
