import pytest
import numpy as np
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import ReducedSpaceNeuralNetworkFormulation, NeuralNetworkFormulation, NetworkDefinition
from omlt.neuralnet.layer import InputLayer, DenseLayer

def two_node_network(activation, input_value):
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
        activation=activation,
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([1.0, 2.0])
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation=activation,
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([3.0, 4.0])
    )
    net.add_layer(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    y = input_layer.eval(np.asarray([input_value]))
    y = dense_layer_0.eval(y)
    y = dense_layer_1.eval(y)

    return net, y

def _test_two_node_reduced_space(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(ReducedSpaceNeuralNetworkFormulation(net))
    assert m.nvariables() == 6
    assert m.nconstraints() == 5

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

def _test_two_node_full_space_smooth(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(NeuralNetworkFormulation(net))
    assert m.nvariables() == 15
    assert m.nconstraints() == 14

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

def test_two_node_full_space_relu():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network('relu', -2.0)
    m.neural_net_block.build_formulation(NeuralNetworkFormulation(net))
    assert m.nvariables() == 19
    assert m.nconstraints() == 26

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

    net, y = two_node_network('relu', 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - y[0,0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - y[0,1]) < 1e-6

def test_two_node_reduced_space():
    _test_two_node_reduced_space('linear')
    _test_two_node_reduced_space('sigmoid')
    _test_two_node_reduced_space('tanh')

def test_two_node_full_space():
    _test_two_node_full_space_smooth('linear')
    _test_two_node_full_space_smooth('sigmoid')
    _test_two_node_full_space_smooth('tanh')

