import keras
import pyomo.environ as pyo
import pytest

from omlt.block import OmltBlock
from omlt.neuralnet.full_space import FullSpaceContinuousFormulation
from omlt.neuralnet.keras_reader import load_keras_sequential
from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.reduced_space import ReducedSpaceContinuousFormulation


def test_two_node_full_space():
    """
    Test of the following model:

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

    n_inputs = 1
    n_hidden = 2
    n_outputs = 2
    w = {1: {0: 1.0}, 2: {0: -1.0}, 3: {1: 1.0, 2: 5.0}, 4: {2: 1.0}}
    b = {1: 0, 2: 0, 3: 0, 4: 0}
    a = {1: "tanh", 2: "tanh"}

    net = NetworkDefinition(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        weights=w,
        biases=b,
        activations=a,
    )

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    assert m.nvariables() == 12
    assert m.nconstraints() == 11

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=True)
    pyo.assert_optimal_termination(status)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 0.9640275800758169) < 1e-8

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    pyo.assert_optimal_termination(status)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - -0.7615941559557649) < 1e-8


def test_two_node_reduced_space_1():
    """
    Test of the following model:

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

    n_inputs = 1
    n_hidden = 2
    n_outputs = 2
    w = {1: {0: 1.0}, 2: {0: -1.0}, 3: {1: 1.0, 2: 5.0}, 4: {2: 1.0}}
    b = {1: 1, 2: 2, 3: 3, 4: 4}

    net = NetworkDefinition(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        weights=w,
        biases=b,
        activations={},
    )

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    m.neural_net_block.build_formulation(ReducedSpaceContinuousFormulation(net))
    assert m.nvariables() == 3
    assert m.nconstraints() == 2

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 22) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 8) < 1e-8

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 10) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 5) < 1e-8


# todo: Build more tests with different activations and edge cases
def xtest_two_node_pass_variables():
    """
    Test of the following model:

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

    n_inputs = 1
    n_nodes = 3
    n_outputs = 2
    w = {1: {0: 1.0}, 2: {0: -1.0}, 3: {1: 1.0, 2: 5.0}, 4: {2: 1.0}}
    b = {1: 0, 2: 0, 3: 0, 4: 0}

    m = pyo.ConcreteModel()
    outputs = pyo.Set(initialize=list(range(2)), ordered=True)
    m.x = pyo.Var()
    m.y = pyo.Var(outputs)
    m.neural_net_block = NeuralNetBlock()
    network_definition = ReducedSpaceNonlinear(pyo.tanh)
    network_definition.set_weights(w, b, n_inputs, n_outputs, n_nodes)
    m.neural_net_block.define_network(
        network_definition=network_definition, input_vars=[m.x], output_vars=m.y
    )

    m.x.fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.y[3]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - 0.9640275800758169) < 1e-8

    m.x.fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.y[3]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - -0.7615941559557649) < 1e-8
