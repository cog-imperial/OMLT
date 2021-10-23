import pyomo.environ as pyo
import pytest

from omlt.block import OmltBlock
from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.relu import ReLUBigMFormulation, ReLUComplementarityFormulation


def test_two_node_bigm():
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
    a = {1: "relu", 2: "relu"}

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
    formulation = ReLUBigMFormulation(net, M=1e6)
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 10) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 2) < 1e-8

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 1) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 0) < 1e-8


def test_two_node_complementarity():
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
    a = {1: "relu", 2: "relu"}

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
    formulation = ReLUComplementarityFormulation(net, transform="mpec.simple_nonlinear")
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 10) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 2) < 1e-6

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0]) - 1) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[1]) - 0) < 1e-6
