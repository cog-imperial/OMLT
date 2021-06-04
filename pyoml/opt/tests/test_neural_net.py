import pytest
import pyomo.environ as pyo
from pyoml.opt.neuralnet import NeuralNetBlock, FullSpaceNonlinear, ReducedSpaceNonlinear

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
    n_nodes = 3
    n_outputs = 2
    w = {1: {0: 1.0},
         2: {0: -1.0},
         3: {1: 1.0, 2: 5.0},
         4: {2: 1.0}}
    b = {1: 0, 2:0, 3:0, 4:0}

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetBlock()
    network_definition = FullSpaceNonlinear(pyo.tanh)
    network_definition.set_weights(w,b,n_inputs,n_outputs,n_nodes)
    m.neural_net_block.define_network(network_definition = network_definition)
    assert m.nvariables() == 8
    assert m.nconstraints() == 7

    m.neural_net_block.x[0].fix(-2)
    m.obj1 = pyo.Objective(expr = 0)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.y[3]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - 0.9640275800758169) < 1e-8

    m.neural_net_block.x[0].fix(1)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.y[3]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - -0.7615941559557649) < 1e-8

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
    n_nodes = 3
    n_outputs = 2
    w = {1: {0: 1.0},
         2: {0: -1.0},
         3: {1: 1.0, 2: 5.0},
         4: {2: 1.0}}
    b = {1: 0, 2:0, 3:0, 4:0}

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetBlock()
    network_definition = ReducedSpaceNonlinear(pyo.tanh)
    network_definition.set_weights(w,b,n_inputs,n_outputs,n_nodes)
    m.neural_net_block.define_network(network_definition = network_definition)
    assert m.nvariables() == 3
    assert m.nconstraints() == 2

    m.neural_net_block.x[0].fix(-2)
    m.obj1 = pyo.Objective(expr = 0)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.y[3]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - 0.9640275800758169) < 1e-8

    m.neural_net_block.x[0].fix(1)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.y[3]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - -0.7615941559557649) < 1e-8


def test_two_node_pass_variables():
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
    w = {1: {0: 1.0},
         2: {0: -1.0},
         3: {1: 1.0, 2: 5.0},
         4: {2: 1.0}}
    b = {1: 0, 2:0, 3:0, 4:0}

    m = pyo.ConcreteModel()
    outputs = pyo.Set(initialize=list(range(2)), ordered=True)
    m.x = pyo.Var()
    m.y = pyo.Var(outputs)
    m.neural_net_block = NeuralNetBlock()
    network_definition = ReducedSpaceNonlinear(pyo.tanh)
    network_definition.set_weights(w,b,n_inputs,n_outputs,n_nodes)
    m.neural_net_block.define_network(network_definition = network_definition,input_vars = [m.x],output_vars = m.y)

    m.x.fix(-2)
    m.obj1 = pyo.Objective(expr = 0)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.y[3]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - 0.9640275800758169) < 1e-8

    m.x.fix(1)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.y[3]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_net_block.y[4]) - -0.7615941559557649) < 1e-8
