import pytest
import pyomo.environ as pyo
from pyoml.opt.relu import ReLUBlock,BigMReluMode,ComplementarityReluMode


def test_two_node():
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
    m.relu_block = ReLUBlock()
    m.relu_block.set_neural_data(n_inputs,n_outputs,n_nodes,w,b)

    mode = ComplementarityReluMode(transform = "mpec.simple_nonlinear")
    m.relu_block.build(mode)

    m.relu_block.x[0].fix(-2)
    m.obj1 = pyo.Objective(expr = 0)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.relu_block.y[3]) - 10) < 1e-6
    assert abs(pyo.value(m.relu_block.y[4]) - 2) < 1e-6

    m.relu_block.x[0].fix(1)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.relu_block.y[3]) - 1) < 1e-6
    assert abs(pyo.value(m.relu_block.y[4]) - 0) < 1e-6
