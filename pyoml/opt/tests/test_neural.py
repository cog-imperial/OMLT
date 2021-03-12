import pytest
import pyomo.environ as pyo
from pyoml.opt.neural import NeuralBlock, EncodedMode

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
    m.neural_block = NeuralBlock()
    m.neural_block.set_neural_data(n_inputs,n_outputs,n_nodes,w,b,activation = pyo.tanh)

    m.neural_block.build(mode = EncodedMode())

    m.neural_block.x[0].fix(-2)
    m.obj1 = pyo.Objective(expr = 0)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)

    assert abs(pyo.value(m.neural_block.y[3]) - 3.856110320303267) < 1e-8
    assert abs(pyo.value(m.neural_block.y[4]) - 0.9640275800758169) < 1e-8

    m.neural_block.x[0].fix(1)
    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    assert abs(pyo.value(m.neural_block.y[3]) - -3.046376623823058) < 1e-8
    assert abs(pyo.value(m.neural_block.y[4]) - -0.7615941559557649) < 1e-8
