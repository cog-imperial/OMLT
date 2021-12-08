import pyomo.environ as pyo

from omlt.block import OmltBlock
from omlt.neuralnet import ComplementarityReLUActivation, NeuralNetworkFormulation


# TODO: Add tests for single dimensional outputs as well

def test_two_node_bigm(two_node_network):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = NeuralNetworkFormulation(two_node_network)
    m.neural_net_block.build_formulation(formulation)
    m.obj1 = pyo.Objective(expr=m.neural_net_block.outputs[0, 0])

    m.neural_net_block.inputs[0].fix(-2)
    _status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < 1e-8

    m.neural_net_block.inputs[0].fix(1)
    _status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < 1e-8
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < 1e-8


def test_two_node_complementarity(two_node_network):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = NeuralNetworkFormulation(
        two_node_network,
        activation_constraints={
            "relu": ComplementarityReLUActivation()
        }
    )
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2) < 1e-6

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0) < 1e-6
