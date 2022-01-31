import pyomo.environ as pyo

from omlt.block import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation, ReluBigMFormulation, ReluComplementarityFormulation
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
