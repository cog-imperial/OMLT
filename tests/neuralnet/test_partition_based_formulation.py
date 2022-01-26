from omlt.io.onnx import load_onnx_neural_network, load_onnx_neural_network_with_bounds
import pyomo.environ as pyo
import onnx

from omlt.block import OmltBlock
from omlt.neuralnet import PartitionBasedNeuralNetworkFormulation
from omlt.neuralnet.activations import ComplementarityReLUActivation


def test_two_node_partition_based(two_node_network_relu):
    m = pyo.ConcreteModel()

    m.neural_net_block = OmltBlock()
    formulation = PartitionBasedNeuralNetworkFormulation(two_node_network_relu)
    m.neural_net_block.build_formulation(formulation)
    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0,0]) - 10) < 1e-3
    assert abs(pyo.value(m.neural_net_block.outputs[0,1]) - 2) < 1e-3
