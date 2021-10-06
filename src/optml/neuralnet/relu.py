import pyomo.environ as pyo
import pyomo.mpec as mpec

from ..formulation import _PyomoFormulation
from .full_space import build_full_space_formulation


class ReLUBigMFormulation(_PyomoFormulation):
    def __init__(self, network_structure, M=1e6):
        """This class provides a full-space formulation of a neural network with ReLU
        activation functions using a MILP representation.
        """
        super(ReLUBigMFormulation, self).__init__(network_structure)
        self.M = M

    def _build_formulation(self):
        """This method is called by the OptMLBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_relu_mip_formulation(
            block=self.block, network_structure=self.network_definition, M=self.M
        )


class ReLUComplementarityFormulation(_PyomoFormulation):
    def __init__(self, network_structure, transform="mpec.simple_nonlinear"):
        """This class provides a full-space formulation of a neural network with ReLU
        activation functions using a MILP representation.
        """
        super(ReLUComplementarityFormulation, self).__init__(network_structure)
        self.transform = transform

    def _build_formulation(self):
        """This method is called by the OptMLBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_relu_complementarity_formulation(
            block=self.block,
            network_structure=self.network_definition,
            transform=self.transform,
        )


def build_relu_mip_formulation(block, network_structure, M=1e6):
    # build the full space structure without activations
    build_full_space_formulation(block, network_structure, skip_activations=True)

    net = network_structure
    linear_nodes = list()
    relu_nodes = list()
    activations = net.activations
    # block.activation_constraints = pyo.Constraint(block.hidden_output_nodes)
    for i in block.hidden_output_nodes:
        if i not in activations or activations[i] is None or activations[i] == "linear":
            linear_nodes.append(i)
        elif activations[i] == "relu":
            relu_nodes.append(i)
        else:
            raise ValueError(
                "Activation function {} not supported in the ReLU formulation".format(
                    activations[i]
                )
            )

    block.relu_nodes = pyo.Set(initialize=relu_nodes, ordered=True)
    block.linear_nodes = pyo.Set(initialize=linear_nodes, ordered=True)

    # # activation indicator q=0 means z=zhat (positive part of the hinge)
    # # q=1 means we are on the zero part of the hinge
    # block.hidden_nodes = net.hidden_node_ids()
    block.q = pyo.Var(block.relu_nodes, within=pyo.Binary)
    block._z_lower_bound = pyo.Constraint(block.relu_nodes)
    block._z_hat_bound = pyo.Constraint(block.relu_nodes)
    block._z_hat_positive = pyo.Constraint(block.relu_nodes)
    block._z_hat_negative = pyo.Constraint(block.relu_nodes)
    block._linear_activation = pyo.Constraint(block.linear_nodes)

    # relu logic
    for i in block.relu_nodes:
        block._z_lower_bound[i] = block.z[i] >= 0
        block._z_hat_bound[i] = block.z[i] >= block.zhat[i]
        block._z_hat_positive[i] = block.z[i] <= block.zhat[i] + M * block.q[i]
        block._z_hat_negative[i] = block.z[i] <= M * (1.0 - block.q[i])

    # linear activations
    for i in block.linear_nodes:
        block._linear_activation[i] = block.z[i] == block.zhat[i]


def build_relu_complementarity_formulation(
    block, network_structure, transform="mpec.simple_nonlinear"
):
    # build the full space structure without activations
    build_full_space_formulation(block, network_structure, skip_activations=True)

    net = network_structure
    linear_nodes = list()
    relu_nodes = list()
    activations = net.activations
    # block.activation_constraints = pyo.Constraint(block.hidden_output_nodes)
    for i in block.hidden_output_nodes:
        if i not in activations or activations[i] is None or activations[i] == "linear":
            linear_nodes.append(i)
        elif activations[i] == "relu":
            relu_nodes.append(i)
        else:
            raise ValueError(
                "Activation function {} not supported in the ReLU formulation".format(
                    activations[i]
                )
            )

    block.relu_nodes = pyo.Set(initialize=relu_nodes, ordered=True)
    block.linear_nodes = pyo.Set(initialize=linear_nodes, ordered=True)

    block._complementarity = mpec.Complementarity(block.relu_nodes)
    block._linear_activation = pyo.Constraint(block.linear_nodes)

    # relu logic
    for i in block.relu_nodes:
        block._complementarity[i] = mpec.complements(
            (block.z[i] - block.zhat[i]) >= 0, block.z[i] >= 0
        )
    xfrm = pyo.TransformationFactory(transform)
    xfrm.apply_to(block)

    # linear activations
    for i in block.linear_nodes:
        block._linear_activation[i] = block.z[i] == block.zhat[i]
