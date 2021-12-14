from warnings import warn

import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from ..formulation import _PyomoFormulation
from .full_space import build_full_space_formulation


class ReLUBigMFormulation(_PyomoFormulation):
    def __init__(self, network_structure, M=None):
        """This class provides a full-space formulation of a neural network with ReLU
        activation functions using a MILP representation.
        """
        super(ReLUBigMFormulation, self).__init__(network_structure)
        self.M = M

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
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
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_relu_complementarity_formulation(
            block=self.block,
            network_structure=self.network_definition,
            transform=self.transform,
        )


def build_relu_mip_formulation(block, network_structure, M=None):
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
    block._big_m_lb = pyo.Param(block.relu_nodes, default=None, mutable=True)
    block._big_m_ub = pyo.Param(block.relu_nodes, default=None, mutable=True)
    block._linear_activation = pyo.Constraint(block.linear_nodes)

    w = net.weights
    b = net.biases
    # relu logic
    for i in block.relu_nodes:
        # \hat z = w^T x + b
        zhat_def = sum(w[i][j] * block.z[j] for j in w[i]) + b[i]
        lb, ub = compute_bounds_on_expr(zhat_def)

        if lb is not None:
            # z = max(0, \hat z)
            # so lower bound is bounded by 0
            block._big_m_lb[i] = lb
            # lb = max(0.0, lb)
            block.z[i].setlb(lb)
        else:
            # use default big-m
            if M is None:
                raise ValueError("could not propagate bounds and M is None")
            warn("setting relu big M (lb) to {}".format(M))
            block._big_m_lb[i] = -M

        if ub is not None:
            # the upper bound should always be at least 0
            block.z[i].setub(max(ub, 0))
            # use upper bound on z as big-m
            block._big_m_ub[i] = ub
        else:
            # use default big-m
            if M is None:
                raise ValueError("could not propagate bounds and M is None")
            warn("setting relu big M (ub) to {}".format(M))
            block._big_m_ub[i] = M

        
        # block._z_lower_bound[i] = block.z[i] >= block._big_m_ub[i] * (1.0 - block.q[i])
        # block._z_hat_bound[i] = (
        #     block.z[i] >= block.zhat[i] - block._big_m_ub[i] * block.q[i]
        # )

        block._z_lower_bound[i] = block.z[i] >= 0 
        block._z_hat_bound[i] = block.z[i] >= block.zhat[i]

        block._z_hat_positive[i] = (
            block.z[i] <= block.zhat[i] - block._big_m_lb[i] * block.q[i]
        )
        block._z_hat_negative[i] = block.z[i] <= block._big_m_ub[i] * (1.0 - block.q[i])

    # linear activations
    for i in block.linear_nodes:
        zhat_def = sum(w[i][j] * block.z[j] for j in w[i]) + b[i]
        lb, ub = compute_bounds_on_expr(zhat_def)
        if lb is not None:
            block.z[i].setlb(lb)
        if ub is not None:
            block.z[i].setub(ub)
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
