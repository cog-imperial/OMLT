from pyomo.contrib.fbbt.interval import exp
import pyomo.environ as pyo
import pyomo.mpec as mpec
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from ..formulation import _PyomoFormulation
from .full_space import build_full_space_formulation


class ReLUBigMFormulation(_PyomoFormulation):
    def __init__(self, network_structure):
        """This class provides a full-space formulation of a neural network with ReLU
        activation functions using a MILP representation.
        """
        super(ReLUBigMFormulation, self).__init__(network_structure)

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_relu_mip_formulation(
            block=self.block,
            network_structure=self.network_definition
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
    layers = list(net.nodes)

    for layer_no, layer in enumerate(layers):
        layer_block = block.layer[layer_no]
        if not hasattr(layer_block, "__dense_expr"):
            continue

        if layer.activation == "relu":
            layer_block.q = pyo.Var(layer.output_indexes, within=pyo.Binary)

            layer_block._z_lower_bound = pyo.Constraint(layer.output_indexes)
            layer_block._z_lower_bound_zhat = pyo.Constraint(layer.output_indexes)
            layer_block._z_upper_bound = pyo.Constraint(layer.output_indexes)
            layer_block._z_upper_bound_zhat = pyo.Constraint(layer.output_indexes)

            # set dummy parameters here to avoid warning message from Pyomo
            layer_block._big_m_lb = pyo.Param(layer.output_indexes, default=-1e6, mutable=True)
            layer_block._big_m_ub = pyo.Param(layer.output_indexes, default=1e6, mutable=True)
        elif layer.activation == "linear":
            @layer_block.Constraint(layer.output_indexes)
            def _linear_activation(b, *output_index):
                return b.z[output_index] == b.zhat[output_index]

        for output_index, expr in layer_block.__dense_expr:
            lb, ub = compute_bounds_on_expr(expr)

            assert lb is not None
            assert ub is not None

            # propagate bounds
            if layer.activation == "linear":
                layer_block.z[output_index].setlb(lb)
                layer_block.z[output_index].setub(ub)
            elif layer.activation == "relu":
                # Relu formulation based on Equation 4 from
                # Bunel, Rudy, et al. "Branch and bound for piecewise linear neural network verification."
                # Journal of Machine Learning Research 21.2020 (2020).
                layer_block._big_m_lb[output_index] = lb
                layer_block.z[output_index].setlb(lb)

                layer_block._big_m_ub[output_index] = ub
                layer_block.z[output_index].setub(ub)

                layer_block._z_lower_bound[output_index] = (
                    layer_block.z[output_index] >= 0
                )

                layer_block._z_lower_bound_zhat[output_index] = (
                    layer_block.z[output_index] >= layer_block.zhat[output_index]
                )

                layer_block._z_upper_bound[output_index] = (
                    layer_block.z[output_index] <= layer_block._big_m_ub[output_index] * layer_block.q[output_index]
                )

                layer_block._z_upper_bound_zhat[output_index] = (
                    layer_block.z[output_index] <= layer_block.zhat[output_index] - layer_block._big_m_lb[output_index] * (1.0 - layer_block.q[output_index])
                )



def build_relu_complementarity_formulation(
    block, network_structure, transform="mpec.simple_nonlinear"
):
    # build the full space structure without activations
    build_full_space_formulation(block, network_structure, skip_activations=True)

    net = network_structure
    layers = list(net.nodes)

    for layer_no, layer in enumerate(layers):
        layer_block = block.layer[layer_no]
        if layer.activation == "linear":
            @layer_block.Constraint(layer.output_indexes)
            def _linear_activation(b, *output_index):
                return b.z[output_index] == b.zhat[output_index]

        elif layer.activation == "relu":
            layer_block._complementarity = mpec.Complementarity(
                layer.output_indexes,
                rule=_relu_complementarity
            )
        xfrm = pyo.TransformationFactory(transform)
        xfrm.apply_to(layer_block)


def _relu_complementarity(b, *output_index):
    return mpec.complements(
        b.z[output_index] - b.zhat[output_index] >= 0,
        b.z[output_index] >= 0
    )