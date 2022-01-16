import pyomo.environ as pyo
import pyomo.mpec as mpec

def bigm_relu_activation_constraint(net_block, net, layer_block, layer):
    """
    Big-M ReLU activation formulation.
    """
    layer_block.q = pyo.Var(layer.output_indexes, within=pyo.Binary)

    layer_block._z_lower_bound = pyo.Constraint(layer.output_indexes)
    layer_block._z_lower_bound_zhat = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_zhat = pyo.Constraint(layer.output_indexes)

    # set dummy parameters here to avoid warning message from Pyomo
    layer_block._big_m_lb = pyo.Param(layer.output_indexes, default=-1e6, mutable=True)
    layer_block._big_m_ub = pyo.Param(layer.output_indexes, default=1e6, mutable=True)

    for output_index in layer.output_indexes:
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block._big_m_lb[output_index] = lb
        layer_block.z[output_index].setlb(0)

        layer_block._big_m_ub[output_index] = ub
        layer_block.z[output_index].setub(max(0, ub))

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


class ComplementarityReLUActivation:
    """
    Complementarity-based ReLU activation forumlation.
    """
    def __init__(self, transform = None):
        if transform is None:
            transform = "mpec.simple_nonlinear"
        self.transform = transform

    def __call__(self, net_block, net, layer_block, layer):
        layer_block._complementarity = mpec.Complementarity(
            layer.output_indexes,
            rule=_relu_complementarity
        )
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(layer_block)


def _relu_complementarity(b, *output_index):
    return mpec.complements(
        b.z[output_index] - b.zhat[output_index] >= 0,
        b.z[output_index] >= 0
    )
