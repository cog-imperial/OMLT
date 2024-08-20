import pyomo.environ as pyo
from pyomo import mpec


def bigm_relu_activation_constraint(net_block, net, layer_block, layer):
    r"""Big-M ReLU activation formulation.

    Generates the constraints for the ReLU activation function:

    .. math::

        \begin{align*}
            y=\max(0,x)
        \end{align*}

    We additionally introduce the following notations to describe this formulation:

    .. math::

        \begin{align*}
            \sigma &:= \text{denote if $y=x$, $\sigma\in\{0,1\}$}\\
            l      &:= \text{the lower bound of $x$}\\
            u      &:= \text{the upper bound of $x$}\\
        \end{align*}

    The big-M formulation is given by:

    .. math::

        \begin{align*}
            y&\ge 0\\
            y&\ge x\\
            y&\le x-(1-\sigma)l\\
            y&\le \sigma u
        \end{align*}

    The lower bound of :math:`y` is :math:`\max(0,l)`, and the upper bound of :math:`y`
    is :math:`\max(0,u)`.

    """
    layer_block.q_relu = pyo.Var(layer.output_indexes, within=pyo.Binary)

    layer_block._z_lower_bound_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_lower_bound_zhat_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_relu = pyo.Constraint(layer.output_indexes)
    layer_block._z_upper_bound_zhat_relu = pyo.Constraint(layer.output_indexes)

    # set dummy parameters here to avoid warning message from Pyomo
    layer_block._big_m_lb_relu = pyo.Param(
        layer.output_indexes, default=-1e6, mutable=True
    )
    layer_block._big_m_ub_relu = pyo.Param(
        layer.output_indexes, default=1e6, mutable=True
    )

    for output_index in layer.output_indexes:
        lb, ub = layer_block.zhat[output_index].bounds
        layer_block._big_m_lb_relu[output_index] = lb
        layer_block.z[output_index].setlb(max(0, lb))

        layer_block._big_m_ub_relu[output_index] = ub
        layer_block.z[output_index].setub(max(0, ub))

        layer_block._z_lower_bound_relu[output_index] = layer_block.z[output_index] >= 0

        layer_block._z_lower_bound_zhat_relu[output_index] = (
            layer_block.z[output_index] >= layer_block.zhat[output_index]
        )

        layer_block._z_upper_bound_relu[output_index] = (
            layer_block.z[output_index]
            <= layer_block._big_m_ub_relu[output_index]
            * layer_block.q_relu[output_index]
        )

        layer_block._z_upper_bound_zhat_relu[output_index] = layer_block.z[
            output_index
        ] <= layer_block.zhat[output_index] - layer_block._big_m_lb_relu[
            output_index
        ] * (1.0 - layer_block.q_relu[output_index])


class ComplementarityReLUActivation:
    r"""Complementarity-based ReLU activation formulation.

    Generates the constraints for the ReLU activation function:

    .. math::

        \begin{align*}
            y=\max(0,x)
        \end{align*}

    The complementarity-based formulation is given by:

    .. math::

        \begin{align*}
            0\le y \perp (y-x)\ge 0
        \end{align*}

    which denotes that:

    .. math::

        \begin{align*}
            y\ge 0\\
            y(y-x)=0\\
            y-x\ge 0
        \end{align*}

    """

    def __init__(self, transform=None):
        if transform is None:
            transform = "mpec.simple_nonlinear"
        self.transform = transform

    def __call__(self, net_block, net, layer_block, layer):  # noqa: ARG002
        layer_block._complementarity = mpec.Complementarity(
            layer.output_indexes, rule=_relu_complementarity
        )
        xfrm = pyo.TransformationFactory(self.transform)
        xfrm.apply_to(layer_block)


def _relu_complementarity(b, *output_index):
    return mpec.complements(
        b.z[output_index] - b.zhat[output_index] >= 0, b.z[output_index] >= 0
    )
