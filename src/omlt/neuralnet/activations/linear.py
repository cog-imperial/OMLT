from omlt.base import OmltConstraint


def linear_activation_function(zhat):
    return zhat


def linear_activation_constraint(
    net_block, net, layer_block, layer, *, add_constraint=True
):
    r"""Linear activation constraint generator.

    Generates the constraints for the linear activation function:

    .. math::

        \begin{align*}
            y=x
        \end{align*}

    """
    layer_block.linear_activation = OmltConstraint(
        layer.output_indexes, lang=net_block._format
    )
    for output_index in layer.output_indexes:
        zhat_lb, zhat_ub = layer_block.zhat[output_index].bounds
        layer_block.z[output_index].setlb(zhat_lb)
        layer_block.z[output_index].setub(zhat_ub)
        layer_block.linear_activation[output_index] = (
            layer_block.z[output_index] == layer_block.zhat[output_index]
        )
