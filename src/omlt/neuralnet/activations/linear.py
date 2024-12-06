from omlt.base import OmltConstraintFactory, DEFAULT_MODELING_LANGUAGE


def linear_activation_function(zhat, lang=DEFAULT_MODELING_LANGUAGE):
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
    constraint_factory = OmltConstraintFactory()
    layer_block.linear_activation = constraint_factory.new_constraint(
        layer.output_indexes, lang=net_block._format
    )
    for output_index in layer.output_indexes:
        zhat_lb, zhat_ub = layer_block.zhat[output_index].bounds
        layer_block.z[output_index].setlb(zhat_lb)
        layer_block.z[output_index].setub(zhat_ub)
        layer_block.linear_activation[output_index] = (
            layer_block.z[output_index] == layer_block.zhat[output_index]
        )
