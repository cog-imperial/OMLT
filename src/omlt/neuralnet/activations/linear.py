def linear_activation_function(zhat):
    return zhat

def linear_activation_constraint(net_block, net, layer_block, layer, add_constraint=True):
    """
    Linear activation constraint generator

    Generates the constraints for the linear activation function.

    .. math::

        \begin{align*}
        z_i &= \hat{z_i} && \forall i \in N
        \end{align*}

    """
    @layer_block.Constraint(layer.output_indexes)
    def linear_activation(b, *output_index):
        zhat_lb, zhat_ub = b.zhat[output_index].bounds
        b.z[output_index].setlb(zhat_lb)
        b.z[output_index].setub(zhat_ub)
        return b.z[output_index] == b.zhat[output_index]


