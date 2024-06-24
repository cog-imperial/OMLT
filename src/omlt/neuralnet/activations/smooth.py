from pyomo.environ import exp, log, tanh


def softplus_activation_function(x):
    r"""Applies the softplus function.

    .. math::

        \begin{align*}
            y=\log(\exp(x)+1)
        \end{align*}

    """
    return log(exp(x) + 1)


def sigmoid_activation_function(x):
    r"""Applies the sigmoid function.

    .. math::

        \begin{align*}
            y=\frac{1}{1+\exp(-x)}
        \end{align*}

    """
    return 1 / (1 + exp(-x))


def tanh_activation_function(x):
    r"""Applies the tanh function.

    .. math::

        \begin{align*}
            y=\tanh(x)
        \end{align*}

    """
    return tanh(x)


def softplus_activation_constraint(net_block, net, layer_block, layer):
    r"""Softplus activation constraint generator."""
    return smooth_monotonic_activation_constraint(
        net_block, net, layer_block, layer, softplus_activation_function
    )


def sigmoid_activation_constraint(net_block, net, layer_block, layer):
    r"""Sigmoid activation constraint generator."""
    return smooth_monotonic_activation_constraint(
        net_block, net, layer_block, layer, sigmoid_activation_function
    )


def tanh_activation_constraint(net_block, net, layer_block, layer):
    r"""Tanh activation constraint generator."""
    return smooth_monotonic_activation_constraint(
        net_block, net, layer_block, layer, tanh_activation_function
    )


def smooth_monotonic_activation_constraint(net_block, net, layer_block, layer, fcn):
    r"""Activation constraint generator for a smooth monotonic function.

    Generates the constraints for the activation function :math:`f` if it is smooth and
    monotonic:

    .. math::

        \begin{align*}
            y=f(x)
        \end{align*}

    """

    @layer_block.Constraint(layer.output_indexes)
    def _smooth_monotonic_activation_constraint(b, *output_index):
        zhat_lb, zhat_ub = b.zhat[output_index].bounds
        if zhat_lb is not None:
            b.z[output_index].setlb(fcn(zhat_lb))
        if zhat_ub is not None:
            b.z[output_index].setub(fcn(zhat_ub))
        return b.z[output_index] == fcn(b.zhat[output_index])
