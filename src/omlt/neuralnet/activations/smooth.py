from pyomo.environ import exp, log, tanh

def softplus_activation_function(x):
    return log(exp(x) + 1)

def sigmoid_activation_function(x):
    return 1 / (1 + exp(-x))

def tanh_activation_function(x):
    return tanh(x)

def softplus_activation_constraint(net_block, net, layer_block, layer):
    """
    Softplus activation constraint generator

    Generates the constraints for the softplus activation function.

    .. math::

        \begin{align*}
        z_i &= log(exp(\hat z_i) + 1) && \forall i \in N
        \end{align*}

    """
    return smooth_monotonic_activation_constraint(net_block, net, layer_block, layer, softplus_activation_function)

def sigmoid_activation_constraint(net_block, net, layer_block, layer):
    """
    Sigmoid activation constraint generator

    Generates the constraints for the sigmoid activation function.

    .. math::

        \begin{align*}
        z_i &= \frac{1}{1 + exp(-\hat z_i)} && \forall i \in N
        \end{align*}

    """
    return smooth_monotonic_activation_constraint(net_block, net, layer_block, layer,
                                                  sigmoid_activation_function)

def tanh_activation_constraint(net_block, net, layer_block, layer):
    """
    tanh activation constraint generator

    Generates the constraints for the tanh activation function.

    .. math::

        \begin{align*}
        z_i &= tanh(\hat z_i) && \forall i \in N
        \end{align*}

    """
    return smooth_monotonic_activation_constraint(net_block, net, layer_block, layer, tanh_activation_function)

def smooth_monotonic_activation_constraint(net_block, net, layer_block, layer, fcn):
    """
    Activation constraint generator for a smooth monotonic function

    Generates the constraints for the activation function fcn if it
    is smooth and monotonic

    .. math::

        \begin{align*}
        z_i &= fcn(\hat z_i) && \forall i \in N
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
