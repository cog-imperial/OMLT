def reduced_space_dense_layer(net_block, net, layer_block, layer, activation):
    r"""Add reduced-space formulation of the dense layer to the block.

    .. math::

        \begin{align*}
        \hat z_i &= \sum_{j{=}1}^{M_i} w_{ij} z_j + b_i  && \forall i \in N
        \end{align*}

    """
    # not an input layer, process the expressions
    prev_layers = list(net.predecessors(layer))
    if len(prev_layers) == 0:
        msg = f"Layer {layer} is not an input layer, but has no predecessors."
        raise ValueError(msg)
    if len(prev_layers) > 1:
        msg = f"Layer {layer} has multiple predecessors."
        raise ValueError(msg)
    prev_layer = prev_layers[0]
    prev_layer_block = net_block.layer[id(prev_layer)]

    @layer_block.Expression(layer.output_indexes)
    def zhat(b, *output_index):
        # dense layers multiply only the last dimension of
        # their inputs
        expr = 0.0
        for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
            w = layer.weights[local_index[-1], output_index[-1]]
            expr += prev_layer_block.z[input_index] * w
        # move this at the end to avoid numpy/pyomo var bug
        expr += layer.biases[output_index[-1]]

        return expr

    @layer_block.Expression(layer.output_indexes)
    def z(b, *output_index):
        return activation(b.zhat[output_index])
