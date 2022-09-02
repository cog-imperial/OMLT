import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from omlt.neuralnet.activations import NON_INCREASING_ACTIVATIONS
from omlt.neuralnet.layer import ConvLayer, PoolingLayer

# TODO: Change asserts to exceptions with messages (or ensure they
# TODO:      are trapped higher up the call stack)
def full_space_dense_layer(net_block, net, layer_block, layer):
    r"""
    Add full-space formulation of the dense layer to the block

    .. math::

        \begin{align*}
        \hat z_i &= \sum_{j{=}1}^{M_i} w_{ij} z_j + b_i  && \forall i \in N
        \end{align*}

    """
    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)

    @layer_block.Constraint(layer.output_indexes)
    def dense_layer(b, *output_index):
        # dense layers multiply only the last dimension of
        # their inputs
        expr = 0.0
        for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
            w = layer.weights[local_index[-1], output_index[-1]]
            expr += input_layer_block.z[input_index] * w
        # move this at the end to avoid numpy/pyomo var bug
        expr += layer.biases[output_index[-1]]

        lb, ub = compute_bounds_on_expr(expr)
        layer_block.zhat[output_index].setlb(lb)
        layer_block.zhat[output_index].setub(ub)

        return layer_block.zhat[output_index] == expr


def full_space_conv_layer(net_block, net, layer_block, layer):
    r"""
    Add full-space formulation of the 2-D convolutional layer to the block

    A 2-D convolutional layer applies cross-correlation kernels to a 2-D input.
    Specifically, the input is convolved by sliding the kernels along the input vertically and horizontally.
    At each location, the preactivation is computed as the dot product of the kernel weights and the input plus a bias term.

    """ 
    # If activation is an increasing function,
    #  move it onto successor max pooling layer (if it exists) for tighter max pooling formulation
    if layer.activation not in NON_INCREASING_ACTIVATIONS:
        succ_layers = list(net.successors(layer))
        if len(succ_layers) == 1:
            succ_layer = succ_layers[0]
            if isinstance(succ_layer, PoolingLayer):
                # activation applied after convolution layer, so there shouldn't be an activation after max pooling too
                assert succ_layer.activation == "linear"
                succ_layer.activation = layer.activation
                layer.activation = "linear"
    
    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)

    # for out_d, out_r, out_c in layer.output_indexes:
    #   output_index = (out_d, out_r, out_c)
    @layer_block.Constraint(layer.output_indexes)
    def convolutional_layer(b, *output_index):
        out_d, out_r, out_c = output_index
        expr = 0.0
        for weight, input_index in layer.kernel_with_input_indexes(out_d, out_r, out_c):
            expr += weight * input_layer_block.z[input_index]

        lb, ub = compute_bounds_on_expr(expr)
        layer_block.zhat[output_index].setlb(lb)
        layer_block.zhat[output_index].setub(ub)
        # layer_block.constraints.add(layer_block.zhat[output_index] == expr)
        return layer_block.zhat[output_index] == expr

def full_space_maxpool_layer(net_block, net, layer_block, layer):
    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)
    assert isinstance(input_layer, ConvLayer)
    assert input_layer.activation == "linear" # TODO - add support for non-increasing activation functions on preceding convolutional layer

    num_kernel_elements = np.product(layer.kernel_shape)
    assert num_kernel_elements >= 1
    layer_block.flat_input_indexes = pyo.RangeSet(1, num_kernel_elements)
    layer_block.q = pyo.Var(layer.output_indexes, layer_block.flat_input_indexes, within=pyo.Binary)
    layer_block._z_upper_bound = pyo.Constraint(layer_block.output_indexes, layer_block.flat_input_indexes)
    layer_block._z_lower_bound = pyo.Constraint(layer_block.output_indexes, layer_block.flat_input_indexes)
    n_plus = np.full((num_kernel_elements, num_kernel_elements), 1000)
    np.fill_diagonal(n_plus, 0)

    for output_index in layer.output_indexes:
        out_d, out_r, out_c = output_index

        # cannot compute an expr for the max,
        # as pyomo expressions cannot contain functions whose output depends on a comparison (except piecewise linear functions)
        # so compute lb and ub directly
        bounds = (input_layer_block.z[input_index].bounds for _, input_index in layer.kernel_index_with_input_indexes(out_d, out_r, out_c))
        lbs, ubs = zip(*bounds)
        layer_block.zhat[output_index].setlb(max(lbs))
        layer_block.zhat[output_index].setub(max(ubs))

        l = 1
        for _, input_index in layer.kernel_index_with_input_indexes(out_d, out_r, out_c):
            input_index_mapper = layer.input_index_mapper
            if input_index_mapper is None:
                input_index_mapper = lambda x: x
            input_layer_index = input_index_mapper.inverse(input_index)

            layer_block._z_upper_bound[output_index, l] = (
                layer_block.zhat[output_index] <= input_layer_block.z[input_layer_index] + sum(layer_block.q[output_index, k] * n_plus[l - 1, k - 1] for k in layer_block.flat_input_indexes)
            )
            layer_block._z_lower_bound[output_index, l] = (
                layer_block.zhat[output_index] >= input_layer_block.z[input_layer_index]
            )

            l += 1

def _input_layer_and_block(net_block, net, layer):
    input_layers = list(net.predecessors(layer))
    assert len(input_layers) == 1
    input_layer = input_layers[0]
    input_layer_block = net_block.layer[id(input_layer)]
    return input_layer, input_layer_block
