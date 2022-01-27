import pyomo.environ as pyo
import numpy as np
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

# TODO: Change asserts to exceptions with messages (or ensure they
# TODO:      are trapped higher up the call stack)
def full_space_dense_layer(net_block, net, layer_block, layer):
    """
    Add full-space formulation of the dense layer to the block

    .. math::

        \begin{align*}
        \hat z_i &= \sum_{j{=}1}^N w_{ij} z_j + b_i  &&\forall i \in N
        \end{align*}

    """
    input_layers = list(net.predecessors(layer))
    assert len(input_layers) == 1
    input_layer = input_layers[0]
    input_layer_block = net_block.layer[id(input_layer)]

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
    input_layers = list(net.predecessors(layer))
    assert len(input_layers) == 1
    input_layer = input_layers[0]
    input_layer_block = net_block.layer[id(input_layer)]

    #for out_d, out_r, out_c in layer.output_indexes:
    #   output_index = (out_d, out_r, out_c)
    @layer_block.Constraint(layer.output_indexes)
    def convolutional_layer(b,*output_index):
        out_d, out_r, out_c = output_index
        expr = 0.0
        for weight, input_index in layer.kernel_with_input_indexes(out_d, out_r, out_c):
            expr += weight * input_layer_block.z[input_index]

        lb, ub = compute_bounds_on_expr(expr)
        layer_block.zhat[output_index].setlb(lb)
        layer_block.zhat[output_index].setub(ub)
        #layer_block.constraints.add(layer_block.zhat[output_index] == expr)
        return layer_block.zhat[output_index] == expr