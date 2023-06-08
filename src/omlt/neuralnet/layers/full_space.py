import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from omlt.neuralnet.activations import NON_INCREASING_ACTIVATIONS
from omlt.neuralnet.layer import ConvLayer2D, IndexMapper, PoolingLayer2D


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


def full_space_gnn_layer_bilinear(net_block, net, layer_block, layer):
    r"""
    Add full-space formulation of the gnn layer to the block

    .. math::

        \begin{align*}
        & \hat z_i = \sum_{j{=}1}^{M_i} A_{v_i,v_j} w_{ij} z_j + b_i  && \forall i \in N \\
        & A_{v_i,v_j}\in\{0,1\}
        \end{align*}

    where :math:`v_i` is the graph indexing of the i-th node.

    """

    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)

    input_channel = layer.input_size[0] // net_block.N
    output_channel = layer.output_size[0] // net_block.N

    @layer_block.Constraint(layer.output_indexes)
    def gnn_layer_bilinear_formulation(b, *output_index):
        # dense layers multiply only the last dimension of
        # their inputs
        expr = 0.0
        for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
            w = layer.weights[input_index[-1], output_index[-1]]

            input_node_index = input_index[-1] // input_channel
            output_node_index = output_index[-1] // output_channel

            expr += (
                input_layer_block.z[input_index]
                * w
                * net_block.A[input_node_index, output_node_index]
            )
        # move this at the end to avoid numpy/pyomo var bug
        output_node_index = output_index[-1] // output_channel
        expr += layer.biases[output_index[-1]]

        lb, ub = compute_bounds_on_expr(expr)
        layer_block.zhat[output_index].setlb(lb)
        layer_block.zhat[output_index].setub(ub)

        return layer_block.zhat[output_index] == expr


from omlt.neuralnet.layer import InputLayer


def full_space_gnn_layer_bigm(net_block, net, layer_block, layer):
    r"""
    Add full-space formulation of the gnn layer to the block

    .. math::

        \begin{align*}
        \hat z_i &= \sum_{j{=}1}^{M_i} w_{ij} \bar z_{ij} + b_i  && \forall i \in N \\
        \bar z_{ij} &= A_{v_i,v_j} z_{j}
        \end{align*}

    The big-M formulation for :math:`\bar z_{ij}` is given by:

    .. math::

        \begin{align*}
        z_{j} - M_{j}(1-A_{v_i,v_j}) &\le \bar z_{ij} \le z_{j} + M_{j}(1-A_{v_i,v_j})\\
        - M_{j}A_{v_i,v_j} &\le \bar z_{ij} \le M_{j}A_{v_i,v_j}\\
        A_{v_i,v_j}&\in \{0,1\}
        \end{align*}

    where :math:`M_{j}` is upper bound of :math:`|z_{j}|`. 
    """

    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)
    input_channel = layer.input_size[0] // net_block.N
    output_channel = layer.output_size[0] // net_block.N

    input_layer_block.zbar = pyo.Var(
        pyo.Set(initialize=range(layer.input_size[0])),
        pyo.Set(initialize=range(net_block.N)),
        initialize=0,
    )

    input_layer_block._zbar_lower_bound_z_big_m = pyo.Constraint(
        pyo.Set(initialize=range(layer.input_size[0] * net_block.N))
    )
    input_layer_block._zbar_upper_bound_z_big_m = pyo.Constraint(
        pyo.Set(initialize=range(layer.input_size[0] * net_block.N))
    )
    input_layer_block._zbar_lower_bound_big_m = pyo.Constraint(
        pyo.Set(initialize=range(layer.input_size[0] * net_block.N))
    )
    input_layer_block._zbar_upper_bound_big_m = pyo.Constraint(
        pyo.Set(initialize=range(layer.input_size[0] * net_block.N))
    )

    # set dummy parameters here to avoid warning message from Pyomo
    input_layer_block._abs_bound_big_m = pyo.Param(
        input_layer.output_indexes, default=1e6, mutable=True
    )

    for input_index in layer.input_indexes:
        lb, ub = input_layer_block.z[input_index].bounds
        input_layer_block._abs_bound_big_m[input_index] = max(abs(lb), abs(ub))

        for output_node_index in range(net_block.N):
            input_layer_block.zbar[input_index, output_node_index].setlb(min(0, lb))
            input_layer_block.zbar[input_index, output_node_index].setub(max(0, ub))

            input_node_index = input_index[-1] // input_channel

            constraint_index = input_index[-1] * net_block.N + output_node_index
            input_layer_block._zbar_lower_bound_z_big_m[
                constraint_index
            ] = input_layer_block.zbar[
                input_index, output_node_index
            ] >= input_layer_block.z[
                input_index
            ] - input_layer_block._abs_bound_big_m[
                input_index
            ] * (
                1.0 - net_block.A[input_node_index, output_node_index]
            )

            input_layer_block._zbar_upper_bound_z_big_m[
                constraint_index
            ] = input_layer_block.zbar[
                input_index, output_node_index
            ] <= input_layer_block.z[
                input_index
            ] + input_layer_block._abs_bound_big_m[
                input_index
            ] * (
                1.0 - net_block.A[input_node_index, output_node_index]
            )

            input_layer_block._zbar_lower_bound_big_m[constraint_index] = (
                input_layer_block.zbar[input_index, output_node_index]
                >= -input_layer_block._abs_bound_big_m[input_index]
                * net_block.A[input_node_index, output_node_index]
            )

            input_layer_block._zbar_upper_bound_big_m[constraint_index] = (
                input_layer_block.zbar[input_index, output_node_index]
                <= input_layer_block._abs_bound_big_m[input_index]
                * net_block.A[input_node_index, output_node_index]
            )

    # input_layer_block._zbar_lower_bound_z_big_m.pprint()
    # input_layer_block._zbar_upper_bound_z_big_m.pprint()
    # input_layer_block._zbar_lower_bound_big_m.pprint()
    # input_layer_block._zbar_upper_bound_big_m.pprint()

    @layer_block.Constraint(layer.output_indexes)
    def dense_layer(b, *output_index):
        # dense layers multiply only the last dimension of
        # their inputs
        expr = 0.0
        for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
            w = layer.weights[input_index[-1], output_index[-1]]

            input_node_index = input_index[-1] // input_channel
            output_node_index = output_index[-1] // output_channel

            expr += input_layer_block.zbar[input_index, output_node_index] * w
        # move this at the end to avoid numpy/pyomo var bug
        output_node_index = output_index[-1] // output_channel
        expr += layer.biases[output_index[-1]]

        lb, ub = compute_bounds_on_expr(expr)
        layer_block.zhat[output_index].setlb(lb)
        layer_block.zhat[output_index].setub(ub)
        # print(layer_block.zhat[output_index] == expr)
        return layer_block.zhat[output_index] == expr


def full_space_conv2d_layer(net_block, net, layer_block, layer):
    r"""
    Add full-space formulation of the 2-D convolutional layer to the block

    A 2-D convolutional layer applies cross-correlation kernels to a 2-D input.
    Specifically, the input is convolved by sliding the kernels along the input vertically and horizontally.
    At each location, the preactivation is computed as the dot product of the kernel weights and the input plus a bias term.

    """
    # If activation is an increasing function,
    #  move it onto successor max pooling layer (if it exists) for tighter max pooling formulation
    succ_layers = list(net.successors(layer))
    succ_layer = succ_layers[0] if len(succ_layers) == 1 else None
    if (
        isinstance(succ_layer, PoolingLayer2D)
        and layer.activation not in NON_INCREASING_ACTIVATIONS
        and layer.activation != "linear"
    ):
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


def full_space_maxpool2d_layer(net_block, net, layer_block, layer):
    r"""
    Add Big-M max pooling formulation.

    .. math::

        \begin{align*}
            \hat{z_i} \leq w\cdot x_{i}^{l} + \sum_{k{=}1}^{d} M_{i}^{l,k} q_{i}^{k} &&
            \forall i \in N,\ \forall l \in \{ 1,...,d \} \\
            \hat{z_i} \geq w\cdot x_{i}^{l} && \forall i \in N,\ \forall l \in \{ 1,...,d \} \\
            (x_{i},\hat{z_i},q_{i}) \in [L_{i},U_{i}] \times \mathbb{R} \times \Delta^{d} && \forall i \in N \\
            q_{i} \in \{ 0,1 \}^{d} && \forall i \in N \\
            M_{i}^{l,k} = w\cdot max\{ L_{i}^{l} - L_{i}^{k},
            \\ L_{i}^{l} - U_{i}^{k}, U_{i}^{l} - L_{i}^{k}, U_{i}^{l} - U_{i}^{k} \}
            && \forall i \in N,\ \forall l \in \{ 1,...,d \},\ \forall k \in \{ 1,...,d \}
        \end{align*}

    where :math:`w` is the convolution kernel on the preceding convolutional layer; :math:`d` is the number of features
    in each of the :math:`N` max pooling windows;
    :math:`x_{i}` is the set of :math:`d` features in the :math:`i`-th max pooling window;
    :math:`\Delta^{d}` is the :math:`d`-dimensional simplex; and [L_{i},U_{i}] are the bounds on x_{i}.

    NOTE This formulation is adapted from the Anderson et al. (2020) formulation, section 5.1, with the following changes:

    - OMLT presently does not support biases on convolutional layers. Bias terms from the original formulation
      are removed.

    - The original formulation formulates the max of :math:`w^{l}\cdot x + b^{l}`, varying the weights :math:`w`
      and biases :math:`b` and keeping the input :math:`x` constant. Since convolutional layers have constant weights
      and biases convolved with varying portions of the feature map, this formulation formulates the max of
      :math:`w\cdot x^{l} + b`.

    - Due to the above 2 changes, the calculation of :math:`N^{l,k}` is changed.

    """
    input_layer, input_layer_block = _input_layer_and_block(net_block, net, layer)
    assert isinstance(input_layer, ConvLayer2D)
    assert (
        input_layer.activation == "linear"
    )  # TODO - add support for non-increasing activation functions on preceding convolutional layer

    # note kernel indexes are the same set of values for any output index, so wlog get kernel indexes for (0, 0, 0)
    layer_block._kernel_indexes = pyo.Set(
        initialize=(
            kernel_index
            for kernel_index, _ in layer.kernel_index_with_input_indexes(0, 0, 0)
        )
    )
    layer_block.q_maxpool = pyo.Var(
        layer.output_indexes, layer_block._kernel_indexes, within=pyo.Binary
    )
    layer_block._q_sum_maxpool = pyo.Constraint(layer.output_indexes)
    layer_block._zhat_upper_bound = pyo.Constraint(
        layer.output_indexes, layer_block._kernel_indexes
    )
    layer_block._zhat_lower_bound = pyo.Constraint(
        layer.output_indexes, layer_block._kernel_indexes
    )

    for output_index in layer.output_indexes:
        out_d, out_r, out_c = output_index

        # cannot compute an expr for the max,
        # as pyomo expressions cannot contain functions whose output depends on a comparison (except piecewise linear functions)
        # so compute lb and ub directly
        bounds = (
            input_layer_block.z[layer.input_index_mapper(input_index)].bounds
            for _, input_index in layer.kernel_index_with_input_indexes(
                out_d, out_r, out_c
            )
        )
        lbs, ubs = zip(*bounds)
        layer_block.zhat[output_index].setlb(max(lbs))
        layer_block.zhat[output_index].setub(max(ubs))

        layer_block._q_sum_maxpool[output_index] = 1 == sum(
            layer_block.q_maxpool[output_index, k] for k in layer_block._kernel_indexes
        )

        for l, input_index in layer.kernel_index_with_input_indexes(
            out_d, out_r, out_c
        ):
            input_index = layer.input_index_mapper(input_index)

            # Since biases are zero,
            # input_layer_block.z[input_index] is equal to w dot x in the formulation.
            layer_block._zhat_upper_bound[output_index, l] = layer_block.zhat[
                output_index
            ] <= input_layer_block.z[input_index] + sum(
                layer_block.q_maxpool[output_index, k]
                * _calculate_n_plus(output_index, l, k, layer, input_layer_block)
                for k in layer_block._kernel_indexes
            )
            layer_block._zhat_lower_bound[output_index, l] = (
                layer_block.zhat[output_index] >= input_layer_block.z[input_index]
            )


def _calculate_n_plus(out_index, l, k, layer, input_layer_block):
    if l == k:
        return 0
    x_l_index = layer.input_index_mapper(layer.get_input_index(out_index, l))
    x_k_index = layer.input_index_mapper(layer.get_input_index(out_index, k))
    return max(
        x_k_bound - x_l_bound
        for x_k_bound in input_layer_block.z[x_k_index].bounds
        for x_l_bound in input_layer_block.z[x_l_index].bounds
    )


def _input_layer_and_block(net_block, net, layer):
    input_layers = list(net.predecessors(layer))
    assert len(input_layers) == 1
    input_layer = input_layers[0]
    input_layer_block = net_block.layer[id(input_layer)]
    return input_layer, input_layer_block
