import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr


def default_partition_split_func(w, n):
    r"""Default function to partition weights in :math:`w` into :math:`n` partitions.

    Weights in :math:`w` are sorted and partitioned evenly.

    """
    sorted_indexes = np.argsort(w)
    n = min(n, len(sorted_indexes))
    return np.array_split(sorted_indexes, n)


def partition_based_dense_relu_layer(net_block, net, layer_block, layer, split_func):  # noqa: C901, PLR0915
    r"""Partition-based ReLU activation formulation.

    Generates the constraints for the ReLU activation function:

    .. math::

        \begin{align*}
            y_j = \max\left(0,\sum\limits_{i=0}^{F_{in}-1}w_{ij}x_i+b_j\right),
                && \forall 0\le j<F_{out}
        \end{align*}

    We additionally introduce the following notations to describe this formulation:

    .. math::

        \begin{align*}
            n       &:= \text{the number of partitions}\\
            S_k     &:=  \text{indexes of the $k$-th partition satisfying:} \\
                    & \quad\quad \bigcup\limits_{k=0}^{n-1} S_k=\{0,1,\dots,F_{in}-1\},
                    ~S_{k_1}\cap S_{k_2}=\emptyset, ~\forall k_1\neq k_2\\
            \sigma  &:= \text{if this activation function is activated, i.e.,}\\
                    & \quad\quad y_j=
                    \begin{cases}
                        0, & \sigma=1\\
                        \sum\limits_{i=0}^{F_{in}-1}w_{ij}x_i+b_j, & \sigma=0
                    \end{cases}\\
            p_k     &:=\text{auxiliary variable representing the $k$-th partition,
                    i.e., $\sum\limits_{i\in S_k}w_{ij}x_i$}\\
            l_k     &:=\text{the lower bound of $\sum\limits_{i\in S_k}w_{ij}x_i$}\\
            u_k     &:=\text{the upper bound of $\sum\limits_{i\in S_k}w_{ij}x_i$}
        \end{align*}


    The partition-based formulation for :math:`y_j` is given by:

    .. math::

        \begin{align*}
            & y_j=\sum\limits_{k=0}^{n-1}p_k+(1-\sigma)b_j\\
            & \sum\limits_{k=0}^{n-1}\left(\sum\limits_{i\in S_k}w_{ij}x_i-p_k\right)
                +\sigma b_j\le 0\\
            & \sum\limits_{k=0}^{n-1}p_k+(1-\sigma)b_j\ge 0\\
            & \sigma l_k\le \sum\limits_{i\in S_k}w_{ij}x_i-p_k
                \le \sigma u_k,~0\le k<n\\
            & (1-\sigma)l_k\le p_k\le (1-\sigma)u_k,~0\le k<n
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

    @layer_block.Block(layer.output_indexes)
    def output_node_block(b, *output_index):  # noqa: PLR0915
        # dense layers multiply only the last dimension of
        # their inputs
        weights = layer.weights[:, output_index[-1]]
        bias = layer.biases[output_index[-1]]

        splits = split_func(weights)
        num_splits = len(splits)

        b.sig = pyo.Var(domain=pyo.Binary)
        b.z2 = pyo.Var(range(num_splits))

        mapper = layer.input_index_mapper

        b.eq_16_lb = pyo.ConstraintList()
        b.eq_16_ub = pyo.ConstraintList()

        b.eq_17_lb = pyo.ConstraintList()
        b.eq_17_ub = pyo.ConstraintList()

        input_layer_indexes = list(layer.input_indexes_with_input_layer_indexes)

        # Add Equation 16 and 17
        for split_index in range(num_splits):
            expr = 0.0
            for split_local_index in splits[split_index]:
                _, local_index = input_layer_indexes[split_local_index]

                input_index = mapper(local_index) if mapper else local_index

                w = weights[local_index[-1]]
                expr += prev_layer_block.z[input_index] * w

            lb, ub = compute_bounds_on_expr(expr)
            if lb is None:
                msg = "Expression is unbounded below."
                raise ValueError(msg)
            if ub is None:
                msg = "Expression is unbounded above."
                raise ValueError(msg)

            z2 = b.z2[split_index]
            z2.setlb(min(0, lb))
            z2.setub(max(0, ub))

            b.eq_16_lb.add(expr - z2 >= b.sig * lb)
            b.eq_16_ub.add(expr - z2 <= b.sig * ub)
            b.eq_17_lb.add(z2 >= (1 - b.sig) * lb)
            b.eq_17_ub.add(z2 <= (1 - b.sig) * ub)

        # compute dense layer expression to compute bounds
        expr = 0.0
        for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
            w = layer.weights[local_index[-1], output_index[-1]]
            expr += prev_layer_block.z[input_index] * w
        # move this at the end to avoid numpy/pyomo var bug
        expr += bias

        lb, ub = compute_bounds_on_expr(expr)
        if lb is None:
            msg = "Expression is unbounded below."
            raise ValueError(msg)
        if ub is None:
            msg = "Expression is unbounded above."
            raise ValueError(msg)

        layer_block.z[output_index].setlb(0)
        layer_block.z[output_index].setub(max(0, ub))

        eq_13_expr = 0.0
        for split_index in range(num_splits):
            for split_local_index in splits[split_index]:
                _, local_index = input_layer_indexes[split_local_index]
                input_index = mapper(local_index) if mapper else local_index

                w = weights[local_index[-1]]
                eq_13_expr += prev_layer_block.z[input_index] * w
            eq_13_expr -= b.z2[split_index]
        eq_13_expr += bias * b.sig

        b.eq_13 = pyo.Constraint(expr=eq_13_expr <= 0)
        b.eq_14 = pyo.Constraint(
            expr=sum(b.z2[s] for s in range(num_splits)) + bias * (1 - b.sig) >= 0
        )
        b.eq_15 = pyo.Constraint(
            expr=layer_block.z[output_index]
            == sum(b.z2[s] for s in range(num_splits)) + bias * (1 - b.sig)
        )
