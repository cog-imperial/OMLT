import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from omlt.base import OmltConstraintFactory, OmltVarFactory
from omlt.block import OmltBlockCore


def default_partition_split_func(w, n):
    r"""Default function to partition weights in :math:`w` into :math:`n` partitions.

    Weights in :math:`w` are sorted and partitioned evenly.

    """
    sorted_indexes = np.argsort(w)
    n = min(n, len(sorted_indexes))
    return np.array_split(sorted_indexes, n)


def partition_based_dense_relu_layer(net_block, net, layer_block, layer, split_func):  # noqa: C901,PLR0912,PLR0915
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

    if layer_block._format == "pyomo":

        @layer_block.Block(layer.output_indexes)
        def output_node_block(b, *output_index):  # noqa: PLR0915
            # dense layers multiply only the last dimension of
            # their inputs
            weights = layer.weights[:, output_index[-1]]
            bias = layer.biases[output_index[-1]]

            splits = split_func(weights)
            num_splits = len(splits)

            var_factory = OmltVarFactory()
            b.sig = var_factory.new_var(binary=True, lang=net_block._format)
            minus_sig = 1 - b.sig  # type: ignore[operator]
            b.z2 = var_factory.new_var(range(num_splits), lang=net_block._format)

            mapper = layer.input_index_mapper
            constraint_factory = OmltConstraintFactory()
            b.eq_16_lb = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )
            b.eq_16_ub = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )

            b.eq_17_lb = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )
            b.eq_17_ub = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )

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

                b.eq_16_lb[split_index] = b.sig * lb <= expr - z2
                b.eq_16_ub[split_index] = b.sig * ub >= expr - z2

                b.eq_17_lb[split_index] = minus_sig * lb <= z2
                b.eq_17_ub[split_index] = minus_sig * ub >= z2

            # compute dense layer expression to compute bounds
            expr = 0.0
            for (
                local_index,
                input_index,
            ) in layer.input_indexes_with_input_layer_indexes:
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

            b.eq_13 = constraint_factory.new_constraint(
                expr=eq_13_expr <= 0, lang=net_block._format
            )
            b.eq_14 = constraint_factory.new_constraint(
                expr=sum(b.z2[s] for s in range(num_splits))
                + bias * minus_sig._expression
                >= 0,
                lang=net_block._format,
            )
            b.eq_15 = constraint_factory.new_constraint(
                expr=layer_block.z[output_index]
                == sum(b.z2[s] for s in range(num_splits))
                + bias * minus_sig._expression,
                lang=net_block._format,
            )
    else:
        layer_block.output_node_block = {
            output_index: OmltBlockCore() for output_index in layer.output_indexes
        }

        for output_index in layer.output_indexes:
            # dense layers multiply only the last dimension of
            # their inputs
            weights = layer.weights[:, output_index[-1]]
            bias = layer.biases[output_index[-1]]

            splits = split_func(weights)
            num_splits = len(splits)

            var_factory = OmltVarFactory()
            layer_block.output_node_block[output_index].sig = var_factory.new_var(
                domain=pyo.Binary, lang=net_block._format
            )
            layer_block.output_node_block[output_index].z2 = var_factory.new_var(
                range(num_splits), lang=net_block._format
            )

            mapper = layer.input_index_mapper
            constraint_factory = OmltConstraintFactory()
            layer_block.output_node_block[
                output_index
            ].eq_16_lb = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )
            layer_block.output_node_block[
                output_index
            ].eq_16_ub = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )

            layer_block.output_node_block[
                output_index
            ].eq_17_lb = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )
            layer_block.output_node_block[
                output_index
            ].eq_17_ub = constraint_factory.new_constraint(
                range(num_splits), lang=net_block._format
            )

            input_layer_indexes = list(layer.input_indexes_with_input_layer_indexes)

            # Add Equation 16 and 17
            for split_index in range(num_splits):
                expr = 0.0
                lb = 0.0
                ub = 0.0
                for split_local_index in splits[split_index]:
                    _, local_index = input_layer_indexes[split_local_index]

                    input_index = mapper(local_index) if mapper else local_index

                    w = weights[local_index[-1]]
                    expr += prev_layer_block.z[input_index] * w
                    if w >= 0:
                        lb += prev_layer_block.z[input_index].lb * w
                        ub += prev_layer_block.z[input_index].ub * w
                    else:
                        lb += prev_layer_block.z[input_index].ub * w
                        ub += prev_layer_block.z[input_index].lb * w

                z2 = layer_block.output_node_block[output_index].z2[split_index]
                z2.setlb(min(0, lb))
                z2.setub(max(0, ub))

                layer_block.output_node_block[output_index].eq_16_lb[split_index] = (
                    layer_block.output_node_block[output_index].sig * lb <= expr - z2
                )
                layer_block.output_node_block[output_index].eq_16_ub[split_index] = (
                    layer_block.output_node_block[output_index].sig * ub >= expr - z2
                )

                minus_sig = 1 - layer_block.output_node_block[output_index].sig
                layer_block.output_node_block[output_index].eq_17_lb[split_index] = (
                    minus_sig * lb <= z2
                )
                layer_block.output_node_block[output_index].eq_17_ub[split_index] = (
                    minus_sig * ub >= z2
                )

            # compute dense layer expression to compute bounds
            expr = 0.0
            lb = 0.0
            ub = 0.0
            for (
                local_index,
                input_index,
            ) in layer.input_indexes_with_input_layer_indexes:
                w = layer.weights[local_index[-1], output_index[-1]]
                expr += prev_layer_block.z[input_index] * w
                if w >= 0:
                    lb += prev_layer_block.z[input_index].lb * w
                    ub += prev_layer_block.z[input_index].ub * w
                else:
                    lb += prev_layer_block.z[input_index].ub * w
                    ub += prev_layer_block.z[input_index].lb * w

            # move this at the end to avoid numpy/pyomo var bug
            expr += bias
            lb += bias
            ub += bias

            layer_block.z[output_index].setlb(0)
            layer_block.z[output_index].setub(max(0, ub))
            eq_13_expr = 0.0
            for split_index in range(num_splits):
                for split_local_index in splits[split_index]:
                    _, local_index = input_layer_indexes[split_local_index]
                    input_index = mapper(local_index) if mapper else local_index

                    w = weights[local_index[-1]]
                    eq_13_expr += prev_layer_block.z[input_index] * w
                eq_13_expr -= layer_block.output_node_block[output_index].z2[
                    split_index
                ]
            eq_13_expr += bias * layer_block.output_node_block[output_index].sig
            layer_block.output_node_block[
                output_index
            ].eq_13 = constraint_factory.new_constraint(
                lhs=eq_13_expr, sense="<=", rhs=0, lang=net_block._format
            )
            layer_block.output_node_block[
                output_index
            ].eq_14 = constraint_factory.new_constraint(
                lhs=(
                    sum(
                        layer_block.output_node_block[output_index].z2[s]
                        for s in range(num_splits)
                    )
                    + bias * (1 - layer_block.output_node_block[output_index].sig)
                ),
                sense=">=",
                rhs=0,
                lang=net_block._format,
            )
            layer_block.output_node_block[
                output_index
            ].eq_15 = constraint_factory.new_constraint(
                lhs=layer_block.z[output_index],
                sense="==",
                rhs=(
                    sum(
                        layer_block.output_node_block[output_index].z2[s]
                        for s in range(num_splits)
                    )
                    + bias * (1 - layer_block.output_node_block[output_index].sig)
                ),
                lang=net_block._format,
            )
