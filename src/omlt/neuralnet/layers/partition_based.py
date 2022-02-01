import pyomo.environ as pyo
import numpy as np
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr


def default_partition_split_func(w, n):
    sorted_indexes = np.argsort(w)
    n = min(n, len(sorted_indexes))
    return np.array_split(sorted_indexes, n)


def partition_based_dense_relu_layer(net_block, net, layer_block, layer, split_func):
    # not an input layer, process the expressions
    prev_layers = list(net.predecessors(layer))
    assert len(prev_layers) == 1
    prev_layer = prev_layers[0]
    prev_layer_block = net_block.layer[id(prev_layer)]

    @layer_block.Block(layer.output_indexes)
    def output_node_block(b, *output_index):
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

                if mapper:
                    input_index = mapper(local_index)
                else:
                    input_index = local_index

                w = weights[local_index[-1]]
                expr += prev_layer_block.z[input_index] * w

            lb, ub = compute_bounds_on_expr(expr)
            assert lb is not None and ub is not None

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
        assert lb is not None and ub is not None

        layer_block.z[output_index].setlb(0)
        layer_block.z[output_index].setub(max(0, ub))

        eq_13_expr = 0.0
        for split_index in range(num_splits):
            for split_local_index in splits[split_index]:
                _, local_index = input_layer_indexes[split_local_index]
                if mapper:
                    input_index = mapper(local_index)
                else:
                    input_index = local_index

                w = weights[local_index[-1]]
                eq_13_expr += prev_layer_block.z[input_index] * w
            eq_13_expr -= b.z2[split_index]
        eq_13_expr += bias * b.sig

        b.eq_13 = pyo.Constraint(expr=eq_13_expr <= 0)
        b.eq_14 = pyo.Constraint(expr=sum(b.z2[s] for s in range(num_splits)) + bias * (1 - b.sig) >= 0)
        b.eq_15 = pyo.Constraint(expr=layer_block.z[output_index] == sum(b.z2[s] for s in range(num_splits)) + bias * (1 - b.sig))