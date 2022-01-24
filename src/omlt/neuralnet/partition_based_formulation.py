from multiprocessing.sharedctypes import Value
import pyomo.environ as pyo
import numpy as np
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs
from omlt.neuralnet.layer import ConvLayer, DenseLayer, InputLayer
from omlt.neuralnet.nn_formulation import _DEFAULT_ACTIVATION_CONSTRAINTS, _DEFAULT_LAYER_CONSTRAINTS


class PartitionBasedNeuralNetworkFormulation(_PyomoFormulation):
    """
    This class is used to build partition-based formulations
    of neural networks.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    split_func : callable
        the function used to compute the splits
    """
    def __init__(self, network_structure, split_func=None):
        super().__init__()

        self.__network_definition = network_structure
        self.__scaling_object = network_structure.scaling_object
        self.__scaled_input_bounds = network_structure.scaled_input_bounds

        if split_func is None:
            split_func = lambda w: _default_split_func(w, 2)

        self.__split_func = split_func

    def _build_formulation(self):
        _setup_scaled_inputs_outputs(self.block,
                                     self.__scaling_object,
                                     self.__scaled_input_bounds)
        
        _build_neural_network_formulation(
            block=self.block,
            network_structure=self.__network_definition,
            split_func=self.__split_func,
        )

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Unsupported multiple network input variables'
        return network_inputs[0].input_indexes

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Unsupported multiple network output variables'
        return network_outputs[0].output_indexes


def _build_neural_network_formulation(block, network_structure, split_func):
    """
    Adds the partition-based neural network formulation to the given Pyomo block.

    Parameters
    ----------
    block : Block
        the Pyomo block
    network_structure : NetworkDefinition
        the neural network definition
    split_func : callable
        the function used to compute the splits
    """

    layer_constraints = _DEFAULT_LAYER_CONSTRAINTS
    activation_constraints = _DEFAULT_ACTIVATION_CONSTRAINTS
    net = network_structure
    layers = list(net.layers)

    block.layers = pyo.Set(initialize=[id(layer) for layer in layers], ordered=True)

    # create the z and z_hat variables for each of the layers
    @block.Block(block.layers)
    def layer(b, layer_id):
        net_layer = net.layer(layer_id)
        b.z = pyo.Var(net_layer.output_indexes, initialize=0)
        if isinstance(net_layer, InputLayer):
            for index in net_layer.output_indexes:
                input_var = block.scaled_inputs[index]
                z_var = b.z[index]
                z_var.setlb(input_var.lb)
                z_var.setub(input_var.ub)
        else:
            # add zhat only to non input layers
            b.zhat = pyo.Var(net_layer.output_indexes, initialize=0)

        return b

    for layer in layers:
        if isinstance(layer, InputLayer):
            continue
        layer_id = id(layer)
        layer_block = block.layer[layer_id]
        if isinstance(layer, DenseLayer) and layer.activation == "relu":
            _partition_based_dense_relu_layer(block, net, layer_block, layer, split_func)
        else:
            constraint_func = layer_constraints[type(layer)]
            activation_func = activation_constraints[layer.activation]
            constraint_func(block, net, layer_block, layer)
            activation_func(block, net, layer_block, layer)

    # setup input variables constraints
    # currently only support a single input layer
    input_layers = list(net.input_layers)
    assert len(input_layers) == 1
    input_layer = input_layers[0]

    @block.Constraint(input_layer.output_indexes)
    def input_assignment(b, *output_index):
        return b.scaled_inputs[output_index] == b.layer[id(input_layer)].z[output_index]

    # setup output variables constraints
    # currently only support a single output layer
    output_layers = list(net.output_layers)
    assert len(output_layers) == 1
    output_layer = output_layers[0]

    @block.Constraint(output_layer.output_indexes)
    def output_assignment(b, *output_index):
        return b.scaled_outputs[output_index] == b.layer[id(output_layer)].z[output_index]


def _default_split_func(w, n):
    sorted_indexes = np.argsort(w)
    n = min(n, len(sorted_indexes))
    return np.array_split(sorted_indexes, n)


def _partition_based_dense_relu_layer(net_block, net, layer_block, layer, split_func):
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
