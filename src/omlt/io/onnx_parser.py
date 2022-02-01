from onnx import numpy_helper
import numpy as np

from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import (
    ConvLayer,
    DenseLayer,
    InputLayer,
    IndexMapper,
)


_ACTIVATION_OP_TYPES = ["Relu", "Sigmoid", "LogSoftmax"]


class NetworkParser:
    """
    References
    ----------
    * https://github.com/onnx/onnx/blob/master/docs/Operators.md
    """
    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self._graph = None
        self._initializers = None
        self._constants = None
        self._nodes = None
        self._nodes_by_output = None
        self._inputs = None
        self._outputs = None
        self._node_stack = None
        self._node_map = None

    def parse_network(self, graph, scaling_object, input_bounds):
        self._reset_state()
        self._graph = graph

        # initializers contain constant data
        initializers = dict()
        for initializer in self._graph.initializer:
            initializers[initializer.name] = numpy_helper.to_array(initializer)

        self._initializers = initializers

        # Build graph
        nodes = dict()
        nodes_by_output = dict()
        inputs = set()
        outputs = set()
        self._node_map = dict()

        network = NetworkDefinition(scaling_object=scaling_object, scaled_input_bounds=input_bounds)

        network_input = None
        for input in self._graph.input:
            nodes[input.name] = ("input", input.type, [])
            nodes_by_output[input.name] = input.name
            inputs.add(input.name)
            # onnx inputs are tensors. Flatten tensors to a vector.
            dim_value = None
            size = []
            for dim in input.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    if dim_value is None:
                        dim_value = 1
                    size.append(dim.dim_value)
                    dim_value *= dim.dim_value
            assert dim_value is not None
            assert network_input is None
            network_input = InputLayer(size)
            self._node_map[input.name] = network_input
            network.add_layer(network_input)

        assert network_input is not None

        self._nodes = nodes
        self._nodes_by_output = nodes_by_output
        self._inputs = inputs
        self._outputs = outputs

        # The node.output field contains the name of this node
        # output.
        # Here map output name to node name.
        for node in self._graph.node:
            for output in node.output:
                nodes_by_output[output] = node.name

        self._constants = dict()
        for node in self._graph.node:
            # add node not connected to anything
            self._nodes[node.name] = ("node", node, [])

            # Map inputs by their output name
            node_inputs = [
                nodes_by_output[input]
                for input in node.input
                if input not in initializers
            ]

            if node_inputs:
                # Now connect inputs to the current node
                for input in node_inputs:
                    self._nodes[input][2].append(node.name)
            else:
                assert node.op_type == "Constant"
                for output in node.output:
                    value = _parse_constant_value(node)
                    self._constants[output] = value

        # traverse graph
        self._node_stack = list(inputs)

        self._weights = dict()
        self._biases = dict()
        self._activations = dict()

        while self._node_stack:
            node_name = self._node_stack.pop()
            type_, node, next_nodes = self._nodes[node_name]

            # no need to process inputs or outputs
            if type_ == "node":
                new_layer, new_layer_inputs = self._visit_node(
                    node, next_nodes
                )
                if new_layer is not None:
                    network.add_layer(new_layer)
                    for layer_input in new_layer_inputs:
                        network.add_edge(layer_input, new_layer)
            else:
                for next in next_nodes:
                    self._node_stack.append(next)

        return network

    def _visit_node(self, node, next_nodes):
        if node.op_type == "MatMul":
            next_nodes, new_layer, new_layer_inputs = self._consume_dense_nodes(node, next_nodes)
        elif node.op_type == "Gemm":
            next_nodes, new_layer, new_layer_inputs = self._consume_gemm_dense_nodes(node, next_nodes)
        elif node.op_type == "Conv":
            next_nodes, new_layer, new_layer_inputs = self._consume_conv_nodes(node, next_nodes)
        elif node.op_type == "Reshape":
            next_nodes = self._consume_reshape_nodes(node, next_nodes)
            new_layer = new_layer_inputs = None
        else:
            raise Exception(f"Unhandled node type {node.op_type}")

        for next in next_nodes:
            self._node_stack.append(next)

        return new_layer, new_layer_inputs

    def _consume_dense_nodes(self, node, next_nodes):
        """Starting from a MatMul node, consume nodes to form a dense Ax + b node."""
        assert node.op_type == "MatMul"
        assert len(node.input) == 2

        [in_0, in_1] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        node_weights = self._initializers[in_1]

        assert len(next_nodes) == 1

        # expect 'Add' node ahead
        type_, node, maybe_next_nodes = self._nodes[next_nodes[0]]
        assert type_ == "node"
        assert node.op_type == "Add"

        # extract biases
        next_nodes = maybe_next_nodes
        assert len(node.input) == 2
        [in_0, in_1] = list(node.input)

        if in_0 in self._initializers:
            node_biases = self._initializers[in_0]
        else:
            assert in_1 in self._initializers
            node_biases = self._initializers[in_1]

        assert len(node_weights.shape) == 2
        assert node_weights.shape[1] == node_biases.shape[0]
        assert len(node.output) == 1

        input_output_size = input_layer.output_size
        if transformer is not None:
            input_output_size = transformer.output_size

        output_size = input_output_size[:-1] + [node_weights.shape[1]]

        activation = "linear"
        if len(next_nodes) == 1:
            # check if Relu
            type_, maybe_node, maybe_next_nodes = self._nodes[next_nodes[0]]
            if maybe_node.op_type in _ACTIVATION_OP_TYPES:
                node = maybe_node
                activation = maybe_node.op_type.lower()
                next_nodes = maybe_next_nodes

        dense_layer = DenseLayer(
            input_output_size,
            output_size,
            node_weights,
            node_biases,
            activation=activation,
            input_index_mapper=None
        )
        self._node_map[node.name] = dense_layer
        self._node_map[node.output[0]] = dense_layer

        return next_nodes, dense_layer, [input_layer]

    def _consume_gemm_dense_nodes(self, node, next_nodes):
        """Starting from a Gemm node, consume nodes to form a dense aAB + bC node."""
        assert node.op_type == "Gemm"
        assert len(node.input) == 3

        attr = _collect_attributes(node)
        alpha = attr["alpha"]
        beta = attr["beta"]
        assert attr["transB"] == 1
        [in_0, in_1, in_2] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        weights = self._initializers[in_1]
        # transpose B
        weights = np.transpose(weights)
        biases = self._initializers[in_2]

        input_output_size = input_layer.output_size
        if transformer is not None:
            input_output_size = transformer.output_size

        # output is the same size as input except for the last dimension
        output_size = input_output_size[:-1] + [weights.shape[1]]

        activation = "linear"
        if len(next_nodes) == 1:
            # check if Relu
            type_, maybe_node, maybe_next_nodes = self._nodes[next_nodes[0]]
            if maybe_node.op_type in _ACTIVATION_OP_TYPES:
                node = maybe_node
                activation = node.op_type.lower()
                next_nodes = maybe_next_nodes

        weights = weights * alpha
        biases = beta * biases

        dense_layer = DenseLayer(
            input_output_size,
            output_size, 
            weights,
            biases,
            activation=activation,
            input_index_mapper=transformer
        )
        self._node_map[node.name] = dense_layer
        self._node_map[node.output[0]] = dense_layer

        return next_nodes, dense_layer, [input_layer]

    def _consume_conv_nodes(self, node, next_nodes):
        """
        Starting from a Conv node, consume nodes to form a convolution node with
        (optional) activation function.
        """
        assert node.op_type == "Conv"
        assert len(node.input) in [2, 3]
        if len(node.input) == 2:
            [in_0, in_1] = list(node.input)
            in_2 = None
        else:
            [in_0, in_1, in_2] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        input_output_size = input_layer.output_size
        if transformer is not None:
            input_output_size = transformer.output_size
        weights = self._initializers[in_1]
        [out_channels, in_channels, *kernel_shape] = weights.shape

        if in_2 is None:
            biases = np.zeros(out_channels)
        else:
            biases = self._initializers[in_2]

        attr = _collect_attributes(node)

        strides = attr['strides']

        # check only kernel shape and stride are set
        # everything else is not supported
        assert biases.shape == (out_channels,)
        assert in_channels == input_output_size[0]
        assert attr['kernel_shape'] == kernel_shape
        assert attr['dilations'] == [1, 1]
        assert attr['group'] == 1
        if 'pads' in attr:
            assert not np.any(attr['pads'])  # pads all zero
        assert len(kernel_shape) == len(strides)
        assert len(input_output_size) == len(kernel_shape) + 1

        # generate new nodes for the node output
        padding = 0
        output_size = [out_channels]
        for w, k, s in zip(input_output_size[1:], kernel_shape, strides):
            new_w = int((w - k + 2*padding) / s) + 1
            output_size.append(new_w)

        activation = "linear"
        if len(next_nodes) == 1:
            # check if Relu
            type_, maybe_node, maybe_next_nodes = self._nodes[next_nodes[0]]
            if maybe_node.op_type in _ACTIVATION_OP_TYPES:
                node = maybe_node
                activation = maybe_node.op_type.lower()
                next_nodes = maybe_next_nodes

        # convolute image one channel at the time
        # expect 2d image with channels
        assert len(input_output_size) == 3

        conv_layer = ConvLayer(
            input_output_size,
            output_size,
            strides,
            weights,
            activation=activation,
            input_index_mapper=transformer,
        )
        self._node_map[node.name] = conv_layer
        self._node_map[node.output[0]] = conv_layer

        return next_nodes, conv_layer, [input_layer]

    def _consume_reshape_nodes(self, node, next_nodes):
        """Parse a Reshape node."""
        assert node.op_type == "Reshape"
        assert len(node.input) == 2
        [in_0, in_1] = list(node.input)
        input_layer = self._node_map[in_0]
        new_shape = self._constants[in_1]
        output_size = np.empty(input_layer.output_size).reshape(new_shape).shape
        transformer = IndexMapper(input_layer.output_size, list(output_size))
        self._node_map[node.output[0]] = (transformer, input_layer)
        return next_nodes

    def _node_input_and_transformer(self, node_name):
        maybe_layer = self._node_map[node_name]
        if isinstance(maybe_layer, tuple):
            transformer, input_layer = maybe_layer
            return input_layer, transformer
        else:
            return maybe_layer, None


def _collect_attributes(node):
    r = dict()
    for attr in node.attribute:
        if attr.type == 1:  # FLOAT
            r[attr.name] = attr.f
        elif attr.type == 2:  # INT
            r[attr.name] = int(attr.i)
        elif attr.type == 4:  # TENSOR
            r[attr.name] = numpy_helper.to_array(attr.t)
            pass
        elif attr.type == 7: # INTS
            r[attr.name] = list(attr.ints)
        else:
            raise RuntimeError(f'unhandled attribute type {attr.type}')
    return r


def _parse_constant_value(node):
    attr = _collect_attributes(node)
    value = attr['value']
    return value
