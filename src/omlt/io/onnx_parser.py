import math
from typing import TYPE_CHECKING, Any

import numpy as np
from onnx import numpy_helper

from omlt.neuralnet.layer import (
    ConvLayer2D,
    DenseLayer,
    IndexMapper,
    InputLayer,
    PoolingLayer2D,
)
from omlt.neuralnet.network_definition import NetworkDefinition

if TYPE_CHECKING:
    from collections.abc import Callable

_ACTIVATION_OP_TYPES = ["Relu", "Sigmoid", "LogSoftmax", "Tanh", "Softplus"]
_POOLING_OP_TYPES = ["MaxPool"]
DENSE_INPUT_DIMENSIONS = 2
GEMM_INPUT_DIMENSIONS = 3
CONV_INPUT_DIMENSIONS = [2, 3]
TWO_D_IMAGE_W_CHANNELS = 3
RESHAPE_INPUT_DIMENSIONS = 2
MAXPOOL_INPUT_DIMENSIONS = 1
MAXPOOL_INPUT_OUTPUT_W_BATCHES = 4
# Attribute types enum:
ATTR_FLOAT = 1
ATTR_INT = 2
ATTR_TENSOR = 4
ATTR_INTS = 7


class NetworkParser:
    """Network Parser.

    References:
        * https://github.com/onnx/onnx/blob/master/docs/Operators.md
    """

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self._graph = None
        self._initializers = {}
        self._constants = {}
        self._nodes = {}
        self._nodes_by_output = None
        self._inputs = None
        self._outputs = None
        self._node_stack = []
        self._node_map = {}

    def parse_network(self, graph, scaling_object, input_bounds):  # noqa: C901, PLR0912, PLR0915
        self._reset_state()
        self._graph = graph

        # initializers contain constant data
        initializers: dict[str, Any] = {}
        for initializer in self._graph.initializer:
            initializers[initializer.name] = numpy_helper.to_array(initializer)

        self._initializers = initializers

        # Build graph
        nodes: dict[str, tuple[str, Any, list[Any]]] = {}
        nodes_by_output = {}
        inputs = set()
        outputs: set[Any] = set()
        self._node_map = {}

        network = NetworkDefinition(
            scaling_object=scaling_object, scaled_input_bounds=input_bounds
        )

        network_input = None
        for input_node in self._graph.input:
            nodes[input_node.name] = ("input", input_node.type, [])
            nodes_by_output[input_node.name] = input_node.name
            inputs.add(input_node.name)
            # onnx inputs are tensors. Flatten tensors to a vector.
            dim_value = None
            size = []
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    if dim_value is None:
                        dim_value = 1
                    size.append(dim.dim_value)
                    dim_value *= dim.dim_value
            if dim_value is None:
                msg = (
                    f'All dimensions in graph "{graph.name}" input tensor have 0 value.'
                )
                raise ValueError(msg)
            network_input = InputLayer(size)
            self._node_map[input_node.name] = network_input
            network.add_layer(network_input)

        if network_input is None:
            msg = f'No valid input layer found in graph "{graph.name}".'
            raise ValueError(msg)

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

        self._constants = {}
        for node in self._graph.node:
            # add node not connected to anything
            self._nodes[node.name] = ("node", node, [])

            # Map inputs by their output name
            node_inputs = [
                nodes_by_output[input_node]
                for input_node in node.input
                if input_node not in initializers
            ]

            if node_inputs:
                # Now connect inputs to the current node
                for input_node in node_inputs:
                    self._nodes[input_node][2].append(node.name)
            elif node.op_type == "Constant":
                for output in node.output:
                    value = _parse_constant_value(node)
                    self._constants[output] = value
            else:
                msg = (
                    'Nodes must have inputs or have op_type "Constant". Node'
                    f' "{node.name}" has no inputs and op_type "{node.op_type}".'
                )
                raise ValueError(msg)

        # traverse graph
        self._node_stack = list(inputs)

        self._weights = {}
        self._biases = {}
        self._activations = {}

        while self._node_stack:
            node_name = self._node_stack.pop()
            type_, node, next_nodes = self._nodes[node_name]

            # no need to process inputs or outputs
            if type_ == "node":
                new_layer, new_layer_inputs = self._visit_node(node, next_nodes)
                if new_layer is not None:
                    network.add_layer(new_layer)
                    for layer_input in new_layer_inputs:
                        network.add_edge(layer_input, new_layer)
            else:
                for next_node in next_nodes:
                    self._node_stack.append(next_node)

        return network

    def _visit_node(self, node, next_nodes):
        if node.op_type == "MatMul":
            next_nodes, new_layer, new_layer_inputs = self._consume_dense_nodes(
                node, next_nodes
            )
        elif node.op_type == "Gemm":
            next_nodes, new_layer, new_layer_inputs = self._consume_gemm_dense_nodes(
                node, next_nodes
            )
        elif node.op_type == "Conv":
            next_nodes, new_layer, new_layer_inputs = self._consume_conv_nodes(
                node, next_nodes
            )
        elif node.op_type == "Reshape":
            next_nodes = self._consume_reshape_nodes(node, next_nodes)
            new_layer = new_layer_inputs = None
        elif node.op_type in _POOLING_OP_TYPES:
            next_nodes, new_layer, new_layer_inputs = self._consume_pool_nodes(
                node, next_nodes
            )
        else:
            msg = f"Unhandled node type {node.op_type}"
            raise ValueError(msg)

        for next_node in next_nodes:
            self._node_stack.append(next_node)

        return new_layer, new_layer_inputs

    def _consume_dense_nodes(  # noqa: C901, PLR0912
        self, node: Any, next_nodes: Any
    ) -> tuple[Any, Any, list[Any]]:
        """Starting from a MatMul node, consume nodes to form a dense Ax + b node."""
        if node.op_type != "MatMul":
            msg = (
                f"{node.name} is a {node.op_type} node, but the parsing method for"
                " MatMul nodes was called. This could indicate changes in the"
                " network being parsed."
            )
            raise ValueError(msg)

        if len(node.input) != DENSE_INPUT_DIMENSIONS:
            msg = (
                f"{node.name} input has {len(node.input)} dimensions, only nodes with 2"
                " input dimensions can be used as starting points for parsing."
            )
            raise ValueError(msg)

        [in_0, in_1] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        node_weights = self._initializers[in_1]

        if len(next_nodes) != 1:
            msg = (
                f"Next nodes must have length 1, {next_nodes} has length"
                f" {len(next_nodes)}"
            )
            raise ValueError(msg)

        # expect 'Add' node ahead
        type_, node, maybe_next_nodes = self._nodes[next_nodes[0]]
        if type_ != "node":
            msg = f"Expected a node next, got a {type_} instead."
            raise TypeError(msg)
        if node.op_type != "Add":
            msg = (
                f"The first node to be consumed, {node.name}, is a {node.op_type} node."
                " Only Add nodes are supported."
            )
            raise ValueError(msg)

        # extract biases
        next_nodes = maybe_next_nodes
        [in_0, in_1] = list(node.input)

        if in_0 in self._initializers:
            node_biases = self._initializers[in_0]
        elif in_1 in self._initializers:
            node_biases = self._initializers[in_1]
        else:
            msg = "Node inputs were not found in graph initializers."
            raise ValueError(msg)
        if len(node_weights.shape) != DENSE_INPUT_DIMENSIONS:
            msg = "Node weights must be a 2-dimensional matrix."
            raise ValueError(msg)
        if node_weights.shape[1] != node_biases.shape[0]:
            msg = (
                f"Node weights has {node_weights.shape[1]} columns; node biases has "
                f"{node_biases.shape[0]} rows. These must be equal."
            )
            raise ValueError(msg)
        if len(node.output) != 1:
            msg = f"Node output is {node.output} but should be a single value."
            raise ValueError(msg)

        input_output_size = _get_input_output_size(input_layer, transformer)

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
            input_index_mapper=None,
        )
        self._node_map[node.name] = dense_layer
        self._node_map[node.output[0]] = dense_layer

        return next_nodes, dense_layer, [input_layer]

    def _consume_gemm_dense_nodes(self, node, next_nodes):
        """Starting from a Gemm node, consume nodes to form a dense aAB + bC node."""
        if node.op_type != "Gemm":
            msg = (
                f"{node.name} is a {node.op_type} node, but the parsing method for"
                " Gemm nodes was called. This could indicate changes in the"
                " network being parsed."
            )
            raise ValueError(msg)
        if len(node.input) != GEMM_INPUT_DIMENSIONS:
            msg = (
                f"{node.name} input has {len(node.input)} dimensions, only nodes with"
                " 3 input dimensions can be used as starting points for parsing."
            )
            raise ValueError(msg)

        attr = _collect_attributes(node)
        alpha = attr["alpha"]
        beta = attr["beta"]
        [in_0, in_1, in_2] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        weights = self._initializers[in_1]
        # transpose B
        if attr["transB"] == 1:
            weights = np.transpose(weights)
        biases = self._initializers[in_2]

        input_output_size = _get_input_output_size(input_layer, transformer)

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
            input_index_mapper=transformer,
        )
        self._node_map[node.name] = dense_layer
        self._node_map[node.output[0]] = dense_layer

        return next_nodes, dense_layer, [input_layer]

    def _consume_conv_nodes(self, node, next_nodes):  # noqa: PLR0912, C901, PLR0915
        """Consume Conv nodes.

        Starting from a Conv node, consume nodes to form a convolution node with
        (optional) activation function.
        """
        if node.op_type != "Conv":
            msg = (
                f"{node.name} is a {node.op_type} node, but the parsing method for"
                " Conv nodes was called. This could indicate changes in the"
                " network being parsed."
            )
            raise ValueError(msg)
        if len(node.input) not in CONV_INPUT_DIMENSIONS:
            msg = (
                f"{node.name} input has {len(node.input)} dimensions, only nodes with"
                " 2 or 3 input dimensions can be used as starting points for parsing."
            )
            raise ValueError(msg)

        if len(node.input) == CONV_INPUT_DIMENSIONS[0]:
            [in_0, in_1] = list(node.input)
            in_2 = None
        else:
            [in_0, in_1, in_2] = list(node.input)
        input_layer, transformer = self._node_input_and_transformer(in_0)
        input_output_size = _get_input_output_size(input_layer, transformer)
        weights = self._initializers[in_1]
        [out_channels, in_channels, *kernel_shape] = weights.shape

        biases = np.zeros(out_channels) if in_2 is None else self._initializers[in_2]

        attr = _collect_attributes(node)

        strides = attr["strides"]
        # check only kernel shape and stride are set
        if attr["kernel_shape"] != kernel_shape:
            msg = (
                f"Kernel shape attribute {attr['kernel_shape']} does not match"
                f" initialized kernel shape {kernel_shape}."
            )
            raise ValueError(msg)
        if len(kernel_shape) != len(strides):
            msg = (
                f"Initialized kernel shape {kernel_shape} has {len(kernel_shape)} "
                f"dimensions. Strides attribute has {len(strides)} dimensions. "
                "These must be equal."
            )
            raise ValueError(msg)
        if len(input_output_size) != len(kernel_shape) + 1:
            msg = (
                f"Input/output size ({input_output_size}) must have one more dimension "
                f"than initialized kernel shape ({kernel_shape})."
            )
            raise ValueError(msg)

        # Check input, output have correct dimensions
        if biases.shape != (out_channels,):
            msg = (
                f"Biases shape {biases.shape} must match output weights channels"
                f" {(out_channels,)}."
            )
            raise ValueError(msg)
        if in_channels != input_output_size[0]:
            msg = (
                f"Input/output size ({input_output_size}) first dimension must match "
                f"input weights channels ({in_channels})."
            )
            raise ValueError(msg)

        # Other attributes are not supported
        if "dilations" in attr and attr["dilations"] != [1, 1]:
            msg = (
                f"{node} has non-identity dilations ({attr['dilations']}). This is not"
                " supported."
            )
            raise ValueError(msg)
        if attr["group"] != 1:
            msg = f"{node} has multiple groups ({attr['group']}). This is unsupported."
            raise ValueError(msg)
        if "pads" in attr and np.any(attr["pads"]):
            msg = f"{node} has non-zero pads ({attr['pads']}). This is not supported."
            raise ValueError(msg)

        # generate new nodes for the node output
        padding = 0
        output_size = [out_channels]
        for w, k, s in zip(input_output_size[1:], kernel_shape, strides):
            new_w = int((w - k + 2 * padding) / s) + 1
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
        if len(input_output_size) != TWO_D_IMAGE_W_CHANNELS:
            msg = f"Expected a 2D image with channels, got {input_output_size}."
            raise ValueError(msg)

        conv_layer = ConvLayer2D(
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
        if node.op_type != "Reshape":
            msg = (
                f"{node.name} is a {node.op_type} node, but the parsing method for"
                " Reshape nodes was called. This could indicate changes in the"
                " network being parsed."
            )
            raise ValueError(msg)
        if len(node.input) != RESHAPE_INPUT_DIMENSIONS:
            msg = (
                f"{node.name} input has {len(node.input)} dimensions, only nodes with"
                " 2 input dimensions can be used as starting points for parsing."
            )
            raise ValueError(msg)
        [in_0, in_1] = list(node.input)
        input_layer = self._node_map[in_0]
        new_shape = self._constants[in_1]
        output_size = np.empty(input_layer.output_size).reshape(new_shape).shape
        transformer = IndexMapper(input_layer.output_size, list(output_size))
        self._node_map[node.output[0]] = (transformer, input_layer)
        return next_nodes

    def _consume_pool_nodes(self, node, next_nodes):  # noqa: PLR0912, C901, PLR0915
        """Consume MaxPool nodes.

        Starting from a MaxPool node, consume nodes to form a pooling node with
        (optional) activation function.
        """
        if node.op_type not in _POOLING_OP_TYPES:
            msg = (
                f"{node.name} is a {node.op_type} node, but the parsing method for"
                " MaxPool nodes was called. This could indicate changes in the"
                " network being parsed."
            )
            raise ValueError(msg)
        pool_func_name = "max"

        # ONNX network should not contain indices output from MaxPool -
        # not supported by OMLT
        if len(node.output) != 1:
            msg = (
                "The ONNX network contains indices output from MaxPool. This is not"
                " supported by OMLT."
            )
            raise ValueError(msg)
        if len(node.input) != MAXPOOL_INPUT_DIMENSIONS:
            msg = (
                f"{node.name} input has {len(node.input)} dimensions, only nodes with "
                "1 input dimension can be used as starting points for parsing."
            )
            raise ValueError(msg)
        input_layer, transformer = self._node_input_and_transformer(node.input[0])
        input_output_size = _get_input_output_size(input_layer, transformer)

        # currently only support 2D image with channels.
        if len(input_output_size) == MAXPOOL_INPUT_OUTPUT_W_BATCHES:
            # this means there is an extra dimension for number of batches
            # batches not supported, so only accept if they're not there or there is
            # only 1 batch
            if input_output_size[0] != 1:
                msg = (
                    f"{node.name} has {input_output_size[0]} batches, only single batch"
                    " is supported."
                )
                raise ValueError(msg)
            input_output_size = input_output_size[1:]

        in_channels = input_output_size[0]

        attr = _collect_attributes(node)
        kernel_depth = attr["kernel_shape"][0]
        kernel_shape = attr["kernel_shape"][1:]
        strides = attr["strides"] if "strides" in attr else [1] * len(kernel_shape)

        # check only kernel shape, stride, storage order are set
        # everything else is not supported
        if "dilations" in attr and attr["dilations"] != [1, 1]:
            msg = (
                f"{node.name} has non-identity dilations ({attr['dilations']})."
                " This is not supported."
            )
            raise ValueError(msg)
        if "pads" in attr and np.any(attr["pads"]):
            msg = (
                f"{node.name} has non-zero pads ({attr['pads']})."
                " This is not supported."
            )
            raise ValueError(msg)
        if ("auto_pad" in attr) and (attr["auto_pad"] != "NOTSET"):
            msg = (
                f"{node.name} has autopad set ({attr['auto_pad']})."
                " This is not supported."
            )
            raise ValueError(msg)
        if len(kernel_shape) != len(strides):
            msg = (
                f"Kernel shape {kernel_shape} has {len(kernel_shape)} dimensions. "
                f"Strides attribute has {len(strides)} dimensions. These must be equal."
            )
            raise ValueError(msg)
        if len(input_output_size) != len(kernel_shape) + 1:
            msg = (
                f"Input/output size ({input_output_size}) must have one more dimension"
                f" than kernel shape ({kernel_shape})."
            )
            raise ValueError(msg)

        output_shape_wrapper: Callable[[float], int] = math.floor
        if "ceil_mode" in attr and attr["ceil_mode"] == 1:
            output_shape_wrapper = math.ceil

        output_size = [in_channels] + [
            output_shape_wrapper(
                (input_output_size[i] - kernel_shape[i - 1]) / strides[i - 1] + 1
            )
            for i in range(1, len(input_output_size))
        ]

        activation = "linear"
        if len(next_nodes) == 1:
            # check if Relu
            type_, maybe_node, maybe_next_nodes = self._nodes[next_nodes[0]]
            if maybe_node.op_type in _ACTIVATION_OP_TYPES:
                node = maybe_node
                activation = maybe_node.op_type.lower()
                next_nodes = maybe_next_nodes

        pooling_layer = PoolingLayer2D(
            input_output_size,
            output_size,
            strides,
            pool_func_name,
            tuple(kernel_shape),
            kernel_depth,
            activation=activation,
            input_index_mapper=transformer,
        )
        self._node_map[node.name] = pooling_layer
        self._node_map[node.output[0]] = pooling_layer

        return next_nodes, pooling_layer, [input_layer]

    def _node_input_and_transformer(self, node_name):
        maybe_layer = self._node_map[node_name]
        if isinstance(maybe_layer, tuple):
            transformer, input_layer = maybe_layer
            return input_layer, transformer
        return maybe_layer, None


def _collect_attributes(node):
    r = {}
    for attr in node.attribute:
        if attr.type == ATTR_FLOAT:  # FLOAT
            r[attr.name] = attr.f
        elif attr.type == ATTR_INT:  # INT
            r[attr.name] = int(attr.i)
        elif attr.type == ATTR_TENSOR:  # TENSOR
            r[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == ATTR_INTS:  # INTS
            r[attr.name] = list(attr.ints)
        else:
            msg = f"unhandled attribute type {attr.type}"
            raise RuntimeError(msg)
    return r


def _parse_constant_value(node):
    attr = _collect_attributes(node)
    return attr["value"]


def _get_input_output_size(input_layer, transformer):
    if transformer is not None:
        return transformer.output_size
    return input_layer.output_size
