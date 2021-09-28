from onnx import numpy_helper
from pyoml.opt.network_definition import NetworkDefinition


def load_onnx_neural_network(onnx):
    """
    Load a NetworkDefinition from an onnx object.
    """
    parser = NetworkParser()
    return parser.parse_network(onnx.graph)


class NetworkParser:
    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        self._graph = None
        self._initializers = None
        self._nodes = None
        self._nodes_by_output = None
        self._inputs = None
        self._outputs = None
        self._node_stack = None
        self._node_id = 0
        self._node_map = None

        self._weights = None
        self._biases = None
        self._activations = None

    def parse_network(self, graph):
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

        for input in self._graph.input:
            nodes[input.name] = ('input', input.type, [])
            nodes_by_output[input.name] = input.name
            inputs.add(input.name)
            nid = self._next_node_id()
            self._node_map[input.name] = [nid]

        for output in self._graph.output:
            nodes[output.name] = ('output', output.type, [])
            outputs.add(output.name)

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

        for node in self._graph.node:
            # add node not connected to anything
            self._nodes[node.name] = ('node', node, [])

            # Map inputs by their output name
            node_inputs = [
                nodes_by_output[input]
                for input in node.input
                if input not in initializers
            ]

            # Now connect inputs to the current node
            for input in node_inputs:
                self._nodes[input][2].append(node.name)

        # travers graph
        self._node_stack = list(inputs)

        self._weights = dict()
        self._biases = dict()
        self._activations = dict()

        while self._node_stack:
            node_name = self._node_stack.pop()
            type_, node, next_nodes = self._nodes[node_name]

            # no need to process inputs or outputs
            if type_ == 'node':
                self._visit_node(node, next_nodes)
            else:
                for next in next_nodes:
                    self._node_stack.append(next)

        n_inputs = len(inputs)
        n_outputs = len(outputs)
        n_hidden = len(self._weights) - n_outputs

        return NetworkDefinition(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs,
            weights=self._weights,
            biases=self._biases,
            activations=self._activations
        )

    def _next_node_id(self):
        n = self._node_id
        self._node_id += 1
        return n

    def _visit_node(self, node, next_nodes):
        if node.op_type == 'MatMul':
            next_nodes = self._consume_dense_nodes(node, next_nodes)
            for next in next_nodes:
                self._node_stack.append(next)
        else:
            raise Exception(f'Unhandled node type {node.op_type}')

    def _consume_dense_nodes(self, node, next_nodes):
        """Starting from a MatMul node, consume nodes to form a dense Ax + b node."""
        assert node.op_type == 'MatMul'
        assert len(node.input) == 2

        [in_0, in_1] = list(node.input)

        if in_0 in self._initializers:
            node_weights = self._initializers[in_0]
            input_node = self._node_map[in_1]
        else:
            assert in_1 in self._initializers
            node_weights = self._initializers[in_1]
            input_node = self._node_map[in_0]

        assert len(next_nodes) == 1

        # expect 'Add' node ahead
        type_, node, maybe_next_nodes = self._nodes[next_nodes[0]]
        assert type_ == 'node'
        assert node.op_type == 'Add'

        # extract biases
        next_nodes = maybe_next_nodes
        assert len(node.input) == 2
        [in_0, in_1] = list(node.input)

        if in_0 in self._initializers:
            node_biases = self._initializers[in_0]
        else:
            assert in_1 in self._initializers
            node_biases = self._initializers[in_1]

        assert node_weights.shape[1] == node_biases.shape[0]
        new_node_ids = [
            self._next_node_id() for _ in node_biases
        ]

        assert len(node.output) == 1

        self._node_map[node.output[0]] = new_node_ids

        # update weights and biases
        assert len(input_node) == node_weights.shape[0]

        for i, nid in enumerate(new_node_ids):
            weights = dict(
                (in_nid, node_weights[in_idx][i])
                for in_idx, in_nid in enumerate(input_node)
            )
            self._weights[nid] = weights
            self._biases[nid] = node_biases[i]
            # TODO: use real activations
            self._activations[nid] = 'linear'

        # look for activation type, if any
        return next_nodes