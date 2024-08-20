import warnings

import numpy as np

from omlt.neuralnet.layer import DenseLayer, GNNLayer, InputLayer, Layer
from omlt.neuralnet.network_definition import NetworkDefinition


def _compute_gcn_norm(A):
    """Calculate the norm for a GCN layer.

    Parameters
    ----------
    A : matrix-like
        the adjacency matrix.
    """
    N = A.shape[0]
    Ahat = A + np.eye(N)
    degrees = np.sum(Ahat, axis=0)
    gcn_norm = np.zeros(A.shape)
    for u in range(N):
        for v in range(N):
            gcn_norm[u, v] = Ahat[u, v] / np.sqrt(degrees[u] * degrees[v])
    return gcn_norm


def _compute_sage_norm(A, aggr):
    """Calculate the norm for a SAGE layer.

    Parameters
    ----------
    A : matrix-like
        the adjacency matrix.
    aggr : str
        the aggregation function, "sum" (default) or "mean"
    """
    N = A.shape[0]
    # sum aggregation
    sage_norm = A + np.eye(N)
    # mean aggregation
    if aggr == "mean":
        degrees = np.sum(A, axis=0)
        for u in range(N):
            for v in range(N):
                if u != v and degrees[u] > 0:
                    sage_norm[u, v] = sage_norm[u, v] / degrees[u]
    return sage_norm


def _process_gnn_parameters(gnn_weights_uv, gnn_weights_vv, gnn_biases, gnn_norm):
    """Construct the weights and biases for the GNNLayer class.

    Parameters
    ----------
    gnn_weights_uv : matrix-like
        the weights between two different nodes, shape: (out_channels, in_channels).
    gnn_weights_vv : matrix-like
        the weights between the same node, shape: (out_channels, in_channels).
    gnn_biases : array-like
        the biases, shape: (out_channels, )
    gnn_norm : matrix-like
        the norm for the GNN layer, shape: (N, N)

    Returns:
    -------
    weights : matrix-like
        the weights for the GNNLayer class, shape: (N * in_channels, N * out_channels)
    biases: array-like
        the biases for the GNNLayer class, shape: (N * out_channels, )
    """
    out_channels, in_channels = gnn_weights_uv.shape
    N = gnn_norm.shape[0]
    weights = np.zeros((N * in_channels, N * out_channels), dtype=gnn_weights_uv.dtype)
    biases = np.zeros(N * out_channels, dtype=gnn_biases.dtype)
    for output_index in range(N * out_channels):
        biases[output_index] = gnn_biases[output_index % out_channels]
        for input_index in range(N * in_channels):
            input_node_index = input_index // in_channels
            output_node_index = output_index // out_channels
            if input_node_index != output_node_index:
                weights[input_index, output_index] = (
                    gnn_norm[output_node_index, input_node_index]
                    * gnn_weights_uv[
                        output_index % out_channels, input_index % in_channels
                    ]
                )
            else:
                weights[input_index, output_index] = (
                    gnn_norm[output_node_index, input_node_index]
                    * gnn_weights_vv[
                        output_index % out_channels, input_index % in_channels
                    ]
                )
    return weights, biases


_LAYER_OP_TYPES_FIXED_GRAPH = ["Linear", "GCNConv", "SAGEConv"]
_LAYER_OP_TYPES_NON_FIXED_GRAPH = ["Linear", "SAGEConv"]
_ACTIVATION_OP_TYPES = ["ReLU", "Sigmoid", "LogSoftmax", "Softplus", "Tanh"]
_POOLING_OP_TYPES = ["global_mean_pool", "global_add_pool"]
_AGGREGATION_OP_TYPES = ["sum", "mean"]
_OP_TYPES = _LAYER_OP_TYPES_FIXED_GRAPH + _ACTIVATION_OP_TYPES + _POOLING_OP_TYPES


def load_torch_geometric_sequential(  # noqa: C901, PLR0913, PLR0912, PLR0915
    nn,
    N,
    A=None,
    scaling_object=None,
    scaled_input_bounds=None,
    unscaled_input_bounds=None,
):
    """Load a torch_geometric graph neural network model.

    Load a torch_geometric graph neural network model (built with Sequential) into
    an OMLT network definition object. This network definition object
    can be used in different formulations.

    Parameters
    ----------
    nn : torch_geometric.model
        A torch_geometric model that was built with Sequential
    N : int
        The number of nodes of input graph
    A : matrix-like
        The adjacency matrix of input graph
    scaling_object : instance of ScalingInterface or None
        Provide an instance of a scaling object to use to scale iputs --> scaled_inputs
        and scaled_outputs --> outputs. If None, no scaling is performed. See
        scaling.py.
    scaled_input_bounds : dict or None
        A dict that contains the bounds on the scaled variables (the
        direct inputs to the neural network). If None, then no bounds
        are specified or they are generated using unscaled bounds.
    unscaled_input_bounds : dict or None
        A dict that contains the bounds on the unscaled variables (the
        direct inputs to the neural network). If specified the scaled_input_bounds
        dictionary will be generated using the provided scaling object.
        If None, then no bounds are specified.

    Returns:
    -------
    NetworkDefinition
    """
    n_inputs = N * nn[0].in_channels

    net = NetworkDefinition(
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_input_bounds,
        unscaled_input_bounds=unscaled_input_bounds,
    )

    prev_layer: Layer = InputLayer([n_inputs])
    net.add_layer(prev_layer)

    operations = []
    for layer in nn:
        op_name = None
        if layer.__class__.__name__ == "function":
            op_name = layer.__name__
        else:
            op_name = layer.__class__.__name__

        if op_name not in _OP_TYPES:
            msg = f"Operation {op_name} is not supported."
            raise ValueError(msg)
        operations.append(op_name)

    if A is None:
        supported_layers = {
            "Linear",
            *_ACTIVATION_OP_TYPES,
            *_POOLING_OP_TYPES,
        }
        # If A is None, then the graph is not fixed.
        # Only layers in _LAYER_OP_TYPES_NON_FIXED_GRAPH are supported.
        # Only "sum" aggregation is supported.
        # Since all weights and biases are possibly needed, A is set to correspond to a
        # complete graph.
        for index, layer in enumerate(nn):
            if operations[index] in supported_layers:
                # nonlinear activation results in a MINLP
                if operations[index] in {"Sigmoid", "LogSoftmax", "Softplus", "Tanh"}:
                    warnings.warn(
                        "nonlinear activation results in a MINLP", stacklevel=2
                    )
                # Linear layers, all activation functions, and all pooling functions are
                # still supported.
                continue
            if operations[index] not in _LAYER_OP_TYPES_NON_FIXED_GRAPH:
                msg = "this layer is not supported when the graph is not fixed."
                raise ValueError(msg)
            if layer.aggr != "sum":
                msg = "this aggregation is not supported when the graph is not fixed"
                raise ValueError(msg)

        A = np.ones((N, N)) - np.eye(N)

    for index, layer in enumerate(nn):
        if operations[index] in _ACTIVATION_OP_TYPES:
            # Skip activation layers since they are already handled in last layer
            continue

        activation = None
        if index + 1 < len(nn) and operations[index + 1] in _ACTIVATION_OP_TYPES:
            # Check if this layer has an activation function
            activation = operations[index + 1].lower()

        if operations[index] == "Linear":
            gnn_weights = layer.weight.detach().numpy()
            gnn_biases = layer.bias.detach().numpy()
            # A linear layer is either applied on each node's features (i.e.,
            # prev_layer.output_size[-1] = N * gnn_weights.shape[1])
            # or the features after pooling (i.e.,
            # prev_layer.output_size[-1] = gnn_weights.shape[1])
            gnn_norm = np.eye(prev_layer.output_size[-1] // gnn_weights.shape[1])
            weights, biases = _process_gnn_parameters(
                gnn_weights, gnn_weights, gnn_biases, gnn_norm
            )
            n_layer_inputs, n_layer_outputs = weights.shape
            curr_layer = DenseLayer(
                [n_layer_inputs],
                [n_layer_outputs],
                activation=activation,
                weights=weights,
                biases=biases,
            )
        elif operations[index] == "GCNConv":
            gnn_weights = layer.lin.weight.detach().numpy()
            gnn_biases = layer.bias.detach().numpy()
            gnn_norm = _compute_gcn_norm(A)
            weights, biases = _process_gnn_parameters(
                gnn_weights, gnn_weights, gnn_biases, gnn_norm
            )
            n_layer_inputs, n_layer_outputs = weights.shape
            curr_layer = GNNLayer(
                [n_layer_inputs],
                [n_layer_outputs],
                activation=activation,
                weights=weights,
                biases=biases,
                N=N,
            )
        elif operations[index] == "SAGEConv":
            gnn_weights_uv = layer.lin_l.weight.detach().numpy()
            gnn_biases = layer.lin_l.bias.detach().numpy()
            gnn_weights_vv = np.zeros(shape=gnn_weights_uv.shape)
            if layer.root_weight:
                gnn_weights_vv = layer.lin_r.weight.detach().numpy()
            gnn_norm = _compute_sage_norm(A, layer.aggr)
            weights, biases = _process_gnn_parameters(
                gnn_weights_uv, gnn_weights_vv, gnn_biases, gnn_norm
            )
            n_layer_inputs, n_layer_outputs = weights.shape
            curr_layer = GNNLayer(
                [n_layer_inputs],
                [n_layer_outputs],
                activation=activation,
                weights=weights,
                biases=biases,
                N=N,
            )
        elif operations[index] in _POOLING_OP_TYPES:
            # Both mean and add pooling layers can be transformed into a DenseLayer
            n_layer_inputs = prev_layer.output_size[-1]
            n_layer_outputs = prev_layer.output_size[-1] // N
            weights = np.zeros((n_layer_inputs, n_layer_outputs))
            biases = np.zeros(n_layer_outputs)
            for input_index in range(n_layer_inputs):
                for output_index in range(n_layer_outputs):
                    if input_index % n_layer_outputs == output_index:
                        # mean pooling
                        if operations[index] == "global_mean_pool":
                            weights[input_index, output_index] = 1.0 / N
                        # add pooling
                        else:
                            weights[input_index, output_index] = 1.0
            curr_layer = DenseLayer(
                [n_layer_inputs],
                [n_layer_outputs],
                weights=weights,
                biases=biases,
            )
        net.add_layer(curr_layer)
        net.add_edge(prev_layer, curr_layer)
        prev_layer = curr_layer
    return net
