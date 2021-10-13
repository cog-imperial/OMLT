from omlt.neuralnet.network_definition import NetworkDefinition


def load_keras_sequential(nn, scaling_object=None, input_bounds=None):
    """
    Load a keras neural network model (built with Sequential) into
    a pyoml network definition object. This network definition object
    can be used in different formulations.
    Parameters
    ----------
    nn : keras.model
        A keras model that was built with Sequential
    scaling_object : instance of object supporting ScalingInterface (see scaling.py)
    input_bounds: list of tuples
    Returns
    -------
    NetworkDefinition
    """
    # Todo: Add support for DistributionLambda layers
    n_inputs = len(nn.layers[0].get_weights()[0])
    n_outputs = len(nn.layers[-1].get_weights()[1])
    node_id_offset = n_inputs
    layer_offset = 0
    w = dict()
    b = dict()
    a = dict()
    for l in nn.layers:
        cfg = l.get_config()
        weights, biases = l.get_weights()
        n_layer_inputs, n_layer_nodes = weights.shape
        for i in range(n_layer_nodes):
            layer_w = dict()
            for j in range(n_layer_inputs):
                layer_w[j + layer_offset] = weights[j, i]
            w[node_id_offset] = layer_w
            b[node_id_offset] = biases[i]
            # ToDo: leaky ReLU
            a[node_id_offset] = cfg["activation"]
            node_id_offset += 1
        layer_offset += n_layer_inputs
    n_nodes = len(a) + n_inputs
    n_hidden = n_nodes - n_inputs - n_outputs
    return NetworkDefinition(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        weights=w,
        biases=b,
        activations=a,
        scaling_object=scaling_object,
        input_bounds=input_bounds,
    )
