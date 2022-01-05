from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import InputLayer, DenseLayer

def load_keras_sequential(nn, scaling_object=None, scaled_input_bounds=None):
    """
    Load a keras neural network model (built with Sequential) into
    a pyoml network definition object. This network definition object
    can be used in different formulations.
    Parameters
    ----------
    nn : keras.model
        A keras model that was built with Sequential
    scaling_object : instance of object supporting ScalingInterface (see scaling.py)
    scaled_input_bounds: list of tuples
    
    Returns
    -------
    NetworkDefinition
    """
    # TODO: Add exceptions for unsupported layer types
    n_inputs = len(nn.layers[0].get_weights()[0])
    print('n_inputs:', n_inputs)

    net = NetworkDefinition(scaled_input_bounds=scaled_input_bounds)

    prev_layer = InputLayer([n_inputs])
    net.add_layer(prev_layer)

    for l in nn.layers:
        cfg = l.get_config()
        weights, biases = l.get_weights()
        n_layer_inputs, n_layer_nodes = weights.shape

        dense_layer = DenseLayer([n_layer_inputs],
                [n_layer_nodes],
                activation=cfg["activation"],
                weights=weights,
                biases=biases)
        net.add_layer(dense_layer)
        net.add_edge(prev_layer, dense_layer)
        prev_layer = dense_layer

    return net
