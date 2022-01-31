from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import InputLayer, DenseLayer
import tensorflow.keras as keras

def load_keras_sequential(nn, scaling_object=None, scaled_input_bounds=None):
    """
    Load a keras neural network model (built with Sequential) into
    an OMLT network definition object. This network definition object
    can be used in different formulations.
    Parameters
    ----------
    nn : keras.model
        A keras model that was built with Sequential
    scaling_object : instance of ScalingInterface or None
        Provide an instance of a scaling object to use to scale iputs --> scaled_inputs
        and scaled_outputs --> outputs. If None, no scaling is performed. See scaling.py.
    scaled_input_bounds : dict or None
        A dict that contains the bounds on the scaled variables (the
        direct inputs to the neural network). If None, then no bounds
        are specified.
    
    Returns
    -------
    NetworkDefinition
    """
    # TODO: Add exceptions for unsupported layer types
    n_inputs = len(nn.layers[0].get_weights()[0])

    net = NetworkDefinition(scaling_object=scaling_object, scaled_input_bounds=scaled_input_bounds)

    prev_layer = InputLayer([n_inputs])
    net.add_layer(prev_layer)

    for l in nn.layers:
        cfg = l.get_config()
        if not isinstance(l, keras.layers.Dense):
            raise ValueError('Layer type {} encountered. The function load_keras_sequential '
                             'only supports dense layers at this time. Consider using '
                             'ONNX and the ONNX parser'.format(type(l)))
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
