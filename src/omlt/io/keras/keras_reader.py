from tensorflow import keras

from omlt.neuralnet.layer import DenseLayer, InputLayer, Layer
from omlt.neuralnet.network_definition import NetworkDefinition


def load_keras_sequential(
    nn, scaling_object=None, scaled_input_bounds=None, unscaled_input_bounds=None
):
    """Load Keras sequential network.

    Load a keras neural network model (built with Sequential) into
    an OMLT network definition object. This network definition object
    can be used in different formulations.

    Parameters
    ----------
    nn : keras.model
        A keras model that was built with Sequential
    scaling_object : instance of ScalingInterface or None
        Provide an instance of a scaling object to use to scale inputs --> scaled_inputs
        and scaled_outputs --> outputs. If None, no scaling is performed.
        See scaling.py.
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
    n_inputs = len(nn.layers[0].get_weights()[0])

    net = NetworkDefinition(
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_input_bounds,
        unscaled_input_bounds=unscaled_input_bounds,
    )

    prev_layer: Layer = InputLayer([n_inputs])
    net.add_layer(prev_layer)

    for layer in nn.layers:
        cfg = layer.get_config()
        if not isinstance(layer, keras.layers.Dense):
            msg = (
                f"Layer type {type(layer)} encountered. The load_keras_sequential "
                "function only supports dense layers at this time. Consider using "
                "ONNX and the ONNX parser."
            )
            raise TypeError(msg)
        weights, biases = layer.get_weights()
        n_layer_inputs, n_layer_nodes = weights.shape

        dense_layer = DenseLayer(
            [n_layer_inputs],
            [n_layer_nodes],
            activation=cfg["activation"],
            weights=weights,
            biases=biases,
        )
        net.add_layer(dense_layer)
        net.add_edge(prev_layer, dense_layer)
        prev_layer = dense_layer

    return net
