import tensorflow.keras as keras
import pytest

from omlt.io.keras_reader import load_keras_sequential


def test_keras_reader(datadir):
    nn = keras.models.load_model(datadir.file("keras_linear_131"), compile=False)
    net = load_keras_sequential(nn)

    layers = list(net.layers)
    assert len(layers) == 3
    for layer in layers:
        assert layer.activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(datadir.file("keras_linear_131_sigmoid"), compile=False)
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == 3
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(
        datadir.file("keras_linear_131_sigmoid_output_activation"), 
        compile=False
        )
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == 3
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "sigmoid"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(datadir.file("big"), compile=False)
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == 5
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "sigmoid"
    assert layers[3].activation == "sigmoid"
    assert layers[4].activation == "softplus"
    assert layers[1].weights.shape == (1, 100)
    assert layers[2].weights.shape == (100,100)
    assert layers[3].weights.shape == (100,100)
    assert layers[4].weights.shape == (100,1)
    
