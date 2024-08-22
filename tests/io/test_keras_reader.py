import pytest

from omlt.dependencies import keras, keras_available

NUM_LAYERS_131 = 3
NUM_LAYERS_BIG = 5

if keras_available:
    from omlt.io import load_keras_sequential


@pytest.mark.skipif(
    not keras_available, reason="Test only valid when keras is available"
)
def test_keras_reader(datadir):
    nn = keras.models.load_model(datadir.file("keras_linear_131.keras"), compile=False)
    net = load_keras_sequential(nn)

    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    for layer in layers:
        assert layer.activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(
        datadir.file("keras_linear_131_sigmoid.keras"), compile=False
    )
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(
        datadir.file("keras_linear_131_sigmoid_output_activation.keras"), compile=False
    )
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "sigmoid"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)

    nn = keras.models.load_model(datadir.file("big.keras"), compile=False)
    net = load_keras_sequential(nn)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_BIG
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "sigmoid"
    assert layers[3].activation == "sigmoid"
    assert layers[4].activation == "softplus"
    assert layers[1].weights.shape == (1, 100)
    assert layers[2].weights.shape == (100, 100)
    assert layers[3].weights.shape == (100, 100)
    assert layers[4].weights.shape == (100, 1)
