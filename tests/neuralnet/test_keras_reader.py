import os.path

import keras
import pytest
from pyomo.common.fileutils import this_file_dir

from omlt.neuralnet.keras_reader import load_keras_sequential


def test_keras_reader():
    nn = keras.models.load_model(
        os.path.join(this_file_dir(), "models/keras_linear_131")
    )
    net = load_keras_sequential(nn)
    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    assert len(net.activations) == 4
    for k in net.activations:
        assert net.activations[k] == "linear"  # or net.activations[k] is None

    nn = keras.models.load_model(
        os.path.join(this_file_dir(), "./models/keras_linear_131_sigmoid")
    )
    net = load_keras_sequential(nn)
    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert sorted(net.weights.keys()) == [1, 2, 3, 4]
    assert len(net.biases) == 4
    assert sorted(net.biases.keys()) == [1, 2, 3, 4]
    assert len(net.activations) == 4
    assert sorted(net.activations.keys()) == [1, 2, 3, 4]

    for k in [1, 2, 3]:
        assert net.activations[k] == "sigmoid"
    assert net.activations[4] == "linear"  # or net.activations[4] is None

    nn = keras.models.load_model(
        os.path.join(
            this_file_dir(), "./models/keras_linear_131_sigmoid_output_activation"
        )
    )
    net = load_keras_sequential(nn)
    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert sorted(net.weights.keys()) == [1, 2, 3, 4]
    assert len(net.biases) == 4
    assert sorted(net.biases.keys()) == [1, 2, 3, 4]
    assert len(net.activations) == 4
    assert sorted(net.activations.keys()) == [1, 2, 3, 4]

    for k in [1, 2, 3, 4]:
        assert net.activations[k] == "sigmoid"

    nn = keras.models.load_model(os.path.join(this_file_dir(), "./models/big"))
    net = load_keras_sequential(nn)
    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 100 * 3
    assert len(net.weights) == 100 * 3 + 1
    assert len(net.biases) == 100 * 3 + 1
    assert len(net.activations) == 100 * 3 + 1

    for k in range(1, 100 * 3 + 1):
        assert net.activations[k] == "sigmoid"
    assert net.activations[100 * 3 + 1] == "softplus"
