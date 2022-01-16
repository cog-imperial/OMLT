import pytest
from pathlib import Path
import numpy as np

from pyomo.common.fileutils import this_file_dir

from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.layer import DenseLayer, InputLayer



def get_neural_network_data(desc):
    """
    Return input and test data for a neural network.

    Parameters
    ----------
    desc : string
        model name. One of 131 or 2353.
    """
    if desc == "131":
        # build data with 1 input and 1 output and 500 data points
        x = np.random.uniform(-1, 1, 500)
        y = np.sin(x)
        x_test = np.random.uniform(-1, 1, 5)
        return x, y, x_test

    elif desc == "2353":
        # build data with 2 inputs, 3 outputs, and 500 data points
        np.random.seed(42)
        x = np.random.uniform([-1, 2], [1, 3], (500, 2))
        y1 = np.sin(x[:, 0] * x[:, 1])
        y2 = x[:, 0] + x[:, 1]
        y3 = np.cos(x[:, 0] / x[:, 1])
        y = np.column_stack((y1, y2, y3))
        x_test = np.random.uniform([-1, 2], [1, 3], (5, 2))
        return x, y, x_test

    return None


class _Datadir:
    """
    Give access to files in the `models` directory.
    """

    def __init__(self, basedir):
        self._basedir = basedir

    def file(self, filename):
        return str(self._basedir / filename)


@pytest.fixture
def datadir():
    basedir = Path(this_file_dir()) / "models"
    return _Datadir(basedir)


@pytest.fixture
def two_node_network_relu():
    """
            1           1
    x0 -------- (1) --------- (3)
     |                   /
     |                  /
     |                 / 5
     |                /
     |               |
     |    -1         |     1
     ---------- (2) --------- (4)
    """
    net = NetworkDefinition(scaled_input_bounds={0: (-10.0, 10.0)})

    input_layer = InputLayer([1])
    net.add_layer(input_layer)

    dense_layer_0 = DenseLayer(
        input_layer.output_size,
        [1, 2],
        activation="relu",
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([0.0, 0.0])
    )
    net.add_layer(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    return net

