import pytest
import numpy as np

from omlt.neuralnet.layer import (
    InputLayer,
    DenseLayer,
    IndexMapper,
)

def test_input_layer():
    layer = InputLayer([1, 2, 3])
    input_indexes = list(layer.input_indexes)
    assert input_indexes == [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2)]


def test_dense_layer_with_input_index_mapper():
    w = np.ones((3, 2))
    b = np.ones(2)

    # input has size [2, 3], but the previous node output is [3, 2]
    # use mapper to map between the two
    t = IndexMapper([3, 2], [2, 3])
    layer = DenseLayer([2, 3], [2, 2], w, b, input_index_mapper=t)

    index_iter = layer.input_indexes_with_input_layer_indexes

    assert next(index_iter) == ((0, 0), (0, 0))
    assert next(index_iter) == ((0, 1), (0, 1))
    assert next(index_iter) == ((0, 2), (1, 0))
    assert next(index_iter) == ((1, 0), (1, 1))
    assert next(index_iter) == ((1, 1), (2, 0))
    assert next(index_iter) == ((1, 2), (2, 1))

@pytest.mark.skip('Need to add test for ConvLayer')
def test_convolutional_layer():
    assert False
