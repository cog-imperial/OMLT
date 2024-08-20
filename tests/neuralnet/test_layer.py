import numpy as np
import pytest

from omlt.neuralnet.layer import (
    ConvLayer2D,
    DenseLayer,
    GNNLayer,
    IndexMapper,
    InputLayer,
    PoolingLayer2D,
)


def test_input_layer():
    layer = InputLayer([1, 2, 3])
    input_indexes = list(layer.input_indexes)
    assert input_indexes == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]


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


def test_convolutional_layer():
    # Test data adapted from https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    # Public methods to test are eval and kernel_with_input_indexes,
    # but eval must call kernel_with_input_indexes on all output indexes to work,
    # so we only need to test eval.
    x = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 7, 5) input tensor
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
            ]
        ]
    ).astype(np.float32)
    weights = np.array(
        [
            [
                [
                    [1.0, 0.0, 2.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [0.0, 1.0, 0.0],
                    [2.0, 0.0, 1.0],
                ]
            ]
        ]
    ).astype(np.float32)
    layer = ConvLayer2D([1, 7, 5], [1, 3, 3], [2, 1], weights)
    y = layer.eval_single_layer(x)
    assert np.array_equal(y, [[[42, 49, 56], [112, 119, 126], [182, 189, 196]]])


def test_maxpool_layer():
    # Test data adapted from https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    # Public methods to test are eval and kernel_with_input_indexes,
    # but eval must call kernel_with_input_indexes on all output indexes to work,
    # so we only need to test eval.
    x = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]
        ]
    ).astype(np.float32)
    layer = PoolingLayer2D([1, 4, 4], [1, 2, 2], [2, 2], "max", [3, 3], 1)
    y = layer.eval_single_layer(x)
    assert np.array_equal(y, [[[11, 12], [15, 16]]])


def test_gnn_layer_with_input_index_mapper():
    weights = np.array(
        [
            [1, 0, 1, 1, -1, 1, 1, -1, 1],
            [0, 1, 1, -1, 1, 1, -1, 1, 1],
            [1, -1, 1, 1, 0, 1, 1, -1, 1],
            [-1, 1, 1, 0, 1, 1, -1, 1, 1],
            [1, -1, 1, 1, -1, 1, 1, 0, 1],
            [-1, 1, 1, -1, 1, 1, 0, 1, 1],
        ]
    )

    biases = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])

    # input has size [6], but the previous node output is [2, 3]
    # use mapper to map between the two
    t = IndexMapper([1, 2, 2, 3], [1, 2, 6])
    layer = GNNLayer([1, 2, 6], [1, 2, 9], weights, biases, N=3, input_index_mapper=t)

    inputs = np.array([[[[-3, 2, -1], [1, -2, 3]], [[0, 0, 0], [0, 0, 0]]]])

    A1 = np.ones([3, 3], dtype=int)
    y1 = np.array(
        [[[-11, 9, 1, -12, 11, 1, -10, 10, 1], [-1, 0, 1, -1, 0, 1, -1, 0, 1]]]
    )
    assert np.array_equal(layer._eval_with_adjacency(inputs, A1), y1)
    assert np.array_equal(layer.eval_single_layer(inputs), y1)

    A2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y2 = np.array([[[-4, 2, 0, -2, 1, 1, -3, 3, 2], [-1, 0, 1, -1, 0, 1, -1, 0, 1]]])
    assert np.array_equal(layer._eval_with_adjacency(inputs, A2), y2)

    A3 = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    y3 = np.array([[[-6, 4, 0, -12, 11, 1, -5, 5, 2], [-1, 0, 1, -1, 0, 1, -1, 0, 1]]])
    assert np.array_equal(layer._eval_with_adjacency(inputs, A3), y3)

    expected_msg = (
        "Input size must equal to the number of nodes multiplied by the number of"
        " input node features"
    )
    with pytest.raises(ValueError, match=expected_msg):
        layer = GNNLayer([5], [9], weights, biases, N=3)

    expected_msg = (
        "Output size must equal to the number of nodes multiplied by the number of"
        " output node features"
    )
    with pytest.raises(ValueError, match=expected_msg):
        layer = GNNLayer([6], [8], weights, biases, N=3)
