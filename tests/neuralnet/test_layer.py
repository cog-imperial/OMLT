import numpy as np
import pytest

from omlt.neuralnet.layer import ConvLayer, DenseLayer, IndexMapper, InputLayer, PoolingLayer


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
    x = np.array([[[0., 1., 2., 3., 4.],  # (1, 7, 5) input tensor
                    [5., 6., 7., 8., 9.],
                    [10., 11., 12., 13., 14.],
                    [15., 16., 17., 18., 19.],
                    [20., 21., 22., 23., 24.],
                    [25., 26., 27., 28., 29.],
                    [30., 31., 32., 33., 34.]]]).astype(np.float32)
    weights = np.array([[[[1., 0., 2.],  # (1, 1, 3, 3) tensor for convolution weights
                          [0., 1., 0.],
                          [2., 0., 1.]]]]).astype(np.float32)
    layer = ConvLayer([1, 7, 5], [1, 3, 3], [2, 1], weights)
    y = layer.eval(x)
    assert np.array_equal(y, [[[42, 49, 56],
                               [112, 119, 126],
                               [182, 189, 196]]])


def test_maxpool_layer():
    # Test data adapted from https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
    # Public methods to test are eval and kernel_with_input_indexes,
    # but eval must call kernel_with_input_indexes on all output indexes to work,
    # so we only need to test eval.
    x = np.array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]).astype(np.float32)
    layer = PoolingLayer([1, 4, 4], [1, 2, 2], [2, 2], 'max', [3, 3], 1)
    y = layer.eval(x)
    assert np.array_equal(y, [[[11, 12], [15, 16]]])
