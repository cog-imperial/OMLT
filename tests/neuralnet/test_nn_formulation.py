import re
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pyomo.environ as pyo
import pytest
from pyomo.contrib.fbbt import interval

from omlt import OmltBlock
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    NetworkDefinition,
    ReducedSpaceNNFormulation,
    ReducedSpaceSmoothNNFormulation,
    ReluPartitionFormulation,
)
from omlt.neuralnet.layer import (
    ConvLayer2D,
    DenseLayer,
    GNNLayer,
    IndexMapper,
    InputLayer,
    PoolingLayer2D,
)
from omlt.neuralnet.layers.full_space import (
    _input_layer_and_block,
    full_space_maxpool2d_layer,
)
from omlt.neuralnet.layers.partition_based import (
    default_partition_split_func,
    partition_based_dense_relu_layer,
)
from omlt.neuralnet.layers.reduced_space import reduced_space_dense_layer

if TYPE_CHECKING:
    from omlt.formulation import _PyomoFormulation

formulations = {
    "FullSpace": FullSpaceNNFormulation,
    "ReducedSpace": ReducedSpaceNNFormulation,
    "relu": ReluPartitionFormulation,
}

NEAR_EQUAL = 1e-6
FULLSPACE_SMOOTH_VARS = 15
FULLSPACE_SMOOTH_CONSTRAINTS = 14
FULLSPACE_RELU_VARS = 19
FULLSPACE_RELU_CONSTRAINTS = 26
REDUCED_VARS = 6
REDUCED_CONSTRAINTS = 5
THREE_NODE_VARS = 81
THREE_NODE_CONSTRAINTS = 120


def two_node_network(activation, input_value):
    """Two node network.

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
    net = NetworkDefinition(scaled_input_bounds=[(-10.0, 10.0)])

    input_layer = InputLayer([1])
    net.add_layer(input_layer)

    dense_layer_0 = DenseLayer(
        input_layer.output_size,
        [1, 2],
        activation=activation,
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([1.0, 2.0]),
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation=activation,
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([3.0, 4.0]),
    )
    net.add_layer(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    y = input_layer.eval_single_layer(np.asarray([input_value]))
    y = dense_layer_0.eval_single_layer(y)
    y = dense_layer_1.eval_single_layer(y)

    return net, y


def _test_two_node_full_space_nn_formulation_smooth(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))
    assert m.nvariables() == FULLSPACE_SMOOTH_VARS
    assert m.nconstraints() == FULLSPACE_SMOOTH_CONSTRAINTS

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL


def _test_two_node_full_space_nn_formulation_relu():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network("relu", -2.0)
    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))
    assert m.nvariables() == FULLSPACE_RELU_VARS
    assert m.nconstraints() == FULLSPACE_RELU_CONSTRAINTS

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("cbc").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL

    net, y = two_node_network("relu", 1.0)
    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL


def _test_two_node_full_space_smooth_nn_formulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(FullSpaceSmoothNNFormulation(net))
    assert m.nvariables() == FULLSPACE_SMOOTH_VARS
    assert m.nconstraints() == FULLSPACE_SMOOTH_CONSTRAINTS

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL


def _test_two_node_reduced_space_nn_formulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(ReducedSpaceNNFormulation(net))
    assert m.nvariables() == REDUCED_VARS
    assert m.nconstraints() == REDUCED_CONSTRAINTS

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL


def _test_two_node_reduced_space_smooth_nn_formulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(ReducedSpaceSmoothNNFormulation(net))
    assert m.nvariables() == REDUCED_VARS
    assert m.nconstraints() == REDUCED_CONSTRAINTS

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < NEAR_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < NEAR_EQUAL


def test_two_node_reduced_space_nn_formulation():
    _test_two_node_reduced_space_nn_formulation("linear")
    _test_two_node_reduced_space_nn_formulation("sigmoid")
    _test_two_node_reduced_space_nn_formulation("tanh")


def test_two_node_reduced_space_smooth_nn_formulation():
    _test_two_node_reduced_space_smooth_nn_formulation("linear")
    _test_two_node_reduced_space_smooth_nn_formulation("sigmoid")
    _test_two_node_reduced_space_smooth_nn_formulation("tanh")


def test_two_node_reduced_space_smooth_nn_formulation_invalid_activation():
    expected_msg = "Activation relu is not supported by this formulation."
    with pytest.raises(ValueError, match=expected_msg):
        _test_two_node_reduced_space_smooth_nn_formulation("relu")


def test_two_node_full_space_nn_formulation():
    _test_two_node_full_space_nn_formulation_smooth("linear")
    _test_two_node_full_space_nn_formulation_smooth("sigmoid")
    _test_two_node_full_space_nn_formulation_smooth("tanh")
    _test_two_node_full_space_nn_formulation_relu()


def test_two_node_full_space_smooth_nn_formulation():
    _test_two_node_full_space_smooth_nn_formulation("linear")
    _test_two_node_full_space_smooth_nn_formulation("sigmoid")
    _test_two_node_full_space_smooth_nn_formulation("tanh")


def test_two_node_full_space_smooth_nn_formulation_invalid_activation():
    expected_msg = "Activation relu is not supported by this formulation."
    with pytest.raises(ValueError, match=expected_msg):
        _test_two_node_full_space_smooth_nn_formulation("relu")


@pytest.mark.skip(reason="Need to add checks on layer types")
def test_invalid_layer_type():
    msg = "Layer type test not yet implemented"
    raise AssertionError(msg)


def _maxpool_conv_network(inputs):
    input_size = [1, 8, 6]
    input_bounds = {}
    for i in range(input_size[1]):
        for j in range(input_size[2]):
            input_bounds[(0, i, j)] = (-10.0, 10.0)
    net = NetworkDefinition(scaled_input_bounds=input_bounds)

    input_layer = InputLayer(input_size)
    net.add_layer(input_layer)

    conv_layer_1_kernel = np.array([[[[-3, 0], [1, 5]]]])
    conv_layer_1 = ConvLayer2D(
        input_layer.output_size, [1, 4, 5], [2, 1], conv_layer_1_kernel
    )
    net.add_layer(conv_layer_1)
    net.add_edge(input_layer, conv_layer_1)

    # have two consecutive conv layers,
    # to check that conv layer behaves normally when a non-max pool layer succeeds it
    conv_layer_2_kernel = np.array([[[[-2, -2], [-2, -2]]]])
    conv_layer_2 = ConvLayer2D(
        conv_layer_1.output_size,
        [1, 3, 4],
        [1, 1],
        conv_layer_2_kernel,
        activation="relu",
    )
    net.add_layer(conv_layer_2)
    net.add_edge(conv_layer_1, conv_layer_2)

    # test normal ConvLayer -> MaxPoolLayer structure, with monotonic increasing
    # activation part of ConvLayer
    maxpool_layer_1 = PoolingLayer2D(
        conv_layer_2.output_size, [1, 1, 2], [2, 2], "max", [3, 2], 1
    )
    net.add_layer(maxpool_layer_1)
    net.add_edge(conv_layer_2, maxpool_layer_1)

    conv_layer_3_kernel = np.array([[[[4]]]])
    conv_layer_3 = ConvLayer2D(
        maxpool_layer_1.output_size, [1, 1, 2], [1, 1], conv_layer_3_kernel
    )
    net.add_layer(conv_layer_3)
    net.add_edge(maxpool_layer_1, conv_layer_3)

    # test ConvLayer -> MaxPoolLayer when nonlinear activation function is
    # already part of max pooling layer
    # also test index mapping logic in max pooling layers
    maxpool_layer_2_input_size = [1, 2, 1]
    maxpool_layer_2_index_mapper = IndexMapper(
        conv_layer_3.output_size, maxpool_layer_2_input_size
    )
    maxpool_layer_2 = PoolingLayer2D(
        maxpool_layer_2_input_size,
        [1, 1, 1],
        [1, 1],
        "max",
        [2, 1],
        1,
        input_index_mapper=maxpool_layer_2_index_mapper,
        activation="relu",
    )
    net.add_layer(maxpool_layer_2)
    net.add_edge(conv_layer_3, maxpool_layer_2)

    y = input_layer.eval_single_layer(inputs)
    y = conv_layer_1.eval_single_layer(y)
    y = conv_layer_2.eval_single_layer(y)
    y = maxpool_layer_1.eval_single_layer(y)
    y = conv_layer_3.eval_single_layer(y)
    y = maxpool_layer_2.eval_single_layer(y)

    return net, y


def test_maxpool_full_space_nn_formulation():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()

    inputs = np.array(
        [
            [
                [0, -2, -1, -1, 0, -2],
                [-2, 1, 10, 1, -1, -2],
                [-2, -9, 1, -2, -5, 0],
                [-1, 5, 5, -1, 1, 0],
                [4, -2, 10, -1, 1, -1],
                [-1, 1, -1, 0, 1, -1],
                [-10, 0, 0, -2, 7, -1],
                [0, 0, -1, -1, -1, -1],
            ]
        ]
    )

    net, y = _maxpool_conv_network(inputs)
    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))

    for inputs_d in range(inputs.shape[0]):
        for inputs_r in range(inputs.shape[1]):
            for inputs_c in range(inputs.shape[2]):
                m.neural_net_block.inputs[inputs_d, inputs_r, inputs_c].fixed = True
                m.neural_net_block.inputs[inputs_d, inputs_r, inputs_c].value = inputs[
                    inputs_d, inputs_r, inputs_c
                ]
    m.obj1 = pyo.Objective(expr=0)
    pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0, 0]) - y[0, 0, 0]) < NEAR_EQUAL


def _test_formulation_initialize_extra_input(network_formulation):
    """network_formulation can be:'FullSpace', 'ReducedSpace'."""
    net, y = two_node_network("linear", -2.0)
    extra_input = InputLayer([1])
    net.add_layer(extra_input)

    expected_msg = "Multiple input layers are not currently supported."
    if network_formulation == "FullSpace":
        with pytest.raises(ValueError, match=expected_msg):
            FullSpaceNNFormulation(net)
    elif network_formulation == "ReducedSpace":
        with pytest.raises(ValueError, match=expected_msg):
            ReducedSpaceNNFormulation(net)


def _test_formulation_added_extra_input(network_formulation):
    """network_formulation can be:'FullSpace', 'ReducedSpace', 'relu'."""
    net, y = two_node_network("linear", -2.0)
    extra_input = InputLayer([1])
    formulation: _PyomoFormulation = formulations[network_formulation](net)
    net.add_layer(extra_input)
    expected_msg = "Multiple input layers are not currently supported."
    with pytest.raises(ValueError, match=expected_msg):
        _ = formulation.input_indexes


def _test_formulation_build_extra_input(network_formulation):
    """network_formulation can be:'FullSpace', 'ReducedSpace', 'relu'."""
    net, y = two_node_network("linear", -2.0)
    extra_input = InputLayer([1])
    formulation: _PyomoFormulation = formulations[network_formulation](net)
    net.add_layer(extra_input)
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    expected_msg = "Multiple input layers are not currently supported."
    with pytest.raises(ValueError, match=expected_msg):
        m.neural_net_block.build_formulation(formulation)


def _test_formulation_added_extra_output(network_formulation):
    """network_formulation can be: 'FullSpace', 'ReducedSpace' 'relu'."""
    net, y = two_node_network("linear", -2.0)
    extra_output = DenseLayer(
        [1, 2],
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([3.0, 4.0]),
    )
    formulation: _PyomoFormulation = formulations[network_formulation](net)
    net.add_layer(extra_output)
    net.add_edge(list(net.layers)[-2], extra_output)
    expected_msg = "Multiple output layers are not currently supported."
    with pytest.raises(ValueError, match=expected_msg):
        _ = formulation.output_indexes


def _test_formulation_initialize_extra_output(network_formulation):
    """network_formulation can be: 'FullSpace', 'ReducedSpace'."""
    net, y = two_node_network("linear", -2.0)
    extra_output = DenseLayer(
        [1, 2],
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([3.0, 4.0]),
    )
    net.add_layer(extra_output)
    net.add_edge(list(net.layers)[-2], extra_output)

    expected_msg = "Multiple output layers are not currently supported."
    if network_formulation == "FullSpace":
        with pytest.raises(ValueError, match=expected_msg):
            FullSpaceNNFormulation(net)
    elif network_formulation == "ReducedSpace":
        with pytest.raises(ValueError, match=expected_msg):
            ReducedSpaceNNFormulation(net)


def test_full_space_nn_formulation_invalid_network():
    _test_formulation_initialize_extra_input("FullSpace")
    _test_formulation_added_extra_input("FullSpace")
    _test_formulation_build_extra_input("FullSpace")
    _test_formulation_initialize_extra_output("FullSpace")
    _test_formulation_added_extra_output("FullSpace")


def test_reduced_space_nn_formulation_invalid_network():
    _test_formulation_added_extra_input("ReducedSpace")
    _test_formulation_build_extra_input("ReducedSpace")
    _test_formulation_added_extra_output("ReducedSpace")


def test_relu_partition_formulation_invalid_network():
    _test_formulation_added_extra_input("relu")
    _test_formulation_build_extra_input("relu")
    _test_formulation_added_extra_output("relu")


def _test_dense_layer_multiple_predecessors(layer_type):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(None, -2.0)
    extra_input = InputLayer([1])
    test_layer = list(net.layers)[2]
    net.add_layer(extra_input)
    net.add_edge(extra_input, test_layer)

    expected_msg = re.escape(f"Layer {test_layer} has multiple predecessors.")
    if layer_type == "PartitionBased":
        with pytest.raises(ValueError, match=expected_msg):
            partition_based_dense_relu_layer(m, net, m, test_layer, None)
    elif layer_type == "ReducedSpace":
        with pytest.raises(ValueError, match=expected_msg):
            reduced_space_dense_layer(m, net, m, test_layer, None)


def _test_dense_layer_no_predecessors(layer_type):
    """Layer type can be "ReducedSpace", or "PartitionBased"."""
    m = pyo.ConcreteModel()
    net = NetworkDefinition(scaled_input_bounds=[(-10.0, 10.0)])

    test_layer = DenseLayer(
        [1],
        [1, 2],
        activation=None,
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([1.0, 2.0]),
    )
    net.add_layer(test_layer)

    expected_msg = re.escape(
        f"Layer {test_layer} is not an input layer, but has no predecessors."
    )
    if layer_type == "PartitionBased":
        with pytest.raises(ValueError, match=expected_msg):
            partition_based_dense_relu_layer(m, net, m, test_layer, None)
    elif layer_type == "ReducedSpace":
        with pytest.raises(ValueError, match=expected_msg):
            reduced_space_dense_layer(m, net, m, test_layer, None)


def test_partition_based_dense_layer_predecessors():
    _test_dense_layer_multiple_predecessors("PartitionBased")
    _test_dense_layer_no_predecessors("PartitionBased")


def test_reduced_space_dense_layer_predecessors():
    _test_dense_layer_multiple_predecessors("ReducedSpace")
    _test_dense_layer_no_predecessors("ReducedSpace")


def test_partition_based_unbounded_below():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(None, -2.0)
    test_layer = list(net.layers)[2]
    prev_layer_id = id(list(net.layers)[1])
    formulation = ReluPartitionFormulation(net)

    m.neural_net_block.build_formulation(formulation)
    prev_layer_block = m.neural_net_block.layer[prev_layer_id]
    prev_layer_block.z.setlb(-interval.inf)

    split_func = partial(default_partition_split_func, n=2)

    expected_msg = "Expression is unbounded below."
    with pytest.raises(ValueError, match=expected_msg):
        partition_based_dense_relu_layer(
            m.neural_net_block, net, m.neural_net_block, test_layer, split_func
        )


def test_partition_based_unbounded_above():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(None, -2.0)
    test_layer = list(net.layers)[2]
    prev_layer_id = id(list(net.layers)[1])
    formulation = ReluPartitionFormulation(net)

    m.neural_net_block.build_formulation(formulation)
    prev_layer_block = m.neural_net_block.layer[prev_layer_id]
    prev_layer_block.z.setub(interval.inf)

    split_func = partial(default_partition_split_func, n=2)

    expected_msg = "Expression is unbounded above."
    with pytest.raises(ValueError, match=expected_msg):
        partition_based_dense_relu_layer(
            m.neural_net_block, net, m.neural_net_block, test_layer, split_func
        )


def test_partition_based_bias_unbounded_below():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(None, -2.0)
    test_layer = list(net.layers)[2]
    formulation = ReluPartitionFormulation(net)

    m.neural_net_block.build_formulation(formulation)

    test_layer.biases[0] = -interval.inf
    split_func = partial(default_partition_split_func, n=2)

    expected_msg = "Expression is unbounded below."
    with pytest.raises(ValueError, match=expected_msg):
        partition_based_dense_relu_layer(
            m.neural_net_block, net, m.neural_net_block, test_layer, split_func
        )


def test_partition_based_bias_unbounded_above():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(None, -2.0)
    test_layer = list(net.layers)[2]
    formulation = ReluPartitionFormulation(net)

    m.neural_net_block.build_formulation(formulation)

    test_layer.biases[0] = interval.inf
    split_func = partial(default_partition_split_func, n=2)
    expected_msg = "Expression is unbounded above."
    with pytest.raises(ValueError, match=expected_msg):
        partition_based_dense_relu_layer(
            m.neural_net_block, net, m.neural_net_block, test_layer, split_func
        )


def test_fullspace_internal_extra_input():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network("linear", -2.0)
    extra_input = InputLayer([1])
    test_layer = list(net.layers)[1]
    formulation = FullSpaceNNFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    net.add_layer(extra_input)
    net.add_edge(extra_input, test_layer)
    expected_msg = "Multiple input layers are not currently supported."
    with pytest.raises(ValueError, match=expected_msg):
        _input_layer_and_block(m.neural_net_block, net, test_layer)


def test_conv2d_extra_activation():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()

    input_size = [1, 8, 6]
    input_bounds = {}
    for i in range(input_size[1]):
        for j in range(input_size[2]):
            input_bounds[(0, i, j)] = (-10.0, 10.0)
    net = NetworkDefinition(scaled_input_bounds=input_bounds)

    input_layer = InputLayer(input_size)
    net.add_layer(input_layer)

    conv_layer_1_kernel = np.array([[[[-3, 0], [1, 5]]]])
    conv_layer_1 = ConvLayer2D(
        input_layer.output_size, [1, 4, 5], [2, 1], conv_layer_1_kernel
    )
    net.add_layer(conv_layer_1)
    net.add_edge(input_layer, conv_layer_1)

    # have two consecutive conv layers,
    # to check that conv layer behaves normally when a non-max pool layer succeeds it
    conv_layer_2_kernel = np.array([[[[-2, -2], [-2, -2]]]])
    conv_layer_2 = ConvLayer2D(
        conv_layer_1.output_size,
        [1, 3, 4],
        [1, 1],
        conv_layer_2_kernel,
        activation="relu",
    )
    net.add_layer(conv_layer_2)
    net.add_edge(conv_layer_1, conv_layer_2)

    # test normal ConvLayer -> MaxPoolLayer structure, with monotonic
    # increasing activation part of ConvLayer
    maxpool_layer_1 = PoolingLayer2D(
        conv_layer_2.output_size, [1, 1, 2], [2, 2], "max", [3, 2], 1, activation="relu"
    )
    net.add_layer(maxpool_layer_1)
    net.add_edge(conv_layer_2, maxpool_layer_1)
    expected_msg = re.escape(
        "Activation is applied after convolution layer, but the successor maxpooling"
        " layer PoolingLayer(input_size=[1, 3, 4], output_size=[1, 1, 2],"
        " strides=[2, 2], kernel_shape=[3, 2]), pool_func_name=max has an activation"
        " function also."
    )
    with pytest.raises(ValueError, match=expected_msg):
        m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))


def test_maxpool2d_bad_input_activation():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()

    input_size = [1, 8, 6]
    input_bounds = {}
    for i in range(input_size[1]):
        for j in range(input_size[2]):
            input_bounds[(0, i, j)] = (-10.0, 10.0)
    net = NetworkDefinition(scaled_input_bounds=input_bounds)

    input_layer = InputLayer(input_size)
    net.add_layer(input_layer)

    conv_layer_1_kernel = np.array([[[[-3, 0], [1, 5]]]])
    conv_layer_1 = ConvLayer2D(
        input_layer.output_size, [1, 4, 5], [2, 1], conv_layer_1_kernel
    )
    net.add_layer(conv_layer_1)
    net.add_edge(input_layer, conv_layer_1)

    # have two consecutive conv layers,
    # to check that conv layer behaves normally when a non-max pool layer succeeds it
    conv_layer_2_kernel = np.array([[[[-2, -2], [-2, -2]]]])
    conv_layer_2 = ConvLayer2D(
        conv_layer_1.output_size,
        [1, 3, 4],
        [1, 1],
        conv_layer_2_kernel,
        activation="relu",
    )
    net.add_layer(conv_layer_2)
    net.add_edge(conv_layer_1, conv_layer_2)

    # test normal ConvLayer -> MaxPoolLayer structure, with monotonic increasing
    # activation part of ConvLayer
    maxpool_layer_1 = PoolingLayer2D(
        conv_layer_2.output_size,
        [1, 1, 2],
        [2, 2],
        "max",
        [3, 2],
        1,
        activation="linear",
    )
    net.add_layer(maxpool_layer_1)
    net.add_edge(conv_layer_2, maxpool_layer_1)

    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))

    conv_layer_2.activation = "relu"
    expected_msg = (
        "Non-increasing activation functions on the preceding convolutional"
        " layer are not supported."
    )
    with pytest.raises(ValueError, match=expected_msg):
        full_space_maxpool2d_layer(
            m.neural_net_block, net, m.neural_net_block, maxpool_layer_1
        )


def test_maxpool2d_bad_input_layer():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()

    input_size = [1, 8, 6]
    input_bounds = {}
    for i in range(input_size[1]):
        for j in range(input_size[2]):
            input_bounds[(0, i, j)] = (-10.0, 10.0)
    net = NetworkDefinition(scaled_input_bounds=input_bounds)

    input_layer = InputLayer(input_size)
    net.add_layer(input_layer)

    conv_layer_1_kernel = np.array([[[[-3, 0], [1, 5]]]])
    conv_layer_1 = ConvLayer2D(
        input_layer.output_size, [1, 4, 5], [2, 1], conv_layer_1_kernel
    )
    net.add_layer(conv_layer_1)
    net.add_edge(input_layer, conv_layer_1)

    # have two consecutive conv layers,
    # to check that conv layer behaves normally when a non-max pool layer succeeds it
    conv_layer_2_kernel = np.array([[[[-2, -2], [-2, -2]]]])
    conv_layer_2 = ConvLayer2D(
        conv_layer_1.output_size,
        [1, 3, 4],
        [1, 1],
        conv_layer_2_kernel,
        activation="relu",
    )
    net.add_layer(conv_layer_2)
    net.add_edge(conv_layer_1, conv_layer_2)

    # test normal ConvLayer -> MaxPoolLayer structure, with monotonic increasing
    # activation part of ConvLayer
    maxpool_layer_1 = PoolingLayer2D(
        conv_layer_2.output_size,
        [1, 1, 2],
        [2, 2],
        "max",
        [3, 2],
        1,
        activation="linear",
    )
    net.add_layer(maxpool_layer_1)
    net.add_edge(conv_layer_2, maxpool_layer_1)

    maxpool_layer_2 = PoolingLayer2D(
        maxpool_layer_1.output_size,
        [1, 1, 2],
        [2, 2],
        "max",
        [3, 2],
        1,
        activation="linear",
    )
    net.add_layer(maxpool_layer_2)
    net.add_edge(maxpool_layer_1, maxpool_layer_2)

    with pytest.raises(TypeError) as excinfo:
        m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))
    expected_msg = "Input layer must be a ConvLayer2D."
    assert str(excinfo.value) == expected_msg


def three_node_graph_neural_network(activation):
    input_size = [6]
    input_bounds = {}
    for i in range(input_size[0]):
        input_bounds[(i)] = (-10.0, 10.0)
    net = NetworkDefinition(scaled_input_bounds=input_bounds)

    input_layer = InputLayer(input_size)
    net.add_layer(input_layer)

    gnn_layer = GNNLayer(
        input_layer.output_size,
        [9],
        activation=activation,
        weights=np.array(
            [
                [1, 0, 1, 1, -1, 1, 1, -1, 1],
                [0, 1, 1, -1, 1, 1, -1, 1, 1],
                [1, -1, 1, 1, 0, 1, 1, -1, 1],
                [-1, 1, 1, 0, 1, 1, -1, 1, 1],
                [1, -1, 1, 1, -1, 1, 1, 0, 1],
                [-1, 1, 1, -1, 1, 1, 0, 1, 1],
            ]
        ),
        biases=np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1]),
        N=3,
    )
    net.add_layer(gnn_layer)
    net.add_edge(input_layer, gnn_layer)

    return net


def examples_of_graphs(graph_type):
    # complete graph
    if graph_type == "complete":
        A = np.ones([3, 3], dtype=int)
        y = np.array([-11, 9, 1, -12, 11, 1, -10, 10, 1])
    # edgeless graph
    elif graph_type == "edgeless":
        A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = np.array([-4, 2, 0, -2, 1, 1, -3, 3, 2])
    # line graph, i.e., 0-1-2
    elif graph_type == "line":
        A = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        y = np.array([-6, 4, 0, -12, 11, 1, -5, 5, 2])
    return A, y


def _test_three_node_graph_neural_network(graph_type):
    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    inputs = np.array([-3, 2, -1, 1, -2, 3])
    net = three_node_graph_neural_network("linear")

    m.nn.N = 3
    m.nn.A = pyo.Var(
        pyo.Set(initialize=range(m.nn.N)),
        pyo.Set(initialize=range(m.nn.N)),
        within=pyo.Binary,
    )

    m.nn.build_formulation(FullSpaceNNFormulation(net))

    A, y = examples_of_graphs(graph_type)
    for i in range(m.nn.N):
        for j in range(m.nn.N):
            m.nn.A[i, j].fix(A[i, j])
    for i in range(6):
        m.nn.inputs[i].fix(inputs[i])

    assert m.nvariables() == THREE_NODE_VARS
    assert m.nconstraints() == THREE_NODE_CONSTRAINTS

    m.obj = pyo.Objective(expr=0)

    pyo.SolverFactory("cbc").solve(m, tee=False)

    for i in range(9):
        assert abs(pyo.value(m.nn.outputs[i]) - y[i]) < NEAR_EQUAL

    for i in range(6):
        for j in range(3):
            assert (
                abs(
                    pyo.value(m.nn.layer[m.nn.layers.at(1)].zbar[i, j])
                    - pyo.value(m.nn.A[i // 2, j]) * inputs[i]
                )
                < NEAR_EQUAL
            )


def test_three_node_graph_neural_network():
    graph_types = ["complete", "edgeless", "line"]
    for graph_type in graph_types:
        _test_three_node_graph_neural_network(graph_type)
