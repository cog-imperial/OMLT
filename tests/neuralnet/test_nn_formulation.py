import numpy as np
import pyomo.environ as pyo
import pytest

from omlt import OmltBlock
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    NetworkDefinition,
    ReducedSpaceNNFormulation,
    ReducedSpaceSmoothNNFormulation,
)
from omlt.neuralnet.layer import (
    ConvLayer2D,
    DenseLayer,
    IndexMapper,
    InputLayer,
    PoolingLayer2D,
    GNNLayer,
)


def two_node_network(activation, input_value):
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


def _test_two_node_FullSpaceNNFormulation_smooth(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))
    assert m.nvariables() == 15
    assert m.nconstraints() == 14

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6


def _test_two_node_FullSpaceNNFormulation_relu():
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network("relu", -2.0)
    m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))
    assert m.nvariables() == 19
    assert m.nconstraints() == 26

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6

    net, y = two_node_network("relu", 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6


def _test_two_node_FullSpaceSmoothNNFormulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(FullSpaceSmoothNNFormulation(net))
    assert m.nvariables() == 15
    assert m.nconstraints() == 14

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6


def _test_two_node_ReducedSpaceNNFormulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(ReducedSpaceNNFormulation(net))
    assert m.nvariables() == 6
    assert m.nconstraints() == 5

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6


def _test_two_node_ReducedSpaceSmoothNNFormulation(activation):
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    net, y = two_node_network(activation, -2.0)
    m.neural_net_block.build_formulation(ReducedSpaceSmoothNNFormulation(net))
    assert m.nvariables() == 6
    assert m.nconstraints() == 5

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)

    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6

    net, y = two_node_network(activation, 1.0)
    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("ipopt").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - y[0, 0]) < 1e-6
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - y[0, 1]) < 1e-6


def test_two_node_ReducedSpaceNNFormulation():
    _test_two_node_ReducedSpaceNNFormulation("linear")
    _test_two_node_ReducedSpaceNNFormulation("sigmoid")
    _test_two_node_ReducedSpaceNNFormulation("tanh")


def test_two_node_ReducedSpaceSmoothNNFormulation():
    _test_two_node_ReducedSpaceSmoothNNFormulation("linear")
    _test_two_node_ReducedSpaceSmoothNNFormulation("sigmoid")
    _test_two_node_ReducedSpaceSmoothNNFormulation("tanh")


def test_two_node_ReducedSpaceSmoothNNFormulation_invalid_activation():
    with pytest.raises(ValueError) as excinfo:
        _test_two_node_ReducedSpaceSmoothNNFormulation("relu")
    expected_msg = "Activation relu is not supported by this formulation."
    assert str(excinfo.value) == expected_msg


def test_two_node_FullSpaceNNFormulation():
    _test_two_node_FullSpaceNNFormulation_smooth("linear")
    _test_two_node_FullSpaceNNFormulation_smooth("sigmoid")
    _test_two_node_FullSpaceNNFormulation_smooth("tanh")
    _test_two_node_FullSpaceNNFormulation_relu()


def test_two_node_FullSpaceSmoothNNFormulation():
    _test_two_node_FullSpaceSmoothNNFormulation("linear")
    _test_two_node_FullSpaceSmoothNNFormulation("sigmoid")
    _test_two_node_FullSpaceSmoothNNFormulation("tanh")


def test_two_node_FullSpaceSmoothNNFormulation_invalid_activation():
    with pytest.raises(ValueError) as excinfo:
        _test_two_node_FullSpaceSmoothNNFormulation("relu")
    expected_msg = "Activation relu is not supported by this formulation."
    assert str(excinfo.value) == expected_msg


@pytest.mark.skip(reason="Need to add checks on layer types")
def test_invalid_layer_type():
    raise AssertionError("Layer type test not yet implemented")


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

    # test normal ConvLayer -> MaxPoolLayer structure, with monotonic increasing activation part of ConvLayer
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

    # test ConvLayer -> MaxPoolLayer when nonlinear activation function is already part of max pooling layer
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


def test_maxpool_FullSpaceNNFormulation():
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
    # assert m.nvariables() == 15
    # assert m.nconstraints() == 14

    for inputs_d in range(inputs.shape[0]):
        for inputs_r in range(inputs.shape[1]):
            for inputs_c in range(inputs.shape[2]):
                m.neural_net_block.inputs[inputs_d, inputs_r, inputs_c].fixed = True
                m.neural_net_block.inputs[inputs_d, inputs_r, inputs_c].value = inputs[
                    inputs_d, inputs_r, inputs_c
                ]
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0, 0]) - y[0, 0, 0]) < 1e-6


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

    assert m.nvariables() == 81
    assert m.nconstraints() == 120

    m.obj = pyo.Objective(expr=0)

    status = pyo.SolverFactory("cbc").solve(m, tee=False)

    for i in range(9):
        assert abs(pyo.value(m.nn.outputs[i]) - y[i]) < 1e-6

    for i in range(6):
        for j in range(3):
            assert (
                abs(
                    pyo.value(m.nn.layer[m.nn.layers.at(1)].zbar[i, j])
                    - pyo.value(m.nn.A[i // 2, j]) * inputs[i]
                )
                < 1e-6
            )


def test_three_node_graph_neural_network():
    graph_types = ["complete", "edgeless", "line"]
    for graph_type in graph_types:
        _test_three_node_graph_neural_network(graph_type)
