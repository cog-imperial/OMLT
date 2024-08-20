import re

import numpy as np
import pyomo.environ as pyo
import pytest

from omlt import OmltBlock
from omlt.neuralnet.layer import DenseLayer, InputLayer
from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.nn_formulation import FullSpaceNNFormulation
from omlt.scaling import OffsetScaling

ALMOST_EXACTLY_EQUAL = 1e-8


def test_two_node_full_space():
    """Two node full space network.

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
        activation="relu",
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([0.0, 0.0]),
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        dense_layer_0.output_size,
        [1, 2],
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([0.0, 0.0]),
    )
    net.add_layer(dense_layer_1)
    net.add_edge(dense_layer_0, dense_layer_1)

    # verify that we are seeing the correct values
    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = FullSpaceNNFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    m.neural_net_block.inputs[0].fix(-2)
    m.obj1 = pyo.Objective(expr=0)
    status = pyo.SolverFactory("cbc").solve(m, tee=True)
    pyo.assert_optimal_termination(status)
    assert (
        abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 10.0) < ALMOST_EXACTLY_EQUAL
    )
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 2.0) < ALMOST_EXACTLY_EQUAL

    m.neural_net_block.inputs[0].fix(1)
    status = pyo.SolverFactory("cbc").solve(m, tee=False)
    pyo.assert_optimal_termination(status)
    assert abs(pyo.value(m.neural_net_block.outputs[0, 0]) - 1.0) < ALMOST_EXACTLY_EQUAL
    assert abs(pyo.value(m.neural_net_block.outputs[0, 1]) - 0.0) < ALMOST_EXACTLY_EQUAL


def test_input_bounds_no_scaler():
    scaled_input_bounds = {(0, 0): (0, 5), (0, 1): (-2, 2), (0, 2): (0, 1)}
    unscaled_input_bounds = scaled_input_bounds

    net = NetworkDefinition(unscaled_input_bounds=unscaled_input_bounds)
    assert net.scaled_input_bounds == scaled_input_bounds


def test_input_bound_scaling_1d():
    xoffset = {i: float(i) for i in range(3)}
    xfactor = {i: 0.5 * (i + 1) for i in range(3)}
    yoffset = {i: -0.25 * i for i in range(2)}
    yfactor = {i: 0.125 * (i + 1) for i in range(2)}

    scaler = OffsetScaling(
        offset_inputs=xoffset,
        factor_inputs=xfactor,
        offset_outputs=yoffset,
        factor_outputs=yfactor,
    )

    scaled_input_bounds = {0: (0, 5), 1: (-2, 2), 2: (0, 1)}
    unscaled_input_bounds = {}

    for k in scaled_input_bounds:
        lb, ub = scaled_input_bounds[k]
        unscaled_input_bounds[k] = (
            (lb * xfactor[k]) + xoffset[k],
            (ub * xfactor[k]) + xoffset[k],
        )

    net = NetworkDefinition(
        scaler, scaled_input_bounds=None, unscaled_input_bounds=unscaled_input_bounds
    )
    assert net.scaled_input_bounds == scaled_input_bounds


def test_input_bound_scaling_multi_d():
    # Multidimensional test
    xoffset = {(0, i): float(i) for i in range(3)}
    xfactor = {(0, i): 0.5 * (i + 1) for i in range(3)}
    yoffset = {(1, i): -0.25 * i for i in range(2)}
    yfactor = {(1, i): 0.125 * (i + 1) for i in range(2)}

    scaler = OffsetScaling(
        offset_inputs=xoffset,
        factor_inputs=xfactor,
        offset_outputs=yoffset,
        factor_outputs=yfactor,
    )

    scaled_input_bounds = {(0, 0): (0, 5), (0, 1): (-2, 2), (0, 2): (0, 1)}
    unscaled_input_bounds = {}

    for k in scaled_input_bounds:
        lb, ub = scaled_input_bounds[k]
        unscaled_input_bounds[k] = (
            (lb * xfactor[k]) + xoffset[k],
            (ub * xfactor[k]) + xoffset[k],
        )

    net = NetworkDefinition(
        scaler, scaled_input_bounds=None, unscaled_input_bounds=unscaled_input_bounds
    )
    assert net.scaled_input_bounds == scaled_input_bounds


def _test_add_invalid_edge(direction):
    """Direction can be "in" or "out"."""
    net = NetworkDefinition(scaled_input_bounds=[(-10.0, 10.0)])

    input_layer = InputLayer([1])
    net.add_layer(input_layer)

    dense_layer_0 = DenseLayer(
        input_layer.output_size,
        [1, 2],
        activation="relu",
        weights=np.array([[1.0, -1.0]]),
        biases=np.array([0.0, 0.0]),
    )
    net.add_layer(dense_layer_0)
    net.add_edge(input_layer, dense_layer_0)

    dense_layer_1 = DenseLayer(
        input_layer.output_size,
        dense_layer_0.input_size,
        activation="linear",
        weights=np.array([[1.0, 0.0], [5.0, 1.0]]),
        biases=np.array([0.0, 0.0]),
    )

    if direction == "in":
        expected_msg = re.escape(
            "Inbound layer DenseLayer(input_size=[1], output_size=[1]) not"
            " found in network."
        )
        with pytest.raises(ValueError, match=expected_msg):
            net.add_edge(input_layer, dense_layer_1)
    elif direction == "out":
        expected_msg = re.escape(
            "Outbound layer DenseLayer(input_size=[1], output_size=[1]) not"
            " found in network."
        )
        with pytest.raises(ValueError, match=expected_msg):
            net.add_edge(dense_layer_1, dense_layer_0)


def test_add_invalid_edge():
    _test_add_invalid_edge("in")
    _test_add_invalid_edge("out")
