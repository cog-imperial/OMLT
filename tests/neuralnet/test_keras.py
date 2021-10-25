import os.path

import keras
import pyomo.environ as pyo
import pytest
import tensorflow
from pyomo.common.fileutils import this_file_dir

from omlt.block import OmltBlock
from omlt.neuralnet.full_space import FullSpaceContinuousFormulation
from omlt.neuralnet.keras_reader import load_keras_sequential
from omlt.neuralnet.reduced_space import ReducedSpaceContinuousFormulation
from omlt.neuralnet.relu import ReLUBigMFormulation, ReLUComplementarityFormulation
from omlt.scaling import OffsetScaling

from conftest import get_neural_network_data


def _test_keras_linear_131(keras_fname, reduced_space=False):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn,input_bounds = [(-1,1)])

    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    assert len(net.activations) == 4

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_mip_relu_131(keras_fname):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn,input_bounds = [(-1,1)])

    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    assert len(net.activations) == 4

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUBigMFormulation(net)
    m.neural_net_block.build_formulation(formulation)
    m.obj = pyo.Objective(expr=0)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("cbc").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_complementarity_relu_131(keras_fname):
    x, y, x_test = get_neural_network_data("131")

    nn = tensorflow.keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn)

    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    assert len(net.activations) == 4

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    formulation = ReLUComplementarityFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def _test_keras_linear_big(keras_fname, reduced_space=False):
    x, y, x_test = get_neural_network_data("131")

    nn = keras.models.load_model(keras_fname, compile=False)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory("ipopt").solve(m, tee=False)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5


def test_keras_linear_131_full():
    _test_keras_linear_131(os.path.join(this_file_dir(), "models/keras_linear_131"))
    _test_keras_linear_131(
        os.path.join(this_file_dir(), "models/keras_linear_131_sigmoid")
    )
    _test_keras_linear_131(
        os.path.join(
            this_file_dir(), "models/keras_linear_131_sigmoid_output_activation"
        )
    )
    _test_keras_linear_131(
        os.path.join(
            this_file_dir(),
            "models/keras_linear_131_sigmoid_softplus_output_activation",
        )
    )


def test_keras_linear_131_reduced():
    _test_keras_linear_131(
        os.path.join(this_file_dir(), "models/keras_linear_131"), reduced_space=True
    )
    _test_keras_linear_131(
        os.path.join(this_file_dir(), "models/keras_linear_131_sigmoid"),
        reduced_space=True,
    )
    _test_keras_linear_131(
        os.path.join(
            this_file_dir(), "models/keras_linear_131_sigmoid_output_activation"
        ),
        reduced_space=True,
    )
    _test_keras_linear_131(
        os.path.join(
            this_file_dir(),
            "models/keras_linear_131_sigmoid_softplus_output_activation",
        ),
        reduced_space=True,
    )


def test_keras_linear_131_relu():
    _test_keras_mip_relu_131(
        os.path.join(this_file_dir(), "models/keras_linear_131_relu")
    )
    _test_keras_complementarity_relu_131(
        os.path.join(this_file_dir(), "models/keras_linear_131_relu")
    )


def test_keras_linear_big():
    _test_keras_linear_big(
        os.path.join(this_file_dir(), "models/big"), reduced_space=False
    )
    # too slow
    # _test_keras_linear_big('./models/big', reduced_space=True)


# if __name__ == "__main__":
#     test_keras_linear_131_full()
#     test_keras_linear_big()
