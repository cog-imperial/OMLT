import onnx
import numpy as np

from omlt.io.onnx import load_onnx_neural_network


def test_linear_131(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(4):
        assert net.activations[i + 1] == "linear"


def test_linear_131_relu(datadir):
    model = onnx.load(datadir.file("keras_linear_131_relu.onnx"))
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(3):
        assert net.activations[i + 1] == "relu"


def test_linear_131_sigmoid(datadir):
    model = onnx.load(datadir.file("keras_linear_131_sigmoid.onnx"))
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(3):
        assert net.activations[i + 1] == "sigmoid"


def test_gemm(datadir):
    model = onnx.load(datadir.file("gemm.onnx"))
    net = load_onnx_neural_network(model)
    assert net.n_inputs == 784
    assert net.n_outputs == 10
    assert len(net.weights) == 160
    assert len(net.biases) == 160
    output_nodes = net.all_node_ids()[-net.n_outputs :]
    for i in output_nodes:
        assert net.activations[i] == "logsoftmax"


def test_conv(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    net = load_onnx_neural_network(model)
    assert net.n_inputs == 7*7
    assert net.n_outputs == 1
    assert len(net.weights) == 83
    assert len(net.biases) == 83

    # Check some value of conv layer output 
    # size = [2, 6, 6]
    # channel 0
    _compare_weights(net.weights[49+0], {0: -0.008866668, 1: 0.18750042, 7: -0.11404419, 8: -0.025886655})
    _compare_weights(net.weights[49+6], {7: -0.008866668, 14: -0.11404419, 8: 0.18750042, 15: -0.025886655})
    _compare_weights(net.weights[49+7], {8: -0.008866668, 15: -0.11404419, 9: 0.18750042, 16: -0.025886655})
    _compare_weights(net.weights[49+14], {16: -0.008866668, 23: -0.11404419, 17: 0.18750042, 24: -0.025886655})
    _compare_weights(net.weights[49+21], {24: -0.008866668, 31: -0.11404419, 25: 0.18750042, 32: -0.025886655})
    _compare_weights(net.weights[49+28], {32: -0.008866668, 39: -0.11404419, 33: 0.18750042, 40: -0.025886655})
    _compare_weights(net.weights[49+34], {39: -0.008866668, 46: -0.11404419, 40: 0.18750042, 47: -0.025886655})
    _compare_weights(net.weights[49+35], {40: -0.008866668, 47: -0.11404419, 41: 0.18750042, 48: -0.025886655})
    # channel 1
    _compare_weights(net.weights[49+36], {0: -0.075549066, 7: 0.2217437, 1: -0.059391618, 8: 0.14637864})
    _compare_weights(net.weights[49+43], {8: -0.075549066, 15: 0.2217437, 9: -0.059391618, 16: 0.14637864})
    _compare_weights(net.weights[49+50], {16: -0.075549066, 23: 0.2217437, 17: -0.059391618, 24:0.14637864})
    _compare_weights(net.weights[49+57], {24: -0.075549066, 31: 0.2217437, 25: -0.059391618, 32: 0.14637864})
    _compare_weights(net.weights[49+64], {32: -0.075549066, 39: 0.2217437, 33: -0.059391618, 40: 0.14637864})
    _compare_weights(net.weights[49+71], {40: -0.075549066, 47: 0.2217437, 41: -0.059391618, 48: 0.14637864})

    # First dense layer has 10 elements, connected to the 72 conv outputs
    dense_start = 49 + 71 + 1
    assert 72 == len(net.weights[dense_start])
    print(net.weights[dense_start])
    assert False


def _compare_weights(w0, w1):
    assert len(w0) == len(w1)
    assert set(w0.keys()) == set(w1.keys())
    for k, v in w0.items():
        assert np.isclose(v, w1[k])
