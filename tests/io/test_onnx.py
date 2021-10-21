from pathlib import Path

import onnx
import pytest

from omlt.io.onnx import load_onnx_neural_network


def test_linear_131():
    model = onnx.load(Path(__file__).parent / "models" / "keras_linear_131.onnx")
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(4):
        assert net.activations[i + 1] == "linear"


def test_linear_131_relu():
    model = onnx.load(Path(__file__).parent / "models" / "keras_linear_131_relu.onnx")
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(3):
        assert net.activations[i + 1] == "relu"


def test_linear_131_sigmoid():
    model = onnx.load(
        Path(__file__).parent / "models" / "keras_linear_131_sigmoid.onnx"
    )
    net = load_onnx_neural_network(model)
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    for i in range(3):
        assert net.activations[i + 1] == "sigmoid"


def test_gemm():
    model = onnx.load(Path(__file__).parent / "models" / "gemm.onnx")
    net = load_onnx_neural_network(model)
    assert net.n_inputs == 784
    assert net.n_outputs == 10
    assert len(net.weights) == 160
    assert len(net.biases) == 160
    output_nodes = net.all_node_ids()[-net.n_outputs :]
    for i in output_nodes:
        assert net.activations[i] == "logsoftmax"
