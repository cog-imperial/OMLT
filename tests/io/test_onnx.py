import pytest
from pathlib import Path
import onnx
from optml.io.onnx import load_onnx_neural_network

def test_linear_131():
    model = onnx.load(Path(__file__).parent / 'keras_linear_131.onnx')

    net = load_onnx_neural_network(model)
    print(net)
