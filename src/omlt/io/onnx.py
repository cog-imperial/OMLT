import json
from pathlib import Path

import onnx

from omlt.io.input_bounds import load_input_bounds, write_input_bounds
from omlt.io.onnx_parser import NetworkParser


def write_onnx_model_with_bounds(filename, onnx_model=None, input_bounds=None):
    """
    Write the ONNX model to the given file.

    If `input_bounds` is not None, write it alongside the ONNX model.

    Parameters
    ----------
    filename : str
        the path where the ONNX model is written
    onnx_model : onnx model or None
        the onnx model
    input_bounds : None or dict-like or list
        bounds on the input variables
    """
    if onnx_model is not None:
        with open(filename, "wb") as f:
            f.write(onnx_model.SerializeToString())

    if input_bounds is not None:
        write_input_bounds(f"{filename}.bounds.json", input_bounds)


def load_onnx_neural_network_with_bounds(filename):
    """
    Load a NetworkDefinition with input bounds from an onnx object.

    Parameters
    ----------
    filename : str
        the path where the ONNX model and input bounds file are written

    Returns
    -------
    NetworkDefinition
    """
    onnx_model = onnx.load(filename)
    input_bounds_filename = Path(f"{filename}.bounds.json")
    input_bounds = None
    if input_bounds_filename.exists:
        input_bounds = load_input_bounds(input_bounds_filename)

    return load_onnx_neural_network(onnx_model, input_bounds=input_bounds)


def load_onnx_neural_network(onnx, scaling_object=None, input_bounds=None):
    """
    Load a NetworkDefinition from an onnx object.

    Parameters
    ----------
    onnx :
        onnx model
    scaling_object : instance of object supporting ScalingInterface
    input_bounds : list of tuples

    Returns
    -------
    NetworkDefinition
    """
    parser = NetworkParser()
    return parser.parse_network(onnx.graph, scaling_object, input_bounds)
