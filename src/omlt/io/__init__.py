from pyomo.common.dependencies import attempt_import

# check for onnx and create shortcut if available
onnx, onnx_available = attempt_import("onnx")
if onnx_available:
    from omlt.io.onnx import (
        load_onnx_neural_network,
        load_onnx_neural_network_with_bounds,
        write_onnx_model_with_bounds,
    )

# check for keras and create shortcut if available
keras, keras_available = attempt_import("tensorflow.keras")
if keras_available:
    from omlt.io.keras import load_keras_sequential
