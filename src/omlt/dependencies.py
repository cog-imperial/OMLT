from pyomo.common.dependencies import attempt_import

# check for onnx and create shortcut if available
onnx, onnx_available = attempt_import("onnx")

keras, keras_available = attempt_import("tensorflow.keras")
