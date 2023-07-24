from pyomo.common.dependencies import attempt_import

# check for dependencies and create shortcut if available
onnx, onnx_available = attempt_import("onnx")
keras, keras_available = attempt_import("tensorflow.keras")
lightgbm, lightgbm_available = attempt_import("lightgbm")
