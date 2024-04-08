from pyomo.common.dependencies import attempt_import

# check for dependencies and create shortcut if available
onnx, onnx_available = attempt_import("onnx")
keras, keras_available = attempt_import("tensorflow.keras")

torch, torch_available = attempt_import("torch")

torch_geometric, torch_geometric_available = attempt_import("torch_geometric")
lineartree, lineartree_available = attempt_import("lineartree")

julia, julia_available = attempt_import("juliacall")

if julia_available:
    from juliacall import Main as jl
    try:
        jl.seval("import MathOptInterface")
        moi_available = True
    except jl.ArgumentError:
        moi_available = False
