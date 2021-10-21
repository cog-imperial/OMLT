class GradientBoostedTreeModel:
    def __init__(self, onnx_model):
        self.__model = onnx_model
        self.__n_inputs = _model_num_inputs(onnx_model)
        self.__n_outputs = _model_num_outputs(onnx_model)

    @property
    def onnx_model(self):
        return self.__model

    @property
    def n_inputs(self):
        return self.__n_inputs

    @property
    def n_hidden(self):
        return 0

    @property
    def n_outputs(self):
        return self.__n_outputs


def _model_num_inputs(model):
    graph = model.graph
    assert len(graph.input) == 1
    return _tensor_size(graph.input[0])


def _model_num_outputs(model):
    graph = model.graph
    assert len(graph.output) == 1
    return _tensor_size(graph.output[0])


def _tensor_size(tensor):
    tensor_type = tensor.type.tensor_type
    size = None
    for dim in tensor_type.shape.dim:
        if dim.dim_value is not None and dim.dim_value > 0:
            assert size is None
            size = dim.dim_value
    assert size is not None
    return size
