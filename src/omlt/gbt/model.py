class GradientBoostedTreeModel:
    def __init__(self, onnx_model, scaling_object=None, scaled_input_bounds=None):
        self.__model = onnx_model
        self.__n_inputs = _model_num_inputs(onnx_model)
        self.__n_outputs = _model_num_outputs(onnx_model)
        self.__scaling_object = scaling_object
        self.__scaled_input_bounds = scaled_input_bounds

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

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.__scaling_object

    @property
    def scaled_input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.__scaled_input_bounds

    @scaling_object.setter
    def scaling_object(self, scaling_object):
        self.__scaling_object = scaling_object


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
