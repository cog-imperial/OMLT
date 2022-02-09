from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from omlt.scaling import OffsetScaling

import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


from onnx_reader import load_onnx_neural_network

def get_sklearn_scaling_params(sklearn_scaler):

    if isinstance(sklearn_scaler, StandardScaler):
        offset = sklearn_scaler.mean_
        factor = sklearn_scaler.var_

    elif isinstance(sklearn_scaler, MaxAbsScaler):
        factor = sklearn_scaler.scale_
        offset = factor*0

    elif isinstance(sklearn_scaler, MinMaxScaler):
        factor = sklearn_scaler.data_max_ - sklearn_scaler.data_min_
        offset = sklearn_scaler.data_min_

    else:
        raise(ValueError("Scaling object provided is not currently supported. "
                         "Supported objects include StandardScaler, MinMaxScaler, and MaxAbsScaler"))

    return offset, factor

def parse_sklearn_scaler(sklearn_input_scaler, sklearn_output_scaler):

    offset_inputs, factor_inputs = get_sklearn_scaling_params(sklearn_input_scaler)
    offset_outputs, factor_ouputs = get_sklearn_scaling_params(sklearn_output_scaler)

    return OffsetScaling(offset_inputs=offset_inputs, factor_inputs=factor_inputs,
                         offset_outputs=offset_outputs, factor_outputs=factor_ouputs)


def load_sklearn_MLP(model, scaling_object=None, input_bounds=None, initial_types=None):

    # Assume float inputs if no types are supplied to the model
    if initial_types is None:
        initial_types = [('float_input', FloatTensorType([None, model.n_features_in_]))]

    onx = convert_sklearn(model, initial_types=initial_types, target_opset=12)
    onx_model = onx.SerializeToString()

    return load_onnx_neural_network(onx_model, scaling_object, input_bounds)
