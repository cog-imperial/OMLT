from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from skl2onnx import convert_sklearn
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import RobustScaler
from omlt.scaling import OffsetScaling
from omlt.io.onnx_reader import load_onnx_neural_network
import onnxruntime as rt
import numpy as np
import onnx

def parse_sklearn_scaler(sklearn_scaler):

    if isinstance(sklearn_scaler, StandardScaler):
        offset = sklearn_scaler.mean_
        factor = sklearn_scaler.scale_

    elif isinstance(sklearn_scaler, MaxAbsScaler):
        factor = sklearn_scaler.scale_
        offset = factor*0

    elif isinstance(sklearn_scaler, MinMaxScaler):
        factor = sklearn_scaler.data_max_ - sklearn_scaler.data_min_
        offset = sklearn_scaler.data_min_

    elif isinstance(sklearn_scaler, RobustScaler):
        factor = sklearn_scaler.scale_
        offset = sklearn_scaler.center_

    else:
        raise(ValueError("Scaling object provided is not currently supported. Only linear scalers are supported."
                         "Supported scalers include StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler"))

    return offset, factor

def convert_sklearn_scalers(sklearn_input_scaler, sklearn_output_scaler):

    #Todo: support only scaling input or output?

    offset_inputs, factor_inputs = parse_sklearn_scaler(sklearn_input_scaler)
    offset_outputs, factor_ouputs = parse_sklearn_scaler(sklearn_output_scaler)

    return OffsetScaling(offset_inputs=offset_inputs, factor_inputs=factor_inputs,
                         offset_outputs=offset_outputs, factor_outputs=factor_ouputs)

def load_sklearn_MLP(model, scaling_object=None, input_bounds=None, initial_types=None):

    # Assume float inputs if no types are supplied to the model
    if initial_types is None:
        initial_types = [('float_input', FloatTensorType([None, model.n_features_in_]))]

    onx = convert_sklearn(model, initial_types=initial_types, target_opset=12)

    x = np.array([[1.0] * 8])
    print(x)
    x = x.astype(np.float32)
    print(x.shape)
    print(model.predict(x))

    # REMOVE INITIAL CAST LAYER
    graph = onx.graph
    node1 = graph.node[0]
    graph.node.remove(node1)
    new_node = onnx.helper.make_node(
        'MatMul',
        name="MatMul",
        inputs=['float_input', 'coefficient'],
        outputs=['mul_result']
    )
    graph.node.insert(0, new_node)
    node2 = graph.node[1]

    graph.node.remove(node2)

    # onx_model = onx.SerializeToString()
    with open("test.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    sess = rt.InferenceSession("test.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    outputs = sess.run([label_name], {input_name: x})[0][0]

    print(outputs-model.predict(x))
    return load_onnx_neural_network(onx, scaling_object, input_bounds)
