import skl2onnx
import onnx
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from onnx_reader import load_onnx_neural_network

def load_sklearn_MLP(model, scaling_object=None, input_bounds=None, initial_types=None):

    # Assume float inputs if no types are supplied to the model
    if initial_types is None:
        initial_types = [('float_input', FloatTensorType([None, model.n_features_in_]))]

    onx = convert_sklearn(model, initial_types=initial_types, target_opset=12)
    onx_model = onx.SerializeToString()

    return load_onnx_neural_network(onx_model, scaling_object, input_bounds)



X, y = make_regression(n_samples=200, n_features=42, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
print(regr)

initial_type = [('float_input', FloatTensorType([None, regr.n_features_in_]))]
onx = convert_sklearn(regr, initial_types=initial_type)
onx_model = onx.SerializeToString()

sess = rt.InferenceSession(onx_model)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0].transpose()[0]
pred_sklearn = regr.predict(X_test.astype(numpy.float32))
print(max(abs(pred_onx - pred_sklearn)/pred_sklearn*100))
