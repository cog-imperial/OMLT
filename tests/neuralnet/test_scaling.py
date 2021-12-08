from omlt.io.onnx import load_onnx_neural_network
import onnx
import onnxruntime as ort
import numpy as np
import pytest
from pyomo.environ import *

from omlt import OffsetScaling, OmltBlock
from omlt.neuralnet import NeuralNetworkFormulation


def test_offset_scaling():
    xdata = np.random.normal(1, 0.5, (2, 10))
    ydata = np.random.normal(-1, 0.2, (3, 10))
    x_offset = np.mean(xdata, axis=-1)
    x_factor = np.std(xdata, axis=-1)
    y_offset = np.mean(ydata, axis=-1)
    y_factor = np.std(ydata, axis=-1)

    # x = [1,2]
    # y = [-1,2,3]
    # x_scal = (np.asarray(x)-x_offset)/x_factor
    # y_scal = (np.asarray(y)-y_offset)/y_factor
    x = {0: 1, 1: 2}
    y = {0: -1, 1: 2, 2: 3}
    x_scal = {k: (v - x_offset[k]) / x_factor[k] for (k, v) in x.items()}
    y_scal = {k: (v - y_offset[k]) / y_factor[k] for (k, v) in y.items()}

    scaling = OffsetScaling(
        offset_inputs=x_offset,
        factor_inputs=x_factor,
        offset_outputs=y_offset,
        factor_outputs=y_factor,
    )

    test_x_scal = scaling.get_scaled_input_expressions(x)
    test_y_unscal = scaling.get_unscaled_output_expressions(y_scal)
    np.testing.assert_almost_equal(list(test_x_scal.values()), list(x_scal.values()))
    np.testing.assert_almost_equal(list(test_y_unscal.values()), list(y.values()))


@pytest.mark.skip("scaling object not updated")
def test_scaling_neural_network_block(datadir):
    neural_net = onnx.load(datadir.file("keras_linear_131_relu.onnx"))

    model = ConcreteModel()

    scale_x = (1, 0.5)
    scale_y = (-0.25, 0.125)

    scaler = OffsetScaling(
        offset_inputs=[scale_x[0]],
        factor_inputs=[scale_x[1]],
        offset_outputs=[scale_y[0]],
        factor_outputs=[scale_y[1]],
    )

    input_bounds = [
        (0, 5),
    ]
    net = load_onnx_neural_network(neural_net, scaler, input_bounds)
    formulation = NeuralNetworkFormulation(net)
    model.nn = OmltBlock()
    model.nn.build_formulation(formulation)

    @model.Objective()
    def obj(mdl):
        return 1

    net_regression = ort.InferenceSession(datadir.file("keras_linear_131_relu.onnx"))

    for x in np.random.normal(1, 0.5, 10):
        # model.input.fix(x)
        # result = SolverFactory("glpk").solve(model, tee=False)

        x_s = (x - scale_x[0]) / scale_x[1]
        x_s = np.array([[x_s]], dtype=np.float32)
        outputs = net_regression.run(None, {'dense_input:0': x_s})
        y_s = outputs[0][0, 0]
        y = y_s * scale_y[1] + scale_y[0]

        assert y - value(model.output) <= 1e-7
