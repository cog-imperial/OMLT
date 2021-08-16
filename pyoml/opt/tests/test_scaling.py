import pytest
import numpy as np
from pyoml.opt.scaling import OffsetScaling
from pyoml.opt.neuralnet import ReLUBigMFormulation, NeuralNetworkBlock
from pyoml.opt.keras_reader import load_keras_sequential
from pyomo.environ import *
import tensorflow as tf

def test_offset_scaling():
    xdata = np.random.normal(1,0.5,(2,10))
    ydata = np.random.normal(-1,0.2,(3,10))
    x_offset = np.mean(xdata, axis=-1)
    x_factor = np.std(xdata, axis=-1)
    y_offset = np.mean(ydata, axis=-1)
    y_factor = np.std(ydata, axis=-1)

    x = [1,2]
    y = [-1,2,3]
    x_scal = (np.asarray(x)-x_offset)/x_factor
    y_scal = (np.asarray(y)-y_offset)/y_factor

    scaling = OffsetScaling(offset_inputs=x_offset,
                            factor_inputs=x_factor,
                            offset_outputs=y_offset,
                            factor_outputs=y_factor)

    test_x_scal = scaling.get_scaled_input_expressions(x)
    test_y_unscal = scaling.get_unscaled_output_expressions(y_scal)
    np.testing.assert_almost_equal(test_x_scal, x_scal)
    np.testing.assert_almost_equal(test_y_unscal, y)

def test_scaling_NN_block():
    NN = tf.keras.models.load_model("./models/keras_linear_131_relu")

    model = ConcreteModel()
    model.input = Var()
    model.output = Var()

    scale_x = (1, -0.5)
    scale_y = (-0.25, 0.125)

    scaler = OffsetScaling(offset_inputs=[scale_x[0]],
                            factor_inputs=[scale_x[1]],
                            offset_outputs=[scale_y[0]],
                            factor_outputs=[scale_y[1]])

    net = load_keras_sequential(NN)
    net.scaling_object = scaler
    formulation = ReLUBigMFormulation(net)

    model.nn = NeuralNetworkBlock()
    model.nn.build_formulation(formulation, input_vars=[model.input], output_vars=[model.output])

    @model.Objective()
    def obj(mdl):
        return 1

    for x in np.random.normal(1,0.5,10):
        model.input.fix(x)
        result = SolverFactory('gurobi_direct').solve(model, tee=False)

        x_s = (x - scale_x[0])/scale_x[1]
        y_s = NN.predict(x=[x_s])
        y = y_s * scale_y[1] + scale_y[0]

        assert(y - value(model.output) <= 1e-7)
