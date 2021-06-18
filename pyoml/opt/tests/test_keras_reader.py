import pytest
import numpy as np
import pyomo.environ as pyo

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adamax 

from pyoml.opt.neuralnet import NeuralNetworkBlock, \
    FullSpaceContinuousFormulation, ReducedSpaceContinuousFormulation
from pyoml.opt.scaling import OffsetScaling
from pyoml.opt.keras_reader import load_keras_sequential

"""
def test_keras_reader():
    #x_scaled = (x-offset)*multiplier
#    net = load_keras_sequential(keras_model_cop, scaling_offset=asdf, scaling_multiplier=asdf)
    net = load_keras_sequential(keras_model)

    m = pyo.ConcreteModel()
    m.nn = NeuralNetworkBlock()
    m.nn.build_formulation(
        FullSpaceContinuousFormulation(net)
    )

    # continue with the rest of your model
"""

def get_data(desc):
    if desc == '131':
        # build data with 1 input and 1 output and 500 data points
        x = np.random.uniform(-1, 1, 500)
        y = np.sin(x)
        x_test = np.random.uniform(-1,1,5)
        return x,y,x_test

    elif desc == '2353':
        # build data with 2 inputs, 3 outputs, and 500 data points
        np.random.seed(42)
        x = np.random.uniform([-1,2], [1,3], (500,2))
        y1 = np.sin(x[:,0]*x[:,1])
        y2 = x[:,0]+x[:,1]
        y3 = np.cos(x[:,0]/x[:,1])
        y = np.column_stack((y1,y2,y3))
        x_test = np.random.uniform([-1,2], [1,3], (5,2))
        return x,y,x_test

    return None

def train_models():
    x,y,x_test = get_data('131')
    nn = Sequential(name='keras_linear_131')
    nn.add(Dense(units=3, input_dim=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43)))
    nn.add(Dense(units=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63)))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)
    nn.save('models/keras_linear_131')
    
    x,y,x_test = get_data('131')
    nn = Sequential(name='keras_linear_131_sigmoid')
    nn.add(Dense(units=3, input_dim=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63)))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)
    nn.save('models/keras_linear_131_sigmoid')

    x,y,x_test = get_data('131')
    nn = Sequential(name='keras_linear_131_sigmoid_output_activation')
    nn.add(Dense(units=3, input_dim=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63),
                 activation='sigmoid'))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)
    nn.save('models/keras_linear_131_sigmoid_output_activation')

    x,y,x_test = get_data('131')
    nn = Sequential(name='keras_linear_131_sigmoid_softplus_output_activation')
    nn.add(Dense(units=3, input_dim=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63),
                 activation='softplus'))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)
    nn.save('models/keras_linear_131_sigmoid_softplus_output_activation')

    x,y,x_test = get_data('131')
    nn = Sequential(name='keras_big')
    N = 100
    nn.add(Dense(units=N, input_dim=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=N, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=N, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43),
                 activation='sigmoid'))
    nn.add(Dense(units=1, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63),
                 activation='softplus'))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)
    nn.save('models/big')

    x,y,x_test = get_data('2353')    
    nn = Sequential(name='keras_linear_2353')
    nn.add(Dense(units=3, input_dim=2, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=42),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=43)))
    nn.add(Dense(units=5, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=52),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=53)))
    nn.add(Dense(units=3, \
                 kernel_initializer=keras.initializers.RandomNormal(mean=1.0,stddev=0.05,seed=62),
                 bias_initializer=keras.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=63)))
    nn.compile(optimizer=Adamax(learning_rate=0.01), loss='mae')
    history = nn.fit(x=x, y=y, validation_split=0.2,
                     batch_size=16, verbose=1, epochs=15)

    nn.save('models/keras_linear_2353')

def _test_keras_linear_131(keras_fname, reduced_space=False):
    x,y,x_test = get_data('131')
    xo,xs,yo,ys = foo()

    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)
    net.set_scaling_object(OffsetScaling(xo, xs, yo, ys))

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetworkBlock()
    m.neural_net_block.build_formulation(
        ReducedSpaceContinuousFormulation(net)
    )



    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)

    assert net.n_inputs == 1
    assert net.n_outputs == 1
    assert net.n_hidden == 3
    assert len(net.weights) == 4
    assert len(net.biases) == 4
    assert len(net.activations) == 4

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetworkBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory('ipopt').solve(m, tee=True)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5

def _test_keras_linear_big(keras_fname, reduced_space=False):
    x,y,x_test = get_data('131')
    
    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetworkBlock()
    if reduced_space:
        formulation = ReducedSpaceContinuousFormulation(net)
    else:
        formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

    nn_outputs = nn.predict(x=x_test)
    for d in range(len(x_test)):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory('ipopt').solve(m, tee=True)
        pyo.assert_optimal_termination(status)
        assert abs(pyo.value(m.neural_net_block.outputs[0]) - nn_outputs[d][0]) < 1e-5

def test_keras_linear_131():
    _test_keras_linear_131('./models/keras_linear_131')
    _test_keras_linear_131('./models/keras_linear_131_sigmoid')
    _test_keras_linear_131('./models/keras_linear_131_sigmoid_output_activation')
    _test_keras_linear_131('./models/keras_linear_131_sigmoid_softplus_output_activation')
    _test_keras_linear_131('./models/keras_linear_131', reduced_space=True)
    _test_keras_linear_131('./models/keras_linear_131_sigmoid', reduced_space=True)
    _test_keras_linear_131('./models/keras_linear_131_sigmoid_output_activation', reduced_space=True)
    _test_keras_linear_131('./models/keras_linear_131_sigmoid_softplus_output_activation', reduced_space=True)

def xtest_keras_linear_2353():
    d11, d23 = get_data()

    x = d11['x']
    y = d11['y']
    x_test = d11['x_test']
    
    nn = keras.models.load_model("./models/three_layer_linear_keras")
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetworkBlock()
    formulation = FullSpaceContinuousFormulation(net)
    m.neural_net_block.build_formulation(formulation)

#    n_data, n_idx = x_test.shape
#    for d in range(n_data):
#        for i in n_idx:
#            m.neural_network_block.inputs[i].fix(x_test[d][i])

    nn_outputs = nn.predict(x=x_test)
    
    n_data = len(x_test)
    for d in range(n_data):
        m.neural_net_block.inputs[0].fix(x_test[d])
        status = pyo.SolverFactory('ipopt').solve(m, tee=True)
        pyo.assert_optimal_termination(status)

    assert False

if __name__ == '__main__':
    train_models()
    _test_keras_linear_big('./models/big', reduced_space=False)
    
