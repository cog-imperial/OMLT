# OptML
OptML is a Python package for representing machine learning models (such as neural networks) within the Pyomo optimization environment. The package provides various formulations for representing machine-learning models (such as full-space, reduced-space, and MILP), as well as an interface to import sequential Keras models.

# OptML development has moved:
### Current development of this package takes place at: https://github.com/cog-imperial/OptML

# Simple Example:
```python
import tensorflow 
import pyomo.environ as pyo
from optml import OptMLBlock, OffsetScaling
from optml.neuralnet import FullSpaceContinuousFormulation, ReducedSpaceContinuousFormulation
from optml.neuralnet import ReLUBigMFormulation
from optml.neuralnet import load_keras_sequential

#load a Keras model
nn = tensorflow.keras.models.load_model('optml/tests/models/keras_linear_131_sigmoid',compile = False)

#create a Pyomo model with an OptML block
model = pyo.ConcreteModel()
model.nn = OptMLBlock()

#the neural net contains one input and one output
model.input = pyo.Var()
model.output = pyo.Var()

#apply simple offset scaling for the input and output
scale_x = (1, 0.5)       #(mean,stdev) of the input
scale_y = (-0.25, 0.125) #(mean,stdev) of the output
scaler = OffsetScaling(offset_inputs=[scale_x[0]],
                    factor_inputs=[scale_x[1]],
                    offset_outputs=[scale_y[0]],
                    factor_outputs=[scale_y[1]])

#provide bounds on the input variable (e.g. from training)
input_bounds = [(0,5),]

#load the keras model into a network definition
net = load_keras_sequential(nn,scaler,input_bounds)

#multiple neural network formulations are possible
#hides the intermediate variables from the optimizer
formulation = ReducedSpaceContinuousFormulation(net)

#encodes intermediate neural network variables
#formulation = FullSpaceContinuousFormulation(net)

#encodes intermediate relu activations using binary variables
#this requires a neural network with only linear and relu activations
#formulation = ReLUBigMFormulation(net)

#build the formulation on the OptML block
model.nn.build_formulation(formulation, input_vars=[model.input], output_vars=[model.output])

#query inputs and outputs, as well as scaled inputs and outputs 
model.nn.inputs_list
model.nn.outputs_list 
model.nn.scaled_inputs_list 
model.nn.scaled_outputs_list

#solve an inverse problem to find that input that most closely matches the output value of 0.5
model.obj = pyo.Objective(expr=(model.output - 0.5)**2)
status = pyo.SolverFactory('ipopt').solve(model, tee=False)
print(pyo.value(model.input))
print(pyo.value(model.output))
```
