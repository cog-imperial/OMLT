Quick-Start
============

The quick-start uses the same model shown on the OMLT `README <https://github.com/cog-imperial/OMLT/blob/main/README.rst>`_, but it provides a little more detail. The example 
imports a neural network trained in TensorFlow, formulates a Pyomo model, and seeks the neural network input that produces 
the desired output. 

We begin by importing the necessary packages. These include `tensorflow` to import the neural network and `pyomo` to 
build the optimization problem. We also import the necessary objects from `omlt` to formulate the neural network in Pyomo. :: 

    import tensorflow
    import pyomo.environ as pyo
    from omlt import OmltBlock, OffsetScaling
    from omlt.neuralnet import FullSpaceNNFormulation, NetworkDefinition
    from omlt.io import load_keras_sequential

We first load a simple neural network from the tests directory that contains 1 input, 1 output, and 3 hidden nodes 
with sigmoid activation functions. ::

    #load a Keras model
    nn = tensorflow.keras.models.load_model('tests/models/keras_linear_131_sigmoid', compile=False)

We next create a Pyomo model and attach an `OmltBlock` which will be used to formulate the neural network. An `OmltBlock` is a 
custom Pyomo block that we use to build machine learning model formulations. We also create Pyomo model variables to represent the 
input and output of the neural network. ::

    #create a Pyomo model with an OMLT block
    model = pyo.ConcreteModel()
    model.nn = OmltBlock()

    #the neural net contains one input and one output
    model.input = pyo.Var()
    model.output = pyo.Var()

OMLT supports the use of scaling and input bound information. This information informs how the Pyomo model 
applies scaling and unscaling to the neural network inputs and outputs. It also informs variable bounds on the inputs. :: 

    #apply simple offset scaling for the input and output
    scale_x = (1, 0.5)       #(mean,stdev) of the input
    scale_y = (-0.25, 0.125) #(mean,stdev) of the output
    scaler = OffsetScaling(offset_inputs=[scale_x[0]],
                        factor_inputs=[scale_x[1]],
                        offset_outputs=[scale_y[0]],
                        factor_outputs=[scale_y[1]])

    #provide bounds on the input variable (e.g. from training)
    scaled_input_bounds = {0:(0,5)}

We now create a `NetworkDefinition` using the `load\_keras\_sequential` function where we provide the 
scaler object and input bounds. Once we have a `NetworkDefinition`, we can pass it to various formulation objects which 
decide how to build the neural network within the `OmltBlock`. Here, we use the `FullSpaceNNFormulation`, but others are also possible 
(see :ref:`formulations <formulations>`). ::

    #load the keras model into a network definition
    net = load_keras_sequential(nn,scaler,scaled_input_bounds)

    #multiple formulations of a neural network are possible
    #this uses the default NeuralNetworkFormulation object
    formulation = FullSpaceNNFormulation(net)

    #build the formulation on the OMLT block
    model.nn.build_formulation(formulation)

We can query the input and output pyomo variables that the `build_formulation` method produces (as well as scaled input and output varialbes). 
We lastly create pyomo constraints that connect our input and output variables defined earlier to the neural network input and output variables on 
the `OmltBlock`.::

    #query inputs and outputs, as well as scaled inputs and outputs
    model.nn.inputs
    model.nn.outputs
    model.nn.scaled_inputs
    model.nn.scaled_outputs

    #connect pyomo model input and output to the neural network
    @model.Constraint()
    def connect_input(mdl):
        return mdl.input == mdl.nn.inputs[0]

    @model.Constraint()
    def connect_output(mdl):
        return mdl.output == mdl.nn.outputs[0]

Lastly, we formulate an objective function and use Ipopt to solve the optimization problem.::

    #solve an inverse problem to find that input that most closely matches the output value of 0.5
    model.obj = pyo.Objective(expr=(model.output - 0.5)**2)
    status = pyo.SolverFactory('ipopt').solve(model, tee=False)
    print(pyo.value(model.input))
    print(pyo.value(model.output))