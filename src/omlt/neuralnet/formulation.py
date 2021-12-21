import pyomo.environ as pyo

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs
from omlt.neuralnet.layer import ConvLayer, DenseLayer, InputLayer
from omlt.neuralnet.layers.full_space import (full_space_dense_layer, full_space_conv_layer,
                                              reduced_space_dense_layer)
from omlt.neuralnet.activations import (linear_activation_constraint,
                                        linear_activation_function,
                                        bigm_relu_activation_constraint)
from omlt.neuralnet.activations.smooth import (sigmoid_activation_constraint,
                                               softplus_activation_constraint,
                                               sigmoid_activation_function,
                                               softplus_activation_function)

def _ignore_input_layer():
    pass

# TODO: Doesn't this imply that there is a mapping from layer type to formulation
# There are different formulations possible for the same layer
_DEFAULT_LAYER_CONSTRAINTS = {
    InputLayer: _ignore_input_layer,
    DenseLayer: full_space_dense_layer,
    ConvLayer: full_space_conv_layer,
}

_DEFAULT_ACTIVATION_CONSTRAINTS = {
    "linear": linear_activation_constraint,
    "relu": bigm_relu_activation_constraint,
    "sigmoid": sigmoid_activation_constraint,
    "softplus": softplus_activation_constraint
}

_DEFAULT_ACTIVATION_FUNCTIONS = {
    "linear": linear_activation_function,
#    "relu": bigm_relu_activation,
    "sigmoid": sigmoid_activation_function,
    "softplus": softplus_activation_function
}

class NeuralNetworkFormulation(_PyomoFormulation):
    """
    This class is the entry-point to build neural network formulations.

    This class iterates over all nodes in the neural network and for
    each one them, generates the constraints to represent the layer
    and its activation function.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    layer_constraints : dict-like or None
        overrides the constraints generated for the specified layer types
    activation_constraints : dict-like or None
        overrides the constraints generated for the specified activation functions
    """
    def __init__(self, network_structure, layer_constraints=None, activation_constraints=None):
        super().__init__()

        self.__network_definition = network_structure
        self.__scaling_object = network_structure.scaling_object
        self.__scaled_input_bounds = network_structure.scaled_input_bounds
        
        if layer_constraints is None:
            layer_constraints = dict()
        if activation_constraints is None:
            activation_constraints = dict()

        self._layer_constraints = {**_DEFAULT_LAYER_CONSTRAINTS, **layer_constraints}
        self._activation_constraints = {**_DEFAULT_ACTIVATION_CONSTRAINTS, **activation_constraints}

    def _build_formulation(self):
        _setup_scaled_inputs_outputs(self.block,
                                     self.__scaling_object,
                                     self.__scaled_input_bounds)
        
        _build_neural_network_formulation(
            block=self.block,
            network_structure=self.__network_definition,
            layer_constraints=self._layer_constraints,
            activation_constraints=self._activation_constraints,
        )

    # TODO: are these properties used anywhere?
    # @property
    # def layer_constraints(self):
    #     return self._layer_constraints

    # @property
    # def activation_constraints(self):
    #     return self._activation_constraints

    # TODO: asserts to exceptions
    # TODO: push these to the formulation (remove dependency in block.py)
    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Unsupported multiple network input variables'
        return network_inputs[0].input_indexes

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Unsupported multiple network output variables'
        return network_outputs[0].output_indexes


def _build_neural_network_formulation(block, network_structure, layer_constraints, activation_constraints):
    """
    Adds the neural network formulation to the given Pyomo block.

    Parameters
    ----------
    block : Block
        the Pyomo block
    network_structure : NetworkDefinition
        the neural network definition
    layer_constraints : dict-like or None
        the constraints generated for the specified layer types
    activation_constraints : dict-like or None
        the constraints generated for the specified activation functions
    """
    net = network_structure
    layers = list(net.layers)

    block.layers = pyo.Set(initialize=[id(layer) for layer in layers], ordered=True)

    # create the z and z_hat variables for each of the layers
    @block.Block(block.layers)
    def layer(b, layer_id):
        net_layer = net.layer(layer_id)
        b.z = pyo.Var(net_layer.output_indexes, initialize=0)
        if isinstance(net_layer, InputLayer):
            for index in net_layer.output_indexes:
                input_var = block.scaled_inputs[index]
                z_var = b.z[index]
                z_var.setlb(input_var.lb)
                z_var.setub(input_var.ub)
        else:
            # add zhat only to non input layers
            b.zhat = pyo.Var(net_layer.output_indexes, initialize=0)

        return b

    for layer in layers:
        if isinstance(layer, InputLayer):
            continue
        layer_id = id(layer)
        layer_block = block.layer[layer_id]
        constraint_func = layer_constraints[type(layer)]
        activation_func = activation_constraints[layer.activation]

        constraint_func(block, net, layer_block, layer)
        activation_func(block, net, layer_block, layer)

    # setup input variables constraints
    # currently only support a single input layer
    input_layers = list(net.input_layers)
    assert len(input_layers) == 1
    input_layer = input_layers[0]

    @block.Constraint(input_layer.output_indexes)
    def input_assignment(b, *output_index):
        return b.scaled_inputs[output_index] == b.layer[id(input_layer)].z[output_index]

    # setup output variables constraints
    # currently only support a single output layer
    output_layers = list(net.output_layers)
    assert len(output_layers) == 1
    output_layer = output_layers[0]

    @block.Constraint(output_layer.output_indexes)
    def output_assignment(b, *output_index):
        return b.scaled_outputs[output_index] == b.layer[id(output_layer)].z[output_index]



class ReducedSpaceNeuralNetworkFormulation(_PyomoFormulation):
    """
    This class is used to build reduced-space formulations
    of neural networks.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    layer_constraints : dict-like or None
        overrides the constraints generated for the specified layer types
    activation_constraints : dict-like or None
        overrides the constraints generated for the specified activation functions
    """
    def __init__(self, network_structure): #, layer_constraints=None, activation_constraints=None):
        super().__init__()

        self.__network_definition = network_structure
        self.__scaling_object = network_structure.scaling_object
        self.__scaled_input_bounds = network_structure.scaled_input_bounds
        
        #self._layer_constraints = {**_DEFAULT_LAYER_CONSTRAINTS, **layer_constraints}
        self._activation_functions = {**_DEFAULT_ACTIVATION_FUNCTIONS} #, **activation_constraints}

    def _build_formulation(self):
        _setup_scaled_inputs_outputs(self.block,
                                     self.__scaling_object,
                                     self.__scaled_input_bounds)
        
        net = self.__network_definition
        layers = list(net.layers)
        block = self.block

        # create the blocks for each layer
        block.layers = pyo.Set(initialize=[id(layer) for layer in layers], ordered=True)
        block.layer = pyo.Block(block.layers)

        # currently only support a single input layer
        input_layers = list(net.input_layers)
        if len(input_layers) != 1:
            raise ValueError('build_formulation called with a network that has more than'
                             ' one input layer. Only single input layers are supported.')        
        input_layer = input_layers[0]
        input_layer_id = id(input_layer)
        input_layer_block = block.layer[input_layer_id]

        # connect the outputs of the input layer to
        # the main inputs on the block
        @input_layer_block.Expression(input_layer.output_indexes)
        def z(b, *output_index):
            pb = b.parent_block()
            return pb.scaled_inputs[output_index]

        # loop over the layers and build the expressions
        for layer in layers:
            if isinstance(layer, InputLayer):
                # skip the InputLayer
                continue

            # build the linear expressions and the activation function
            layer_id = id(layer)
            layer_block = block.layer[layer_id]
            layer_func = reduced_space_dense_layer #layer_constraints[type(layer)]
            activation_func = self._activation_functions[layer.activation]

            layer_func(block, net, layer_block, layer, activation_func)

        # setup output variable constraints
        # currently only support a single output layer
        output_layers = list(net.output_layers)
        if len(output_layers) != 1:
            raise ValueError('build_formulation called with a network that has more than'
                             ' one output layer. Only single output layers are supported.')
        output_layer = output_layers[0]

        @block.Constraint(output_layer.output_indexes)
        def output_assignment(b, *output_index):
            pb = b.parent_block()
            return b.scaled_outputs[output_index] == b.layer[id(output_layer)].z[output_index]

    # @property
    # def layer_constraints(self):
    #     return self._layer_constraints

    # @property
    # def activation_constraints(self):
    #     return self._activation_constraints

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Unsupported multiple network input variables'
        return network_inputs[0].input_indexes

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Unsupported multiple network output variables'
        return network_outputs[0].output_indexes
