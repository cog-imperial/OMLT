import pyomo.environ as pyo

from omlt.formulation import _PyomoFormulation
from omlt.neuralnet.layer import ConvLayer, DenseLayer, InputLayer
from omlt.neuralnet.layers.full_space import full_space_dense_layer, full_space_conv_layer
from omlt.neuralnet.activations import linear_activation, bigm_relu_activation


def _ignore_input_layer():
    pass


_DEFAULT_LAYER_CONSTRAINTS = {
    InputLayer: _ignore_input_layer,
    DenseLayer: full_space_dense_layer,
    ConvLayer: full_space_conv_layer,
}

_DEFAULT_ACTIVATION_CONSTRAINTS = {
    "linear": linear_activation,
    "relu": bigm_relu_activation,
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
        super().__init__(network_structure)

        if layer_constraints is None:
            layer_constraints = dict()
        if activation_constraints is None:
            activation_constraints = dict()

        self._layer_constraints = {**_DEFAULT_LAYER_CONSTRAINTS, **layer_constraints}
        self._activation_constraints = {**_DEFAULT_ACTIVATION_CONSTRAINTS, **activation_constraints}

    def _build_formulation(self):
        build_neural_network_formulation(
            block=self.block,
            network_structure=self.network_definition,
            layer_constraints=self.layer_constraints,
            activation_constraints=self.activation_constraints,
        )

    @property
    def layer_constraints(self):
        return self._layer_constraints

    @property
    def activation_constraints(self):
        return self._activation_constraints


def build_neural_network_formulation(block, network_structure, layer_constraints, activation_constraints):
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
    @block.Block(block.layers)
    def layer(b, layer_id):
        net_layer = net.layer(layer_id)
        b.z = pyo.Var(net_layer.output_indexes, initialize=0)
        if isinstance(net_layer, InputLayer):
            for index in net_layer.output_indexes:
                input_var = block.inputs[index]
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
    input_layers = list(net.input_layers)
    assert len(input_layers) == 1
    input_layer = input_layers[0]

    @block.Constraint(input_layer.output_indexes)
    def input_assignment(b, *output_index):
        return b.inputs[output_index] == b.layer[id(input_layer)].z[output_index]

    # setup output variables constraints
    output_layers = list(net.output_layers)
    assert len(output_layers) == 1
    output_layer = output_layers[0]

    @block.Constraint(output_layer.output_indexes)
    def output_assignment(b, *output_index):
        return b.outputs[output_index] == b.layer[id(output_layer)].z[output_index]
