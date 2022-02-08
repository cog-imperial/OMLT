import pyomo.environ as pyo
import numpy as np

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs
from omlt.neuralnet.layer import ConvLayer, DenseLayer, InputLayer
from omlt.neuralnet.layers.full_space import (full_space_dense_layer, full_space_conv_layer)
from omlt.neuralnet.layers.reduced_space import reduced_space_dense_layer
from omlt.neuralnet.layers.partition_based import partition_based_dense_relu_layer, default_partition_split_func
from omlt.neuralnet.activations import (linear_activation_constraint,
                                        linear_activation_function,
                                        bigm_relu_activation_constraint,
                                        ComplementarityReLUActivation,
                                        sigmoid_activation_constraint,
                                        softplus_activation_constraint,
                                        tanh_activation_constraint,
                                        sigmoid_activation_function,
                                        softplus_activation_function,
                                        tanh_activation_function)
from omlt.neuralnet.activations import ACTIVATION_FUNCTION_MAP as _DEFAULT_ACTIVATION_FUNCTIONS

def _ignore_input_layer():
    pass

_DEFAULT_LAYER_CONSTRAINTS = {
    InputLayer: _ignore_input_layer,
    DenseLayer: full_space_dense_layer,
    ConvLayer: full_space_conv_layer,
}

_DEFAULT_ACTIVATION_CONSTRAINTS = {
    "linear": linear_activation_constraint,
    "relu": bigm_relu_activation_constraint,
    "sigmoid": sigmoid_activation_constraint,
    "softplus": softplus_activation_constraint,
    "tanh": tanh_activation_constraint
}

class FullSpaceNNFormulation(_PyomoFormulation):
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

        self._layer_constraints = dict(self._supported_default_layer_constraints())
        self._activation_constraints = dict(self._supported_default_activation_constraints())
        if layer_constraints is not None:
            self._layer_constraints.update(layer_constraints)
        if activation_constraints is not None:
            self._activation_constraints.update(activation_constraints)

        # TODO: Change these to exceptions.
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Multiple input layers are not currently supported.'
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Multiple output layers are not currently supported.'

    def _supported_default_layer_constraints(self):
        return _DEFAULT_LAYER_CONSTRAINTS

    def _supported_default_activation_constraints(self):
        return _DEFAULT_ACTIVATION_CONSTRAINTS
    
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

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        network_inputs = list(self.__network_definition.input_nodes)
        assert len(network_inputs) == 1, 'Multiple input layers are not currently supported.'
        return network_inputs[0].input_indexes

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        network_outputs = list(self.__network_definition.output_nodes)
        assert len(network_outputs) == 1, 'Multiple output layers are not currently supported.'
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
        layer_constraints_func = layer_constraints.get(type(layer), None)
        activation_constraints_func = activation_constraints.get(layer.activation, None)

        if layer_constraints_func is None:
            raise ValueError('Layer type {} is not supported by this formulation.'.format(type(layer)))
        if activation_constraints_func is None:
            raise ValueError('Activation {} is not supported by this formulation.'.format(layer.activation))
        
        layer_constraints_func(block, net, layer_block, layer)
        activation_constraints_func(block, net, layer_block, layer)

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


class FullSpaceSmoothNNFormulation(FullSpaceNNFormulation):
    def __init__(self, network_structure):
        """
        This class is used for building "full-space" formulations of 
        neural network models composed of smooth activations (e.g., tanh,
        sigmoid, etc.)

        Parameters
        ----------
        network_structure : NetworkDefinition
           the neural network definition
        """
        super().__init__(network_structure)

    def _supported_default_activation_constraints(self):
        return {
            "linear": linear_activation_constraint,
            "sigmoid": sigmoid_activation_constraint,
            "softplus": softplus_activation_constraint,
            "tanh": tanh_activation_constraint
        }

class ReluBigMFormulation(FullSpaceNNFormulation):
    def __init__(self, network_structure):
        """
        This class is used for building "full-space" formulations of 
        neural network models composed of relu activations using a
        big-M formulation

        Parameters
        ----------
        network_structure : NetworkDefinition
           the neural network definition
        """
        super().__init__(network_structure)

    def _supported_default_activation_constraints(self):
        return {
            "linear": linear_activation_constraint,
            "relu": bigm_relu_activation_constraint
        }

class ReluComplementarityFormulation(FullSpaceNNFormulation):
    def __init__(self, network_structure):
        """
        This class is used for building "full-space" formulations of 
        neural network models composed of relu activations using
        a complementarity formulation (smooth represenation)

        Parameters
        ----------
        network_structure : NetworkDefinition
           the neural network definition
        """
        super().__init__(network_structure)

    def _supported_default_activation_constraints(self):
        return {
            "linear": linear_activation_constraint,
            "relu": ComplementarityReLUActivation()
        }


class ReducedSpaceNNFormulation(_PyomoFormulation):
    """
    This class is used to build reduced-space formulations
    of neural networks.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    activation_functions : dict-like or None
        overrides the actual functions used for particular activations
    """
    def __init__(self, network_structure, activation_functions=None):
        super().__init__()

        self.__network_definition = network_structure
        self.__scaling_object = network_structure.scaling_object
        self.__scaled_input_bounds = network_structure.scaled_input_bounds

        
        # TODO: look into increasing support for other layers / activations
        #self._layer_constraints = {**_DEFAULT_LAYER_CONSTRAINTS, **layer_constraints}
        self._activation_functions = dict(self._supported_default_activation_functions())
        if activation_functions is not None:
            self._activation_functions.update(activation_functions)

    def _supported_default_activation_functions(self):
        return dict(_DEFAULT_ACTIVATION_FUNCTIONS)
    
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

            # TODO: Add error checking on layer type
            # build the linear expressions and the activation function
            layer_id = id(layer)
            layer_block = block.layer[layer_id]
            layer_func = reduced_space_dense_layer #layer_constraints[type(layer)]
            activation_func = self._activation_functions.get(layer.activation, None)
            if activation_func is None:
                raise ValueError('Activation {} is not supported by this formulation.'.format(layer.activation))

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

class ReducedSpaceSmoothNNFormulation(ReducedSpaceNNFormulation):
    """
    This class is used to build reduced-space formulations
    of neural networks with smooth activation functions.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    """
    def __init__(self, network_structure):
        super().__init__(network_structure)

    def _supported_default_activation_functions(self):
        return {
            "linear": linear_activation_function,
            "sigmoid": sigmoid_activation_function,
            "softplus": softplus_activation_function,
            "tanh": tanh_activation_function
            }

class ReluPartitionFormulation(_PyomoFormulation):
    """
    This class is used to build partition-based formulations
    of neural networks.

    Parameters
    ----------
    network_structure : NetworkDefinition
        the neural network definition
    split_func : callable
        the function used to compute the splits
    """
    def __init__(self, network_structure, split_func=None):
        super().__init__()

        self.__network_definition = network_structure
        self.__scaling_object = network_structure.scaling_object
        self.__scaled_input_bounds = network_structure.scaled_input_bounds

        if split_func is None:
            split_func = lambda w: default_partition_split_func(w, 2)

        self.__split_func = split_func

    def _build_formulation(self):
        _setup_scaled_inputs_outputs(self.block,
                                     self.__scaling_object,
                                     self.__scaled_input_bounds)
        block = self.block
        net = self.__network_definition
        layers = list(net.layers)
        split_func = self.__split_func

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
            if isinstance(layer, DenseLayer):
                if layer.activation == "relu":
                    partition_based_dense_relu_layer(block, net, layer_block, layer, split_func)
                elif layer.activation == "linear":
                    full_space_dense_layer(block, net, layer_block, layer)
                    linear_activation_constraint(block, net, layer_block, layer)
                else:
                    raise ValueError("ReluPartitionFormulation supports Dense layers with relu or linear activation")
            else:
                raise ValueError("ReluPartitionFormulation supports only Dense layers")

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
