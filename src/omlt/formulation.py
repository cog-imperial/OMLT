import abc
import weakref
import pyomo.environ as pyo

class _PyomoFormulationInterface(abc.ABC):
    """Base class interface for a Pyomo formulation object. This class
    is largely internal, and developers of new formulations should derive from
    :class:`pyoml.opt.neuralnet.PyomoFormulation`."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def _set_block(self, block):
        pass

    @property
    @abc.abstractmethod
    def block(self):
        pass

    @property
    @abc.abstractmethod
    def input_indexes(self):
        pass

    @property
    @abc.abstractmethod
    def output_indexes(self):
        pass

    @abc.abstractmethod
    def _build_formulation(self):
        pass


class _PyomoFormulation(_PyomoFormulationInterface):
    def __init__(self):
        """This is a base class for different Pyomo formulations. To create a new
        formulation, inherit from this class and implement the build_formulation method. See
        :class:`pyoml.opt.neuralnet.NeuralNetworkFormulation` for an example."""
        super(_PyomoFormulation, self).__init__()
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    @property
    def block(self):
        """The underlying block containing the constraints / variables for this formulation."""
        return self.__block()

    @abc.abstractmethod
    def _build_formulation(self):
        """This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the model.
        See :class:`pyoml.opt.neuralnet.NeuralNetworkFormulation` for
        an example of an implementation.
        """
        pass

    # @property
    # def network_definition(self):
    #     """The object providing a definition of the network structure. Network
    #     definitions can be loaded from common training packages (e.g., see
    #     :func:`optml.io.keras_reader.load_keras_sequential`.) For a description
    #     of the network definition object, see
    #     :class:`pyoml.opt.network_definition.NetworkDefinition`"""
    #     return self.__network_definition

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

    # @property
    # def scaling_object(self):
    #     """The scaling object used in the underlying network definition."""
    #     return self.network_definition.scaling_object

    # @property
    # def input_bounds(self):
    #     """Return a list of tuples containing lower and upper bounds of neural network inputs"""
    #     return self.network_definition.input_bounds

def scaler_or_tuple(x):
    if len(x) == 1:
        return x[0]
    return x

def _setup_scaled_inputs_outputs(block, scaler=None, scaled_input_bounds=None):
    block.scaled_inputs = pyo.Var(block.inputs_set, initialize=0, bounds=scaled_input_bounds)
    block.scaled_outputs = pyo.Var(block.outputs_set, initialize=0)

    if scaled_input_bounds is not None:
        # set the bounds on the scaled variables
        for k in block.scaled_inputs:
            v = block.inputs[k]
            v.setlb(scaled_input_bounds[k][0])
            v.setub(scaled_input_bounds[k][1])

    if scaled_input_bounds is not None and scaler is not None:
        # propagate unscaled bounds to the inputs
        # TODO: add tests to make sure this handles negative bounds correctly
        lbs = scaler.get_unscaled_input_expressions( \
                    {k:t[0] for k,t in scaled_input_bounds.items()})
        ubs = scaler.get_unscaled_input_expressions( \
                    {k:t[1] for k,t in scaled_input_bounds.items()})
        for k in block.inputs:
            v = block.inputs[k]
            v.setlb(lbs[k])
            v.setub(ubs[k])

    # create scaling expressions (just unscaled = scaled if no scaler provided)
    input_scaling_expressions = {k:block.inputs[k] for k in block.inputs}
    output_unscaling_expressions = {k:block.scaled_outputs[k] for k in block.outputs}
    if scaler is not None:
        input_scaling_expressions = scaler.get_scaled_input_expressions(
            input_scaling_expressions)
        output_unscaling_expressions = scaler.get_unscaled_output_expressions(
            output_unscaling_expressions)

    @block.Constraint(block.scaled_inputs.index_set())
    def _scale_input_constraint(b, *args):
        return block.scaled_inputs[args] == input_scaling_expressions[scaler_or_tuple(args)]

    @block.Constraint(block.outputs.index_set())
    def _scale_output_constraint(b, *args):
        return block.outputs[args] == output_unscaling_expressions[scaler_or_tuple(args)]
