import abc
import weakref


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
    def __init__(self, network_structure):
        """This is a base class for different Pyomo formulations. To create a new
        formulation, inherit from this class and implement the build_formulation method. See
        :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for an example."""
        super(_PyomoFormulation, self).__init__()
        self.__network_definition = network_structure
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    @property
    def block(self):
        """The underlying block containing the constraints / variables for this formulation."""
        return self.__block()

    @property
    def network_definition(self):
        """The object providing a definition of the network structure. Network
        definitions can be loaded from common training packages (e.g., see
        :func:`optml.io.keras_reader.load_keras_sequential`.) For a description
        of the network definition object, see
        :class:`pyoml.opt.network_definition.NetworkDefinition`"""
        return self.__network_definition

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

    @property
    def scaling_object(self):
        """The scaling object used in the underlying network definition."""
        return self.network_definition.scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.network_definition.input_bounds

    @abc.abstractmethod
    def _build_formulation(self):
        """This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the model.
        See :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for
        an example of an implementation.
        """
        pass
