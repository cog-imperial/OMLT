from .input_output import _BaseInputOutputBlockData
from pyomo.core.base.block import declare_custom_block
from .utils import build_full_space_formulation, build_reduced_space_formulation
import weakref
import abc

"""
This module contains the implementation of the NeuralNetworkBlock class. This 
class is used in combination with a formulation object and optionally
with a list of input variables and output variables corresponding to the inputs
and outputs of the neural network. 
The formulation object is responsible for managing the construction and any refinement
or manipulation of the actual constraints.

Example 1:
    import tensorflow.keras as keras
    from pyoml.opt.keras_reader import load_keras_sequential
    from pyoml.opt.neuralnet import NeuralNetworkBlock, FullSpaceContinuousFormulation

    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = NeuralNetworkBlock()
    m.neural_net_block.build_formulation(FullSpaceContinuousFormulation(net))

    m.obj = pyo.Objective(expr=(m.neural_net_block.outputs[2]-4.0)**2)
    status = pyo.SolverFactory('ipopt').solve(m, tee=True)
    pyo.assert_optimal_termination(status)

 """

@declare_custom_block(name='NeuralNetworkBlock')
class NeuralNetworkBlockData(_BaseInputOutputBlockData):
    def __init__(self, component):
        super(NeuralNetworkBlockData,self).__init__(component)
        self.__formulation = None

    def build_formulation(self, formulation, input_vars=None, output_vars=None):
        """
        Call this method to construct the constraints (and possibly
        intermediate variables) necessary for the particular neural network
        formulation. The formulation object can be accessed later through the
        "formulation" attribute.

        Parameters
        ----------
        formulation : instance of NeuralNetworkFormulation
            see, for example, FullSpaceContinuousFormulation
        input_vars : list or None
            The list of var data objects that correspond to the inputs.
            This list must match the order of inputs from 0 .. n_inputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "inputs" is created on the
            automatically.
        output_vars :  list or None
            The list of var data objects that correspond to the outputs.
            This list must match the order of inputs from 0 .. n_outputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "outputs" is created on the
            automatically.
        """
        # call to the base class to define the inputs and the outputs
        super(NeuralNetworkBlockData, self)._setup_inputs_outputs(n_inputs=formulation.n_inputs,
                                                                  n_outputs=formulation.n_outputs,
                                                                  input_vars=input_vars,
                                                                  output_vars=output_vars)

        self.__formulation = formulation

        # tell the formulation that it is working on this block (self)
        self.__formulation._set_block(self)

        # tell the formulation object to construct the necessary models
        self.formulation._build_formulation()

    @property
    def formulation(self):
        """ The formulation object used to construct the constraints (and possibly
        intermediate variables) necessary to represent the neural network in Pyomo
        """
        return self.__formulation

class _NeuralNetworkFormulationInterface(abc.ABC):
    """ Base class interface for a neural network formulation object. This class
    is largely internal, and developers of new formulations should derive from
    :class:`pyoml.opt.neuralnet.NeuralNetworkFormulation`."""
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
    def n_inputs(self):
        pass

    @property
    @abc.abstractmethod
    def n_outputs(self):
        pass

    @abc.abstractmethod
    def _build_formulation(self):
        pass

class _NeuralNetworkFormulation(_NeuralNetworkFormulationInterface):
    def __init__(self, network_structure):
        """ This is a base class for different neural network formulations. To create a new
        formulation, inherit from this class and implement the build_formulation method. See
        :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for an example."""
        super(_NeuralNetworkFormulation, self).__init__()
        self.__network_definition = network_structure
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    @property
    def block(self):
        """ The underlying block containing the constraints / variables for this formulation."""
        return self.__block()

    @property
    def network_definition(self):
        """ The object providing a definition of the network structure. Network
        definitions can be loaded from common training packages (e.g., see
        :func:`pyoml.opt.keras_reader.load_keras_sequential`.) For a description
        of the network definition object, see
        :class:`pyoml.opt.network_definition.NetworkDefinition`"""
        return self.__network_definition

    @property
    def n_inputs(self):
        """ The number of inputs to the neural network. """
        return self.__network_definition.n_inputs

    @property
    def n_outputs(self):
        """ The number of outputs from the neural network."""
        return self.__network_definition.n_outputs

    @abc.abstractmethod
    def _build_formulation(self):
        """ This method is called by the NeuralNetworkBlock object to build the
        corresponding mathematical formulation of the neural network model.
        See :class:`pyoml.opt.neuralnet.FullSpaceContinuousFormulation` for
        an example of an implementation.
        """
        pass


class FullSpaceContinuousFormulation(_NeuralNetworkFormulation):
    def __init__(self, network_structure):
        """ This class provides a full-space formulation of a neural network,
        including all intermediate variables and activation functions.

        This class provides the neural network structure in a way that is *similar* to
        that provided in [1] as defined by:

        \begin{align*}
        x_i &= z_i                                   &&\forall i \in 0, ..., n_x - 1 \\
        \hat z_i &= \sum_{j{=}1}^N w_{ij} z_j + b_i  &&\forall i \in n_x, ..., n_x + n_h + n_y - 1 \\
        z_i &= \sigma(\hat z_i)                      &&\forall i \in n_x, ..., n_x + n_h + n_y - 1 \\
        y_i &= z_i                                   &&\forall i \in n_x + n_h, ..., n_x + n_h + n_y - 1.
        \end{align*}

        \noindent
        Here, $\sigma$ refers to the activation function, $x$ refers to the inputs, $z$ the values before activation,
        $\hat z$ the values after activation, and $y$ the outputs. Also, $n_x$ is the number of inputs,
        $n_h$ is the number of hidden nodes, and $n_y$ is the number of outputs.

        [1] Tjandraatmadja, C., Anderson, R., Huchette, J., Ma, W., Patel, K. and Vielma, J.P., 2020.
            The convex relaxation barrier, revisited: Tightened single-neuron relaxations for neural network
            verification. arXiv preprint arXiv:2006.14076.

        """
        super(FullSpaceContinuousFormulation, self).__init__(network_structure)

    def _build_formulation(self):
        """ This method is called by the NeuralNetworkBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_full_space_formulation(block=self.block,
                                     network_structure=self.network_definition,
                                     skip_activations=False)


class ReducedSpaceContinuousFormulation(_NeuralNetworkFormulation):
    def __init__(self, network_structure):
        """ This class builds a reduced-space formulation of the neural network where
        intermediate variables / constraints are eliminated."""
        super(ReducedSpaceContinuousFormulation, self).__init__(network_structure)

    def _build_formulation(self):
        """ This method is called by the NeuralNetworkBlock object to build the
                corresponding mathematical formulation of the neural network model.
        """
        #ToDo: This representation has performance issues with larger networks (likely in the nl writer)
        build_reduced_space_formulation(block=self.block,
                                        network_structure=self.network_definition,
                                        skip_activations=False)
