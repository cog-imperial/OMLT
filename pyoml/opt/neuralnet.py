from .input_output import _BaseInputOutputBlockData
from pyomo.core.base.block import declare_custom_block
from .utils import build_full_space_formulation, build_reduced_space_formulation
import weakref
import abc

@declare_custom_block(name='NeuralNetworkBlock')
class NeuralNetworkBlockData(_BaseInputOutputBlockData):
    def __init__(self, component):
        super(NeuralNetworkBlockData,self).__init__(component)
        self._formulation = None

    def build_formulation(self, formulation, input_vars=None, output_vars=None):
        # call to the base class to define the inputs and the outputs
        super(NeuralNetworkBlockData, self).build_formulation(n_inputs=formulation.n_inputs,
                                                              n_outputs=formulation.n_outputs,
                                                              input_vars=input_vars,
                                                              output_vars=output_vars)

        # call to the formulation to build the constraints and any necessary intermediate variables
        self._formulation = formulation
        self._formulation._set_block(self)
        self.formulation.build_formulation()

    @property
    def formulation(self):
        return self._formulation

class _NeuralNetworkFormulationInterface(abc.ABC):
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
    def build_formulation(self):
        pass

class NeuralNetworkFormulation(_NeuralNetworkFormulationInterface):
    def __init__(self, network_structure):
        super(NeuralNetworkFormulation, self).__init__()
        self._network_structure = network_structure
        self._block = None

    def _set_block(self, block):
        self._block = weakref.ref(block)

    @property
    def block(self):
        return self._block()

    @property
    def network_structure(self):
        return self._network_structure

    @property
    def n_inputs(self):
        return self._network_structure.n_inputs

    @property
    def n_outputs(self):
        return self._network_structure.n_outputs

    @abc.abstractmethod
    def build_formulation(self):
        pass


class FullSpaceContinuousFormulation(NeuralNetworkFormulation):
    def __init__(self, network_structure):
        super(FullSpaceContinuousFormulation,self).__init__(network_structure)

    def build_formulation(self):
        build_full_space_formulation(block=self.block,
                                     network_structure=self.network_structure,
                                     skip_activations=False)


class ReducedSpaceContinuousFormulation(NeuralNetworkFormulation):
    def __init__(self, network_structure):
        super(ReducedSpaceContinuousFormulation,self).__init__(network_structure)

    def build_formulation(self):
        build_reduced_space_formulation(block=self.block,
                                        network_structure=self.network_structure,
                                         skip_activations=False)
