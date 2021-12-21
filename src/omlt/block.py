import warnings
import weakref

import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block

from .utils import _extract_var_data

# TODO: Update documentation
"""
This module defines the base class for implementing a custom block
within Pyomo based on input / output connections.

This module also contains the implementation of the OmltBlock class. This
class is used in combination with a formulation object and optionally
with a list of input variables and output variables corresponding to the inputs
and outputs of the neural network.
The formulation object is responsible for managing the construction and any refinement
or manipulation of the actual constraints.

Example 1:
    import tensorflow.keras as keras
    from pyoml.opt.neuralnet.keras_reader import load_keras_sequential
    from pyoml.opt import OmltBlock, FullSpaceContinuousFormulation

    nn = keras.models.load_model(keras_fname)
    net = load_keras_sequential(nn)

    m = pyo.ConcreteModel()
    m.neural_net_block = OmltBlock()
    m.neural_net_block.build_formulation(FullSpaceContinuousFormulation(net))

    m.obj = pyo.Objective(expr=(m.neural_net_block.outputs[2]-4.0)**2)
    status = pyo.SolverFactory('ipopt').solve(m, tee=True)
    pyo.assert_optimal_termination(status)

"""
@declare_custom_block(name="OmltBlock")
class OmltBlockData(_BlockData):
    def __init__(self, component):
        super(OmltBlockData, self).__init__(component)
        self.__formulation = None
        self.__scaling_object = None
        self.__input_indexes = None
        self.__output_indexes = None

    def _setup_inputs_outputs(
        self, *, input_indexes, output_indexes):
        # TODO: Update this documentation
        """
        This function should be called by the derived class to setup the
        list of inputs and outputs for the input / output block.

        Args:
           TODO

        """
        self.__input_indexes = input_indexes
        self.__output_indexes = output_indexes
        if not input_indexes or not output_indexes:
            # todo: implement this check higher up in the class hierarchy to provide more contextual error msg
            raise ValueError(
                "_BaseInputOutputBlock must have at least one input and at least one output."
            )

        self.inputs_set = pyo.Set(initialize=input_indexes)
        self.inputs = pyo.Var(self.inputs_set, initialize=0)
        self.outputs_set = pyo.Set(initialize=output_indexes)
        self.outputs = pyo.Var(self.outputs_set, initialize=0)
    
    def build_formulation(self, formulation):
        """
        Call this method to construct the constraints (and possibly
        intermediate variables) necessary for the particular neural network
        formulation. The formulation object can be accessed later through the
        "formulation" attribute.

        Parameters
        ----------
        formulation : instance of PyomoFormulation
            see, for example, FullSpaceContinuousFormulation
        """
        # TODO: Do we want to validate formulation.input_indexes with input_vars here?
        # maybe formulation.check_input_vars(input_vars) or something?
        super(OmltBlockData, self)._setup_inputs_outputs(
            input_indexes=list(formulation.input_indexes),
            output_indexes=list(formulation.output_indexes),
        )

        self.__formulation = formulation

        # tell the formulation that it is working on this block (self)
        self.__formulation._set_block(self)

        # tell the formulation object to construct the necessary models
        self.formulation._build_formulation()

    @property
    def formulation(self):
        """The formulation object used to construct the constraints (and possibly
        intermediate variables) necessary to represent the neural network in Pyomo
        """
        return self.__formulation
