"""
The omlt.block module contains the implementation of the OmltBlock class. This
class is used in combination with a formulation object to construct the
necessary constraints and variables to represent ML models.

Example:
    .. code-block:: python

        import tensorflow.keras as keras
        from omlt import OmltBlock
        from omlt.neuralnet import FullSpaceNNFormulation
        from omlt.io import load_keras_sequential

        nn = keras.models.load_model(keras_fname)
        net = load_keras_sequential(nn)

        m = pyo.ConcreteModel()
        m.neural_net_block = OmltBlock()
        m.neural_net_block.build_formulation(FullSpaceNNFormulation(net))

        m.obj = pyo.Objective(expr=(m.neural_net_block.outputs[2]-4.0)**2)
        status = pyo.SolverFactory('ipopt').solve(m, tee=True)
        pyo.assert_optimal_termination(status)
"""

import warnings

from omlt.base import OmltVar, DEFAULT_MODELING_LANGUAGE
from omlt.dependencies import julia_available
if julia_available:
    from omlt.base import jump

import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block


@declare_custom_block(name="OmltBlock")
class OmltBlockData(_BlockData):
    def __init__(self, component):
        super(OmltBlockData, self).__init__(component)
        self.__formulation = None
        self.__input_indexes = None
        self.__output_indexes = None
        self._format = DEFAULT_MODELING_LANGUAGE
        if self._format == "jump":
            self._jumpmodel = jump.Model()
        else:
            self._jumpmodel = None

    def set_format(self, format):
        self._format = format
        if self._format == "jump" and self._jumpmodel is None:
            self._jumpmodel = jump.Model()

    def _setup_inputs_outputs(self, *, input_indexes, output_indexes):
        """
        This function should be called by the derived class to create the
        inputs and outputs on the block

        Args:
           input_indexes : list
              list of indexes (can be tuples) defining the set to be used for
              the input variables
           output_indexes : list
              list of indexes (can be tuples) defining the set to be used for
              the input variables
        """
        self.__input_indexes = input_indexes
        self.__output_indexes = output_indexes

        if not input_indexes or not output_indexes:
            # TODO: implement this check higher up in the class hierarchy to
            # provide more contextual error msg
            raise ValueError(
                "OmltBlock must have at least one input and at least one output."
            )

        self.inputs_set = pyo.Set(initialize=input_indexes)
        self.inputs = OmltVar(self.inputs_set, initialize=0, format=self._format)
        self.outputs_set = pyo.Set(initialize=output_indexes)
        self.outputs = OmltVar(self.outputs_set, initialize=0, format=self._format)

    def build_formulation(self, formulation, format=None):
        """
        Call this method to construct the constraints (and possibly
        intermediate variables) necessary for the particular neural network
        formulation. The formulation object can be accessed later through the
        "formulation" attribute.

        Parameters
        ----------
        formulation : instance of _PyomoFormulation
            see, for example, FullSpaceNNFormulation
        format : str
            Which modelling language to build the formulation in.
            Currently supported are "pyomo" (default) and "jump".

        """

        if format is not None:
            self._format = format

        if self._format == "jump":
            self._jumpmodel = jump.Model()

        self._setup_inputs_outputs(
            input_indexes=list(formulation.input_indexes),
            output_indexes=list(formulation.output_indexes),
        )

        self.__formulation = formulation

        # tell the formulation that it is working on this block (self)
        self.__formulation._set_block(self)

        # tell the formulation object to construct the necessary models
        self.__formulation._build_formulation()
