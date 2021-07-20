import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _extract_var_data

"""
This module defines the base class for implementing a custom block
within Pyomo based on input / output connections.
"""

@declare_custom_block(name='_BaseInputOutputBlock')
class _BaseInputOutputBlockData(_BlockData):
    def __init__(self, component):
        """
        Any block that inherits off of this must implement and call
        this __init__ with the passed component. This is to support
        interactions with Pyomo.
        """
        super(_BaseInputOutputBlockData,self).__init__(component)
        self.__n_inputs = None
        self.__n_outputs = None
        self.__inputs_list = None
        self.__outputs_list = None

    def _setup_inputs_outputs(self, *, n_inputs, n_outputs, input_vars=None, output_vars=None):
        """
        This function should be called by the derived class to setup the
        list of inputs and outputs for the input / output block.

        Parameters
        ----------
        n_inputs : int
            The number of inputs to the block
        n_outputs : int
            The number of outputs from the block
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
        self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        if n_inputs < 1 or n_outputs < 1:
            # todo: implement this check higher up in the class hierarchy to provide more contextual error msg
            raise ValueError('_BaseInputOutputBlock must have at least one input and at least one output.')

        self.__inputs_list = None
        self.__outputs_list = None

        if input_vars is None:
            self.inputs_set = pyo.Set(initialize=range(n_inputs), ordered=True)
            self.inputs = pyo.Var(self.inputs_set, initialize=0)
            self.__inputs_list = list(self.inputs.values())
        else:
            self.__inputs_list = _extract_var_data(input_vars)
            if len(self.__inputs_list) != n_inputs:
                raise ValueError('Length of input_vars does not match n_inputs.')

            # Discuss: This is kind of a cool idea
            # However, it may be confusing since the interface for vars and expressions is not the same?
            # let's discuss
            # def _input_expr(m,i):
            #    return self._inputs_list[i]
            # self.inputs = pyo.Expression(self.inputs_set, rule=_input_expr)

        if output_vars is None:
            self.outputs_set = pyo.Set(initialize=range(n_outputs), ordered=True)
            self.outputs = pyo.Var(self.outputs_set, initialize=0)
            self.__outputs_list = list(self.outputs.values())
        else:
            self.__outputs_list = _extract_var_data(output_vars)
            if len(self.__outputs_list) != n_outputs:
                raise ValueError('Length of output_vars does not match n_outputs.')

            # Discuss: This is kind of a cool idea
            # However, it may be confusing since the interface for vars and expressions is not the same?
            # let's discuss
            # Todo: pass dict instead since I think this won't pickle?
            # def _output_expr(m,i):
            #     return self._outputs_list[i]
            # self.outputs = pyo.Expression(self.outputs_set, rule=_output_expr)

    @property
    def inputs_list(self):
        return list(self.__inputs_list)

    @property
    def outputs_list(self):
        return list(self.__outputs_list)
