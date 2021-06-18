import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block
from .utils import _extract_var_data

"""
This module defines the base class for implementing a custom block
within Pyomo based on input / output connections.
"""

@declare_custom_block(name='BaseInputOutputBlock')
class _BaseInputOutputBlockData(_BlockData):
    def __init__(self, component):
        super(_BaseInputOutputBlockData,self).__init__(component)
        self._n_inputs = None
        self._n_outputs = None
        self._inputs_list = None
        self._outputs_list = None

    def build_formulation(self, *, n_inputs, n_outputs, input_vars=None, output_vars=None):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs

        if input_vars is None:
            self.inputs_set = pyo.Set(initialize=range(n_inputs), ordered=True)
            self.inputs = pyo.Var(self.inputs_set, initialize=0)
            # Todo: why didn't this line work?
            # self._inputs_list = list(self.inputs)
            self._inputs_list = [self.inputs[i] for i in self.inputs.keys()]
        else:
            self._inputs_list = _extract_var_data(input_vars)
            assert len(self._inputs_list) == n_inputs

            # Discuss: This is kind of a cool idea
            # However, it may be confusing since the interface for vars and expressions is not the same?
            # let's discuss
            # def _input_expr(m,i):
            #    return self._inputs_list[i]
            # self.inputs = pyo.Expression(self.inputs_set, rule=_input_expr)

        if output_vars is None:
            self.outputs_set = pyo.Set(initialize=range(n_outputs), ordered=True)
            self.outputs = pyo.Var(self.outputs_set, initialize=0)
            self._outputs_list = [self.outputs[i] for i in self.outputs.keys()]
        else:
            self._outputs_list = _extract_var_data(output_vars)
            assert len(self._outputs_list) == network_definition.n_outputs()

            # Discuss: This is kind of a cool idea
            # However, it may be confusing since the interface for vars and expressions is not the same?
            # let's discuss
            # Todo: pass dict instead since I think this won't pickle?
            # def _output_expr(m,i):
            #     return self._outputs_list[i]
            # self.outputs = pyo.Expression(self.outputs_set, rule=_output_expr)

    def inputs_list(self):
        return self._inputs_list

    def outputs_list(self):
        return self._outputs_list
