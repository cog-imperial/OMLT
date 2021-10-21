import warnings

import pyomo.environ as pyo
from pyomo.core.base.block import _BlockData, declare_custom_block

from .utils import _extract_var_data

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


@declare_custom_block(name="_BaseInputOutputBlock")
class _BaseInputOutputBlockData(_BlockData):
    def __init__(self, component):
        """
        Any block that inherits off of this must implement and call
        this __init__ with the passed component. This is to support
        interactions with Pyomo.
        """
        super(_BaseInputOutputBlockData, self).__init__(component)
        self.__n_inputs = None
        self.__n_outputs = None
        self.__inputs_list = None
        self.__outputs_list = None
        self.__scaled_inputs_list = None
        self.__unscaled_outputs_list = None

    def _setup_inputs_outputs(
        self, *, n_inputs, n_outputs, input_vars=None, output_vars=None
    ):
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
            raise ValueError(
                "_BaseInputOutputBlock must have at least one input and at least one output."
            )

        self.__inputs_list = None
        self.__outputs_list = None

        if input_vars is None:
            self.inputs_set = pyo.Set(initialize=range(n_inputs), ordered=True)
            self.inputs = pyo.Var(self.inputs_set, initialize=0)
            self.__inputs_list = list(self.inputs.values())
        else:
            self.__inputs_list = _extract_var_data(input_vars)
            if len(self.__inputs_list) != n_inputs:
                raise ValueError("Length of input_vars does not match n_inputs.")

        if output_vars is None:
            self.outputs_set = pyo.Set(initialize=range(n_outputs), ordered=True)
            self.outputs = pyo.Var(self.outputs_set, initialize=0)
            self.__outputs_list = list(self.outputs.values())
        else:
            self.__outputs_list = _extract_var_data(output_vars)
            if len(self.__outputs_list) != n_outputs:
                raise ValueError("Length of output_vars does not match n_outputs.")

    def _setup_input_bounds(self, inputs_list, input_bounds=None):
        if input_bounds:
            # set bounds using provided input_bounds
            for (i, var) in enumerate(inputs_list):
                if var.lb == None:  # set lower bound to input_bounds value
                    inputs_list[i].setlb(input_bounds[i][0])
                else:
                    # throw warning if var.lb is more loose than input_bounds value
                    if var.lb < input_bounds[i][0]:
                        warnings.warning(
                            "Variable {} lower bound {} is less tight then network definition bound {}".format(
                                var, var.lb, input_bounds[i][0]
                            )
                        )
                if var.ub == None:
                    inputs_list[i].setub(input_bounds[i][1])
                else:
                    # throw warning if var.ub is more loose than input_bounds value
                    if var.ub > input_bounds[i][1]:
                        warnings.warning(
                            "Variable {} upper bound {} is less tight then network definition bound {}".format(
                                var, var.ub, input_bounds[i][1]
                            )
                        )

    def _setup_scaled_inputs_outputs(
        self, *, scaling_object=None, input_bounds=None, use_scaling_expressions=False
    ):
        if scaling_object == None:
            # if no scaling, set scaled lists to the original lists
            self.__scaled_inputs_list = self.inputs_list
            self.__scaled_outputs_list = self.outputs_list
            self._setup_input_bounds(self.inputs_list, input_bounds)

        elif scaling_object and use_scaling_expressions:
            # use pyomo Expressions for scaled and unscaled terms, variable bounds are not directly captured
            self.__scaled_inputs_list = scaling_object.get_scaled_input_expressions(
                self.inputs_list
            )
            self.__scaled_outputs_list = scaling_object.get_scaled_output_expressions(
                self.outputs_list
            )
            # Bounds only set on unscaled inputs
            self._setup_input_bounds(self.inputs_list, input_bounds)

        else:
            # create pyomo variables for scaled and unscaled terms, input bounds are also scaled
            self.scaled_inputs_set = pyo.Set(
                initialize=range(len(self.__inputs_list)), ordered=True
            )
            self.scaled_inputs = pyo.Var(self.scaled_inputs_set, initialize=0)

            self.scaled_outputs_set = pyo.Set(
                initialize=range(len(self.__outputs_list)), ordered=True
            )
            self.scaled_outputs = pyo.Var(self.scaled_outputs_set, initialize=0)

            # set scaled variables lists
            self.__scaled_inputs_list = list(self.scaled_inputs.values())
            self.__scaled_outputs_list = list(self.scaled_outputs.values())

            # Create constraints connecting scaled and unscaled variables
            self.__scale_input_con = pyo.Constraint(self.scaled_inputs_set)
            self.__unscale_output_con = pyo.Constraint(self.scaled_outputs_set)
            scaled_input_expressions = scaling_object.get_scaled_input_expressions(
                self.inputs_list
            )
            unscaled_output_expressions = (
                scaling_object.get_unscaled_output_expressions(self.scaled_outputs_list)
            )

            # scaled input constraints
            for i in range(len(self.scaled_inputs_set)):
                self.__scale_input_con[i] = (
                    self.scaled_inputs[i] == scaled_input_expressions[i]
                )
            # unscaled output constraints
            for i in range(len(self.scaled_outputs_set)):
                self.__unscale_output_con[i] = (
                    self.outputs_list[i] == unscaled_output_expressions[i]
                )

            # scale input bounds
            if input_bounds:
                input_lower = [input_bounds[i][0] for i in range(len(input_bounds))]
                input_upper = [input_bounds[i][1] for i in range(len(input_bounds))]
                scaled_lower = scaling_object.get_scaled_input_expressions(input_lower)
                scaled_upper = scaling_object.get_scaled_input_expressions(input_upper)
                scaled_input_bounds = list(zip(scaled_lower, scaled_upper))
                self._setup_input_bounds(self.scaled_inputs_list, scaled_input_bounds)
        return

    @property
    def inputs_list(self):
        return list(self.__inputs_list)

    @property
    def outputs_list(self):
        return list(self.__outputs_list)

    @property
    def scaled_inputs_list(self):
        return list(self.__scaled_inputs_list)

    @property
    def scaled_outputs_list(self):
        return list(self.__scaled_outputs_list)


@declare_custom_block(name="OmltBlock")
class OmltBlockData(_BaseInputOutputBlockData):
    def __init__(self, component):
        super(OmltBlockData, self).__init__(component)
        self.__formulation = None
        self.__scaling_object = None

    def build_formulation(self, formulation, input_vars=None, output_vars=None):
        """
        Call this method to construct the constraints (and possibly
        intermediate variables) necessary for the particular neural network
        formulation. The formulation object can be accessed later through the
        "formulation" attribute.

        Parameters
        ----------
        formulation : instance of PyomoFormulation
            see, for example, FullSpaceContinuousFormulation
        input_vars : list or None
            The list of var data objects that correspond to the inputs.
            This list must match the order of inputs from 0 .. n_inputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "inputs" is created on the
            block automatically.
        output_vars :  list or None
            The list of var data objects that correspond to the outputs.
            This list must match the order of inputs from 0 .. n_outputs-1.
            Note that mixing Var, IndexedVar, and VarData objects is fully
            supported, and any indexed variables will be expanded to form
            the final list of input var data objects. Note that any IndexedVar
            included should be defined over a sorted set.

            If set to None, then an indexed variable "outputs" is created on the
            block automatically.
        """
        # call to the base class to define the inputs and the outputs
        super(OmltBlockData, self)._setup_inputs_outputs(
            n_inputs=formulation.n_inputs,
            n_outputs=formulation.n_outputs,
            input_vars=input_vars,
            output_vars=output_vars,
        )

        super(OmltBlockData, self)._setup_scaled_inputs_outputs(
            scaling_object=formulation.scaling_object,
            input_bounds=formulation.input_bounds,
            use_scaling_expressions=False,
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

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.formulation.scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.formulation.input_bounds
