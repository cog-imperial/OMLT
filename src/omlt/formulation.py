import abc
import weakref

import pyomo.environ as pyo

from omlt.base import OmltConstraintFactory, OmltVarFactory


class _PyomoFormulationInterface(abc.ABC):
    """Pyomo Formulation Interface.

    Base class interface for a Pyomo formulation object. This class
    is largely internal, and developers of new formulations should derive from
    _PyomoFormulation.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def _set_block(self, block):
        pass

    @property
    @abc.abstractmethod
    def block(self):
        """Return the block associated with this formulation."""

    @property
    @abc.abstractmethod
    def input_indexes(self):
        """Input indexes.

        Return the indices corresponding to the inputs of the
        ML model. This is a list of entries (which may be tuples
        for higher dimensional inputs).
        """

    @property
    @abc.abstractmethod
    def output_indexes(self):
        """Output indexes.

        Return the indices corresponding to the outputs of the
        ML model. This is a list of entries (which may be tuples
        for higher dimensional outputs).
        """

    @abc.abstractmethod
    def _build_formulation(self):
        """Build formulation.

        This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the model.
        """

    @property
    @abc.abstractmethod
    def pyomo_only(self):
        """Pyomo Only.

        Return True if this formulation can only be built on a Pyomo
        block, and False if it can be built on blocks using other
        modeling languages.
        """


class _PyomoFormulation(_PyomoFormulationInterface):
    """Pyomo Formulation.

    This is a base class for different Pyomo formulations. To create a new
    formulation, inherit from this class and implement the abstract methods
    and properties.
    """

    def __init__(self):
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    @property
    def block(self):
        """Block.

        The underlying block containing the constraints/variables for this formulation.
        """
        return self.__block()


def scalar_or_tuple(x):
    if len(x) == 1:
        return x[0]
    return x


def _setup_scaled_inputs_outputs(block, scaler=None, scaled_input_bounds=None):
    var_factory = OmltVarFactory()
    if scaled_input_bounds is not None:
        bnds = {
            k: (float(scaled_input_bounds[k][0]), float(scaled_input_bounds[k][1]))
            for k in block.inputs_set
        }
        block.scaled_inputs = var_factory.new_var(
            block.inputs_set, initialize=0, lang=block._format, bounds=bnds
        )
    else:
        block.scaled_inputs = var_factory.new_var(
            block.inputs_set, initialize=0, lang=block._format
        )

    block.scaled_outputs = var_factory.new_var(
        block.outputs_set, initialize=0, lang=block._format
    )

    if scaled_input_bounds is not None and scaler is None:
        # set the bounds on the inputs to be the same as the scaled inputs
        for k in block.scaled_inputs:
            v = block.inputs[k]
            v.setlb(pyo.value(block.scaled_inputs[k].lb))
            v.setub(pyo.value(block.scaled_inputs[k].ub))

    if scaled_input_bounds is not None and scaler is not None:
        # propagate unscaled bounds to the inputs
        lbs = scaler.get_unscaled_input_expressions(
            {k: t[0] for k, t in scaled_input_bounds.items()}
        )
        ubs = scaler.get_unscaled_input_expressions(
            {k: t[1] for k, t in scaled_input_bounds.items()}
        )
        for k in block.inputs:
            v = block.inputs[k]
            v.setlb(lbs[k])
            v.setub(ubs[k])

    # create scaling expressions (just unscaled = scaled if no scaler provided)
    input_scaling_expressions = {k: block.inputs[k] for k in block.inputs}
    output_unscaling_expressions = {k: block.scaled_outputs[k] for k in block.outputs}
    if scaler is not None:
        input_scaling_expressions = scaler.get_scaled_input_expressions(
            input_scaling_expressions
        )
        output_unscaling_expressions = scaler.get_unscaled_output_expressions(
            output_unscaling_expressions
        )
    constraint_factory = OmltConstraintFactory()

    block._scale_input_constraint = constraint_factory.new_constraint(
        block.inputs_set, lang=block._format
    )

    for idx in block.inputs_set:
        block._scale_input_constraint[idx] = (
            block.scaled_inputs[idx] == input_scaling_expressions[idx]
        )

    block._scale_output_constraint = constraint_factory.new_constraint(
        block.outputs_set, lang=block._format
    )
    for idx in block.outputs_set:
        block._scale_output_constraint[idx] = (
            block.outputs[idx] == output_unscaling_expressions[idx]
        )
