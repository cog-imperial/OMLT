import abc
import weakref
import pyomo.environ as pyo

class _PyomoFormulationInterface(abc.ABC):
    """
    Base class interface for a Pyomo formulation object. This class
    is largely internal, and developers of new formulations should derive from
    _PyomoFormulation.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def _set_block(self, block):
        pass

    @property
    @abc.abstractmethod
    def block(self):
        """ Return the block associated with this formulation.
        """
        pass

    @property
    @abc.abstractmethod
    def input_indexes(self):
        """ Return the indices corresponding to the inputs of the 
        ML model. This is a list of entries (which may be tuples 
        for higher dimensional inputs).
        """
        pass

    @property
    @abc.abstractmethod
    def output_indexes(self):
        """ Return the indices corresponding to the outputs of the 
        ML model. This is a list of entries (which may be tuples 
        for higher dimensional outputs).
        """
        pass

    @abc.abstractmethod
    def _build_formulation(self):
        """This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the model.
        """
        pass


class _PyomoFormulation(_PyomoFormulationInterface):
    def __init__(self):
        """
        This is a base class for different Pyomo formulations. To create a new
        formulation, inherit from this class and implement the abstract methods and properties.
        """
        super(_PyomoFormulation, self).__init__()
        self.__block = None

    def _set_block(self, block):
        self.__block = weakref.ref(block)

    @property
    def block(self):
        """The underlying block containing the constraints / variables for this formulation."""
        return self.__block()


def scalar_or_tuple(x):
    if len(x) == 1:
        return x[0]
    return x

def _setup_scaled_inputs_outputs(block, scaler=None, scaled_input_bounds=None):
    if scaled_input_bounds is not None:
        def bounds_rule(m, *k):
            return scaled_input_bounds.__getitem__(scalar_or_tuple(k))
        #bounds_rule = lambda m, *k : scaled_input_bounds.__getitem__(scalar_or_tuple(k))
        block.scaled_inputs = pyo.Var(block.inputs_set, initialize=0,
                                      bounds=bounds_rule)
    else:
        block.scaled_inputs = pyo.Var(block.inputs_set, initialize=0)
        
    block.scaled_outputs = pyo.Var(block.outputs_set, initialize=0)

    if scaled_input_bounds is not None and scaler is None:
        # set the bounds on the inputs to be the same as the scaled inputs
        for k in block.scaled_inputs:
            v = block.inputs[k]
            v.setlb(pyo.value(block.scaled_inputs[k].lb))
            v.setub(pyo.value(block.scaled_inputs[k].ub))

    if scaled_input_bounds is not None and scaler is not None:
        # propagate unscaled bounds to the inputs
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
        return block.scaled_inputs[args] == input_scaling_expressions[scalar_or_tuple(args)]

    @block.Constraint(block.outputs.index_set())
    def _scale_output_constraint(b, *args):
        return block.outputs[args] == output_unscaling_expressions[scalar_or_tuple(args)]
