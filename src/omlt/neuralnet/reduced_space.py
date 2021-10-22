import pyomo.environ as pyo

from ..formulation import _PyomoFormulation
from ..utils import pyomo_activations


class ReducedSpaceContinuousFormulation(_PyomoFormulation):
    def __init__(self, network_structure):
        """This class builds a reduced-space formulation of the neural network where
        intermediate variables / constraints are eliminated."""
        super(ReducedSpaceContinuousFormulation, self).__init__(network_structure)

    def _build_formulation(self):
        """This method is called by the OmltBlock object to build the
        corresponding mathematical formulation of the neural network model.
        """
        # ToDo: This representation has performance issues with larger networks (likely in the nl writer)
        build_reduced_space_formulation(
            block=self.block,
            network_structure=self.network_definition,
            skip_activations=False,
        )


def build_reduced_space_formulation(block, network_structure, skip_activations=False):
    # for now, we build the full model with extraneous variables and constraints
    # Todo: provide an option to remove extraneous variables and constraints
    net = network_structure
    # scaling = net.scaling_object

    # map x and y to the inputs and outputs and verify the lengths
    # this is needed since the indexing in the input - output block
    # is not consistent with the nodal network representation
    input_node_ids = net.input_node_ids()
    inputs_list = block.scaled_inputs_list  # these are scaled inputs
    hidden_output_node_ids = net.hidden_node_ids()
    hidden_output_node_ids.extend(net.output_node_ids())
    output_node_ids = net.output_node_ids()
    outputs_list = block.scaled_outputs_list
    x = {input_node_ids[i]: inputs_list[i] for i in range(len(input_node_ids))}
    y = {output_node_ids[i]: outputs_list[i] for i in range(len(output_node_ids))}

    # add the intermediate variables
    block.nodes = pyo.Set(initialize=net.all_node_ids(), ordered=True)
    block.hidden_output_nodes = pyo.Set(initialize=hidden_output_node_ids, ordered=True)
    block.z = pyo.Expression(block.nodes)  # post-activation
    block.zhat = pyo.Expression(block.hidden_output_nodes)  # pre-activation

    # define the input constraints
    inputs = x
    # if scaling is not None:
    #     inputs = scaling.get_scaled_input_expressions(inputs)
    # Todo: We could eliminate these constraints and use x[i] directly where applicable
    for i in input_node_ids:
        block.z[i] = inputs[i]

    # define the linear constraints
    block.linear_constraints = pyo.Constraint(block.hidden_output_nodes)
    w = net.weights
    b = net.biases
    for i in block.hidden_output_nodes:
        block.zhat[i] = sum(w[i][j] * block.z[j] for j in w[i]) + b[i]

    # define the activation constraints
    if not skip_activations:
        activations = net.activations
        for i in block.hidden_output_nodes:
            if (
                i not in activations
                or activations[i] is None
                or activations[i] == "linear"
            ):
                block.z[i] = block.zhat[i]
            elif type(activations[i]) is str:
                afunc = pyomo_activations[activations[i]]
                block.z[i] = afunc(block.zhat[i])
            else:
                # better have given us a function that is valid for pyomo expressions
                block.z[i] = activations[i](block.zhat[i])

    # define the output constraints
    outputs = {i: block.z[i] for i in output_node_ids}
    # if scaling is not None:
    #     outputs = scaling.get_unscaled_output_expressions(outputs)
    # Todo: we could eliminate these constraints and use y[i] directly where applicable
    block.output_constraints = pyo.Constraint(output_node_ids)
    for i in output_node_ids:
        block.output_constraints[i] = y[i] == outputs[i]
