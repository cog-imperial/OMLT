import pyomo.environ as pyo

from ..formulation import _PyomoFormulation
from ..utils import pyomo_activations


class FullSpaceContinuousFormulation(_PyomoFormulation):
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
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        build_full_space_formulation(
            block=self.block,
            network_structure=self.network_definition,
            skip_activations=False,
        )


def build_full_space_formulation(block, network_structure, skip_activations=False):
    # for now, we build the full model with extraneous variables and constraints
    # Todo: provide an option to remove extraneous variables and constraints
    net = network_structure

    # map x and y to the inputs and outputs and verify the lengths
    # this is needed since the indexing in the input - output block
    # is not consistent with the nodal network representation
    input_node_ids = net.input_node_ids()
    inputs_list = block.scaled_inputs_list  # these are scaled inputs
    hidden_output_node_ids = list()
    hidden_output_node_ids.extend(net.hidden_node_ids())
    hidden_output_node_ids.extend(net.output_node_ids())
    output_node_ids = net.output_node_ids()
    outputs_list = block.scaled_outputs_list

    x = {input_node_ids[i]: inputs_list[i] for i in range(len(input_node_ids))}
    y = {output_node_ids[i]: outputs_list[i] for i in range(len(output_node_ids))}

    block.input_node_ids = pyo.Set(initialize=input_node_ids, ordered=True)

    # add the intermediate variables
    block.nodes = pyo.Set(initialize=net.all_node_ids(), ordered=True)
    block.hidden_output_nodes = pyo.Set(initialize=hidden_output_node_ids, ordered=True)
    block.z = pyo.Var(block.nodes, initialize=0)  # post-activation
    block.zhat = pyo.Var(block.hidden_output_nodes, initialize=0)  # pre-activation

    # define the input constraints
    inputs = x

    # Todo: We could eliminate these constraints and use x[i] directly where applicable
    block.input_constraints = pyo.Constraint(input_node_ids)
    for i in input_node_ids:
        lb, ub = inputs[i].bounds
        if lb is not None:
            block.z[i].setlb(lb)
        if ub is not None:
            block.z[i].setub(ub)
        block.input_constraints[i] = block.z[i] == inputs[i]

    # define the linear constraints
    block.linear_constraints = pyo.Constraint(block.hidden_output_nodes)
    w = net.weights
    b = net.biases
    for i in block.hidden_output_nodes:
        block.linear_constraints[i] = (
            block.zhat[i] == sum(w[i][j] * block.z[j] for j in w[i]) + b[i]
        )

    # define the activation constraints
    if not skip_activations:
        activations = net.activations
        block.activation_constraints = pyo.Constraint(block.hidden_output_nodes)
        for i in block.hidden_output_nodes:
            if (
                i not in activations
                or activations[i] is None
                or activations[i] == "linear"
            ):
                block.activation_constraints[i] = block.z[i] == block.zhat[i]
            elif type(activations[i]) is str:
                afunc = pyomo_activations[activations[i]]
                block.activation_constraints[i] = block.z[i] == afunc(block.zhat[i])
            else:
                # better have given us a function that is valid for pyomo expressions
                block.activation_constraints[i] = block.z[i] == activations[i](
                    block.zhat[i]
                )

    # define the output constraints
    outputs = {i: block.z[i] for i in output_node_ids}

    # Todo: we could eliminate these constraints and use y[i] directly where applicable
    block.output_constraints = pyo.Constraint(output_node_ids)
    for i in output_node_ids:
        block.output_constraints[i] = y[i] == outputs[i]
