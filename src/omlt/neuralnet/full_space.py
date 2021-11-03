import pyomo.environ as pyo

from omlt.formulation import _PyomoFormulation
from omlt.utils import pyomo_activations
from omlt.neuralnet.layer import ConvLayer, DenseLayer, InputLayer


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
    layers = list(net.nodes)
    layer_to_index_map = dict((id(l), i) for i, l in enumerate(layers))

    @block.Block([i for i in range(len(layers))])
    def layer(b, layer_no):
        layer = layers[layer_no]
        b.z = pyo.Var(layer.output_indexes, initialize=0)
        if not isinstance(layer, InputLayer):
            b.zhat = pyo.Var(layer.output_indexes, initialize=0)
        else:
            for index in layer.output_indexes:
                input_var = block.inputs[index]
                z_var = b.z[index]
                z_var.setlb(input_var.lb)
                z_var.setub(input_var.ub)

        b.constraints = pyo.ConstraintList()
        b.activations = pyo.ConstraintList()
        return b

    for layer_no, layer in enumerate(layers):
        if isinstance(layer, DenseLayer):
            layer_block = block.layer[layer_no]
            layer_block.__dense_expr = []
            # there should be only one input block for dense layers
            for input_layer in net.predecessors(layer):
                input_layer_no = layer_to_index_map[id(input_layer)]
                input_layer_block = block.layer[input_layer_no]
                for output_index in layer.output_indexes:
                    # dense layers multiply only the last dimension of
                    # their inputs
                    expr = layer.biases[output_index[-1]]
                    for local_index, input_index in layer.input_indexes_with_input_layer_indexes:
                        expr += (
                            layer.weights[local_index[-1], output_index[-1]] * 
                            input_layer_block.z[input_index]
                        )

                    layer_block.__dense_expr.append((output_index, expr))
                    layer_block.constraints.add(layer_block.zhat[output_index] == expr)
        elif isinstance(layer, ConvLayer):
            layer_block = block.layer[layer_no]
            layer_block.__dense_expr = []
            # there should be only one input block for conv layers
            for input_layer in net.predecessors(layer):
                input_layer_no = layer_to_index_map[id(input_layer)]
                input_layer_block = block.layer[input_layer_no]
                for out_d, out_r, out_c in layer.output_indexes:
                    expr = 0.0
                    for weight, input_index in layer.kernel_with_input_indexes(out_d, out_r, out_c):
                        expr += weight * input_layer_block.z[input_index]

                    output_index = (out_d, out_r, out_c)
                    layer_block.__dense_expr.append((output_index, expr))
                    layer_block.constraints.add(layer_block.zhat[output_index] == expr)
        else:
            # TODO: what to do with other layer types?
            pass

        # define the activation constraints
        if not skip_activations:
            for layer_no, layer in enumerate(layers):
                if isinstance(layer, InputLayer):
                    continue
                activation = layer.activation
                if activation is None or activation == "linear":
                    layer_block = block.layer[layer_no]
                    for output_index in layer.output_indexes:
                        layer_block.activations.add(
                            layer_block.z[output_index] == layer_block.zhat[output_index]
                        )
                elif type(activation) is str:
                    afunc = pyomo_activations[activation]
                    for output_index in layer.output_indexes:
                        layer_block.activations.add(
                            layer_block.z[output_index] == afunc(layer_block.zhat[output_index])
                        )
                else:
                    # better have given us a function that is valid for pyomo expressions
                    for output_index in layer.output_indexes:
                        layer_block.activations.add(
                            layer_block.z[output_index] == activation(layer_block.zhat[output_index])
                        )

    # setup output variables and constraints
    @block.Constraint(layers[-1].output_indexes)
    def output_assignment(b, *output_index):
        return b.outputs[output_index] == b.layer[len(layers) - 1].z[output_index]
