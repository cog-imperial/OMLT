import warnings


class NetworkDefinition(object):
    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_outputs,
        weights,
        biases,
        activations,
        scaling_object=None,
        input_bounds=None,
    ):
        """
        This class provides the neural network structure in a way that is *similar* to
        that provided in [1] as defined by:

        \begin{align*}
        x_i &= z_i                                   &&\forall i \in 0, ..., n_x - 1 \\
        \hat z_i &= \sum_{j{=}1}^N w_{ij} z_j + b_i  &&\forall i \in n_x, ..., n_x + n_h + n_y - 1 \\
        z_i &= \sigma(\hat z_i)                      &&\forall i \in n_x, ..., n_x + n_h + n_y - 1 \\
        y_i &= z_i                                   &&\forall i \in n_x + n_h, ..., n_x + n_h + n_y - 1.
        \end{align*}

        \noindent
        Here, $\sigma$ refers to the activation function, $x$ refers to the inputs, $\hat z$ the values before activation,
        $z$ the values after activation, and $y$ the outputs. Also, $n_x$ is the number of inputs,
        $n_h$ is the number of hidden nodes, and $n_y$ is the number of outputs.

        [1] Tjandraatmadja, C., Anderson, R., Huchette, J., Ma, W., Patel, K. and Vielma, J.P., 2020.
            The convex relaxation barrier, revisited: Tightened single-neuron relaxations for neural network
            verification. arXiv preprint arXiv:2006.14076.

        Parameters
        ----------
        n_inputs : int
            Number of inputs (input nodes) for the network
        n_hidden : int
            Number of hidden nodes (all nodes except input and output)
        n_outputs : int
            Number of outputs (output nodes) for the network
        weights : dict of dict
           The weights is a dictionary of dictionaries where w[i][j] indicates that the summation of node "i"
           includes a term from upstream node "j" (there is a connection from node j to node i). With this
           representation there should be an entry in w for every node except the input nodes.
        biases : dict
            The biases for every node. With this representation, biases will have an entry for every node except
            the input nodes.
        activations : dict
            The activation functions for every node. With this representation, activations can have an entry for
            every node except the input nodes. If no entry is present, or the entry is None, it is assumed that
            the activation is identity (i.e. z_i = \hat z_i)
        scaling_object : instance of object supporting ScalingInterface
            This object must support the ScalingInterface (see scaling.py)
        input_bounds: list of tuples
            List of tuples where each tuple contains the lower and upper bound for an UNSCALED input 
            e.g. input_bounds = [(lb1,ub1),(lb2,ub2),(lb3,ub3)] for 3 inputs
        """
        self.__n_inputs = n_inputs
        self.__n_hidden = n_hidden
        self.__n_outputs = n_outputs

        self.__weights = weights
        self.__biases = biases
        self.__activations = activations
        self.__scaling_object = scaling_object
        self.__input_bounds = input_bounds

        if len(weights) != n_hidden + n_outputs:
            raise ValueError(
                "The length of the weights dictionary should match "
                "n_hidden + n_outputs"
            )
        if len(biases) != n_hidden + n_outputs:
            raise ValueError(
                "The length of the biases dictionary should match "
                "n_hidden + n_outputs"
            )
        if input_bounds == None:
            warnings.warn(
                "No input bounds were provided. This may lead to extrapolation outside of the training data"
            )
        else:
            if len(input_bounds) != n_inputs:
                raise ValueError(
                    "The length of the input_bounds list should match n_inputs"
                )
            if not all(len(i) == 2 for i in input_bounds):
                raise ValueError(
                    "The elements of input_bounds must be tuples of length 2 containing (lower_bound,upper_bound)"
                )

        # todo: we should probably add more error checking here

    @property
    def n_inputs(self):
        """Return the number of input nodes"""
        return self.__n_inputs

    @property
    def n_hidden(self):
        """Return the number of hidden nodes"""
        return self.__n_hidden

    @property
    def n_outputs(self):
        """Return the number of output nodes"""
        return self.__n_outputs

    @property
    def weights(self):
        """Return the weights dictionary as described above"""
        return self.__weights

    @property
    def biases(self):
        """Return the biases dictionary as described above"""
        return self.__biases

    @property
    def activations(self):
        """Return the activations dictionary as described above"""
        return self.__activations

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.__scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.__input_bounds

    @scaling_object.setter
    def scaling_object(self, scaling_object):
        self.__scaling_object = scaling_object

    def input_node_ids(self):
        """Return the ids associated with all the input nodes.
        This is {0 .. n_x-1}"""
        return [i for i in range(self.n_inputs)]

    def hidden_node_ids(self):
        """Return the ids associated with all of the hidden nodes.
        This is {n_x .. n_x+n_h-1}"""
        return [i for i in range(self.n_inputs, self.n_inputs + self.n_hidden)]

    def output_node_ids(self):
        """Return the ids associated with all of the output nodes.
        This is {n_x+n_h .. n_x+n_h+n_y-1}"""
        return [
            i
            for i in range(
                self.n_inputs + self.n_hidden,
                self.n_inputs + self.n_hidden + self.n_outputs,
            )
        ]

    def all_node_ids(self):
        """Return the ids associated with all of the nodes.
        This is {0 .. n_x+n_h+n_y-1}"""
        return [i for i in range(0, self.n_inputs + self.n_hidden + self.n_outputs)]
