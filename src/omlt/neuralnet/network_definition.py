import networkx as nx

from omlt.neuralnet.layer import Layer, DenseLayer, GNNLayer


class NetworkDefinition:
    def __init__(
        self, scaling_object=None, scaled_input_bounds=None, unscaled_input_bounds=None
    ):
        """
        Create a network definition object used to create the neural network
        formulation in Pyomo

        Args:
           scaling_object : ScalingInterface or None
              A scaling object to specify the scaling parameters for the
              neural network inputs and outputs. If None, then no
              scaling is performed.
           scaled_input_bounds : dict or None
              A dict that contains the bounds on the scaled variables (the
              direct inputs to the neural network). If None, then no bounds
              are specified or they are generated using unscaled bounds.
           unscaled_input_bounds: dict or None
              A dict that contains the bounds on the scaled variables (the
              direct inputs to the neural network). If supplied the scaled_input_bounds
              parameter will be generated using the scaling object.
              If None, then no bounds are specified.
        """
        self.__layers_by_id = dict()
        self.__graph = nx.DiGraph()
        self.__scaling_object = scaling_object

        # Process input bounds to insure scaled input bounds exist for formulations
        if scaled_input_bounds is None:
            if unscaled_input_bounds is not None and scaling_object is not None:
                lbs = scaling_object.get_scaled_input_expressions(
                    {k: t[0] for k, t in unscaled_input_bounds.items()}
                )
                ubs = scaling_object.get_scaled_input_expressions(
                    {k: t[1] for k, t in unscaled_input_bounds.items()}
                )

                scaled_input_bounds = {
                    k: (lbs[k], ubs[k]) for k in unscaled_input_bounds.keys()
                }

            # If unscaled input bounds provided and no scaler provided, scaled input bounds = unscaled input bounds
            elif unscaled_input_bounds is not None and scaling_object is None:
                scaled_input_bounds = unscaled_input_bounds

        self.__unscaled_input_bounds = unscaled_input_bounds
        self.__scaled_input_bounds = scaled_input_bounds

    def add_layer(self, layer):
        """
        Add a layer to the network.

        Parameters
        ----------
        layer : Layer
            the layer to add to the network
        """
        layer_id = id(layer)
        self.__layers_by_id[layer_id] = layer
        self.__graph.add_node(layer_id)

    def add_edge(self, from_layer, to_layer):
        """
        Add an edge between two layers.

        Parameters
        ----------
        from_layer : Layer
            the layer with the outbound connection
        to_layer : Layer
            the layer with the inbound connection
        """
        id_to = id(to_layer)
        id_from = id(from_layer)
        assert id_to in self.__layers_by_id
        assert id_from in self.__layers_by_id
        self.__graph.add_edge(id_from, id_to)

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.__scaling_object

    @property
    def scaled_input_bounds(self):
        """Return a dict of tuples containing lower and upper bounds of neural network inputs"""
        return self.__scaled_input_bounds

    @property
    def unscaled_input_bounds(self):
        """Return a dict of tuples containing lower and upper bounds of unscaled neural network inputs"""
        return self.__unscaled_input_bounds

    @property
    def input_layers(self):
        """Return an iterator over the input layers"""
        for layer_id, in_degree in self.__graph.in_degree():
            if in_degree == 0:
                yield self.__layers_by_id[layer_id]

    @property
    def input_nodes(self):
        """An alias for input_layers"""
        return self.input_layers

    @property
    def output_layers(self):
        """Return an iterator over the output layer"""
        for layer_id, out_degree in self.__graph.out_degree():
            if out_degree == 0:
                yield self.__layers_by_id[layer_id]

    @property
    def output_nodes(self):
        """An alias for output_layers"""
        return self.output_layers

    def layer(self, layer_id):
        """Return the layer with the given id"""
        return self.__layers_by_id[layer_id]

    @property
    def layers(self):
        """Return an iterator over all the layers"""
        for layer_id in nx.topological_sort(self.__graph):
            yield self.__layers_by_id[layer_id]

    def predecessors(self, layer):
        """Return an iterator over the layers with outbound connections into the layer"""
        if isinstance(layer, Layer):
            layer = id(layer)
        for node_id in self.__graph.predecessors(layer):
            yield self.__layers_by_id[node_id]

    def successors(self, layer):
        """Return an iterator over the layers with an inbound connection from the layer"""
        if isinstance(layer, Layer):
            layer = id(layer)
        for node_id in self.__graph.successors(layer):
            yield self.__layers_by_id[node_id]

    def __str__(self):
        return f"NetworkDefinition(num_layers={len(self.__layers_by_id)})"


def gnn_layer_definition(net, N, gnn_layers):
    """
    Replace dense layers in a NetworkDefinition class with GNN layers

    Parameters
    ----------
    net : NetworkDefinition
        the neural network definition
    N : int
        the number of nodes in the graph structure
    gnn_layers : list
        the list of indexes for GNN layers
    """
    assert isinstance(N, int)
    assert N > 0
    assert isinstance(gnn_layers, list)

    # copy unchanged properties from net
    gnn_net = NetworkDefinition(
        scaling_object=net.scaling_object,
        scaled_input_bounds=net.scaled_input_bounds,
        unscaled_input_bounds=net.unscaled_input_bounds,
    )

    # map the indexes of layers in net to the indexes of layers in gnn_net
    layer_id_mapper = {}

    for layer_id, layer in enumerate(net.layers):
        # these layers are not GNN layers and do not need to be changed
        if layer_id not in gnn_layers:
            new_layer = layer
        else:
            # only dense layers could be replaced with GNN layers
            assert isinstance(layer, DenseLayer)
            # define the corresponding GNN layer
            new_layer = GNNLayer(
                input_size=layer.input_size,
                output_size=layer.output_size,
                weights=layer.weights,
                biases=layer.biases,
                N=N,
                activation=layer.activation,
                input_index_mapper=layer.input_index_mapper,
            )
        # add new_layer to gnn_net
        gnn_net.add_layer(new_layer)
        # map layer (in net) to new_layer (in gnn_net)
        layer_id_mapper[id(layer)] = id(new_layer)
        # add edges between new_layer and its predecessors, which could be retrieved from the predecessors of layer
        for layer_input in net.predecessors(layer):
            gnn_net.add_edge(gnn_net.layer(layer_id_mapper[id(layer_input)]), new_layer)
    return gnn_net
