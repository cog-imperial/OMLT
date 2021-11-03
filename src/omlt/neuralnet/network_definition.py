import networkx as nx


class NetworkDefinition:
    def __init__(self, scaling_object=None, input_bounds=None):
        self.__nodes_by_id = dict()
        self.__graph = nx.DiGraph()
        self.__scaling_object = scaling_object
        self.__input_bounds = input_bounds

    def add_node(self, node):
        """
        Add a node to the network.

        Parameters
        ----------
        node : Layer
            the node to add to the network
        """
        node_id = id(node)
        self.__nodes_by_id[node_id] = node
        self.__graph.add_node(node_id)

    def add_edge(self, from_node, to_node):
        """
        Add an edge between two nodes.

        Parameters
        ----------
        from_node : Layer
            the node with the outbound connection
        to_node : Layer
            the node with the inbound connection
        """
        id_to = id(to_node)
        id_from = id(from_node)
        assert id_to in self.__nodes_by_id
        assert id_from in self.__nodes_by_id
        self.__graph.add_edge(id_from, id_to)

    @property
    def scaling_object(self):
        """Return an instance of the scaling object that supports the ScalingInterface"""
        return self.__scaling_object

    @property
    def input_bounds(self):
        """Return a list of tuples containing lower and upper bounds of neural network inputs"""
        return self.__input_bounds

    @property
    def input_nodes(self):
        """Return an interator over the input nodes"""
        for node_id, in_degree in self.__graph.in_degree():
            if in_degree == 0:
                yield self.__nodes_by_id[node_id]

    @property
    def output_nodes(self):
        """Return an iterator over the output nodes"""
        for node_id, out_degree in self.__graph.out_degree():
            if out_degree == 0:
                yield self.__nodes_by_id[node_id]

    @property
    def nodes(self):
        """Return an iterator over all the nodes"""
        for node_id in nx.topological_sort(self.__graph):
            yield self.__nodes_by_id[node_id]

    def predecessors(self, layer):
        """Return an iterator over the nodes with outbound connections into the layer"""
        for node_id in self.__graph.predecessors(id(layer)):
            yield self.__nodes_by_id[node_id]

    def successors(self, layer):
        """Return an iterator over the nodes with an inbound connection from the layer"""
        for node_id in self.__graph.successors(id(layer)):
            yield self.__nodes_by_id[node_id]

    def __str__(self):
        return f"NetworkDefinition(nodes={len(self.__nodes_by_id)})"
