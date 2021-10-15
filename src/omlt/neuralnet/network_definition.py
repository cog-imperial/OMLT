import networkx as nx


class NetworkDefinition:
    def __init__(self):
        self.__nodes_by_id = dict()
        self.__graph = nx.DiGraph()
        self.__scaling_object = None
        self.__input_bounds = None

    def add_node(self, node):
        node_id = id(node)
        self.__nodes_by_id[node_id] = node
        self.__graph.add_node(node_id)

    def add_edge(self, from_node, to_node):
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
        for node_id, in_degree in self.__graph.in_degree():
            if in_degree == 0:
                yield self.__nodes_by_id[node_id]

    @property
    def output_nodes(self):
        for node_id, out_degree in self.__graph.out_degree():
            if out_degree == 0:
                yield self.__nodes_by_id[node_id]

    @property
    def nodes(self):
        for node_id in nx.topological_sort(self.__graph):
            yield self.__nodes_by_id[node_id]

    def predecessors(self, layer):
        for node_id in self.__graph.predecessors(id(layer)):
            yield self.__nodes_by_id[node_id]

    def successors(self, layer):
        for node_id in self.__graph.successors(id(layer)):
            yield self.__nodes_by_id[node_id]

    def __str__(self):
        return f"NetworkDefinition(nodes={len(self.__nodes_by_id)})"
