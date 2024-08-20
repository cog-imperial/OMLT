import numpy as np
import pyomo.environ as pyo

from omlt.io.torch_geometric.torch_geometric_reader import (
    load_torch_geometric_sequential,
)
from omlt.neuralnet import FullSpaceNNFormulation


def gnn_with_non_fixed_graph(  # noqa: PLR0913
    block,
    nn,
    N,
    scaling_object=None,
    scaled_input_bounds=None,
    unscaled_input_bounds=None,
):
    """Graph neural network with non-fixed graph.

    Build formulation for a torch_geometric graph neural network model (built with
    Sequential). Since the input graph is not fixed, the elements in adjacency matrix
    are decision variables.

    Parameters
    ----------
    block : Block
        the Pyomo block
    nn : torch_geometric.model
        A torch_geometric model that was built with Sequential
    N : int
        The number of nodes of input graph
    scaling_object : instance of ScalingInterface or None
        Provide an instance of a scaling object to use to scale iputs --> scaled_inputs
        and scaled_outputs --> outputs. If None, no scaling is performed. See
        scaling.py.
    scaled_input_bounds : dict or None
        A dict that contains the bounds on the scaled variables (the
        direct inputs to the neural network). If None, then no bounds
        are specified or they are generated using unscaled bounds.
    unscaled_input_bounds : dict or None
        A dict that contains the bounds on the unscaled variables (the
        direct inputs to the neural network). If specified the scaled_input_bounds
        dictionary will be generated using the provided scaling object.
        If None, then no bounds are specified.

    Returns:
    -------
    OmltBlock (formulated)
    """
    # build NetworkDefinition for nn
    net = load_torch_geometric_sequential(
        nn=nn,
        N=N,
        A=None,
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_input_bounds,
        unscaled_input_bounds=unscaled_input_bounds,
    )

    # define binary variables for adjacency matrix
    block.A = pyo.Var(
        pyo.Set(initialize=range(N)),
        pyo.Set(initialize=range(N)),
        within=pyo.Binary,
    )
    # assume that the self contribution always exists
    for u in range(N):
        block.A[u, u].fix(1)
    # assume the adjacency matrix is always symmetric
    block.symmetric_adjacency = pyo.ConstraintList()
    for u in range(N):
        for v in range(u + 1, N):
            block.symmetric_adjacency.add(block.A[u, v] == block.A[v, u])

    # build formulation for GNN
    block.build_formulation(FullSpaceNNFormulation(net))

    return block


def gnn_with_fixed_graph(  # noqa: PLR0913
    block,
    nn,
    N,
    A,
    scaling_object=None,
    scaled_input_bounds=None,
    unscaled_input_bounds=None,
):
    """Graph neural network with non-fixed graph.

    Build formulation for a torch_geometric graph neural network model (built with
    Sequential). Given the adjacency matrix, the input graph structure is fixed.

    Parameters
    ----------
    block : Block
        the Pyomo block
    nn : torch_geometric.model
        A torch_geometric model that was built with Sequential
    N : int
        The number of nodes of input graph
    A : matrix-like
        The adjacency matrix of input graph
    scaling_object : instance of ScalingInterface or None
        Provide an instance of a scaling object to use to scale iputs --> scaled_inputs
        and scaled_outputs --> outputs. If None, no scaling is performed. See
        scaling.py.
    scaled_input_bounds : dict or None
        A dict that contains the bounds on the scaled variables (the
        direct inputs to the neural network). If None, then no bounds
        are specified or they are generated using unscaled bounds.
    unscaled_input_bounds : dict or None
        A dict that contains the bounds on the unscaled variables (the
        direct inputs to the neural network). If specified the scaled_input_bounds
        dictionary will be generated using the provided scaling object.
        If None, then no bounds are specified.

    Returns:
    -------
    OmltBlock (formulated)
    """
    # assume the adjacency matrix is always symmetric
    if not np.array_equal(A, np.transpose(A)):
        msg = (
            f"Adjacency matrix A of the input graph must be symmetrical. {A} was"
            " provided."
        )
        raise ValueError(msg)

    # build NetworkDefinition for nn
    net = load_torch_geometric_sequential(
        nn=nn,
        N=N,
        A=A,
        scaling_object=scaling_object,
        scaled_input_bounds=scaled_input_bounds,
        unscaled_input_bounds=unscaled_input_bounds,
    )

    # define binary variables for adjacency matrix
    block.A = pyo.Var(
        pyo.Set(initialize=range(N)),
        pyo.Set(initialize=range(N)),
        within=pyo.Binary,
    )
    # fix A using given values
    for u in range(N):
        for v in range(N):
            block.A[u, v].fix(A[u, v])

    # assume that the self contribution always exists
    for u in range(N):
        block.A[u, u].fix(1)

    # build formulation for GNN
    block.build_formulation(FullSpaceNNFormulation(net))

    return block
