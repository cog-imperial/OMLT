import pytest
import numpy as np
import pyomo.environ as pyo
from omlt import OmltBlock

from omlt.dependencies import (
    torch,
    torch_available,
    torch_geometric,
    torch_geometric_available,
)

if torch_available and torch_geometric_available:
    from torch.nn import Linear, ReLU, Sigmoid, Softplus, Tanh
    from torch_geometric.nn import Sequential, GCNConv, SAGEConv
    from torch_geometric.nn import global_mean_pool, global_add_pool
    from omlt.io.torch_geometric import (
        load_torch_geometric_sequential,
        gnn_with_fixed_graph,
        gnn_with_non_fixed_graph,
    )


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def GCN_Sequential(activation, pooling):
    return Sequential(
        "x, edge_index",
        [
            (GCNConv(2, 4), "x, edge_index -> x"),
            activation(),
            (GCNConv(4, 4), "x, edge_index -> x"),
            activation(),
            Linear(4, 4),
            (pooling, "x, None -> x"),
            Linear(4, 2),
            activation(),
            Linear(2, 1),
        ],
    )


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def SAGE_Sequential(activation, pooling, aggr, root_weight):
    return Sequential(
        "x, edge_index",
        [
            (SAGEConv(2, 4, aggr=aggr, root_weight=root_weight), "x, edge_index -> x"),
            activation(),
            (SAGEConv(4, 4, aggr=aggr, root_weight=root_weight), "x, edge_index -> x"),
            activation(),
            Linear(4, 4),
            (pooling, "x, None -> x"),
            Linear(4, 2),
            activation(),
            Linear(2, 1),
        ],
    )


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_torch_geometric_reader(nn, activation, pooling):
    N = 4
    A = np.ones((N, N), dtype=int)
    net = load_torch_geometric_sequential(nn, N, A)
    layers = list(net.layers)
    assert len(layers) == 7
    assert layers[1].weights.shape == (8, 16)
    assert layers[2].weights.shape == (16, 16)
    assert layers[3].weights.shape == (16, 16)
    assert layers[4].weights.shape == (16, 4)
    assert layers[5].weights.shape == (4, 2)
    assert layers[6].weights.shape == (2, 1)
    for layer_id in [1, 2, 5]:
        if activation == ReLU:
            assert layers[layer_id].activation == "relu"
        elif activation == Sigmoid:
            assert layers[layer_id].activation == "sigmoid"
        elif activation == Tanh:
            assert layers[layer_id].activation == "tanh"

    if pooling == global_mean_pool:
        assert sum(sum(layers[4].weights)) == N
    elif pooling == global_add_pool:
        assert sum(sum(layers[4].weights)) == N**2


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_gnn_with_fixed_graph(nn):
    N = 4
    F = 2

    input_size = [N * F]
    input_bounds = {}
    for i in range(input_size[0]):
        input_bounds[(i)] = (-1.0, 1.0)
    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    A = np.eye(N, dtype=int)
    gnn_with_fixed_graph(m.nn, nn, N, A, scaled_input_bounds=input_bounds)
    assert m.nvariables() == 282
    assert m.nconstraints() == 614


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_gnn_with_non_fixed_graph(nn):
    N = 4
    F = 2

    input_size = [N * F]
    input_bounds = {}
    for i in range(input_size[0]):
        input_bounds[(i)] = (-1.0, 1.0)
    m = pyo.ConcreteModel()
    m.nn = OmltBlock()
    gnn_with_non_fixed_graph(m.nn, nn, N, scaled_input_bounds=input_bounds)
    assert m.nvariables() == 282
    assert m.nconstraints() == 620


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_torch_geometric_reader():
    for activation in [ReLU, Sigmoid, Tanh]:
        for pooling in [global_mean_pool, global_add_pool]:
            nn = GCN_Sequential(activation, pooling)
            _test_torch_geometric_reader(nn, activation, pooling)
            for aggr in ["sum", "mean"]:
                for root_weight in [False, True]:
                    nn = SAGE_Sequential(activation, pooling, aggr, root_weight)
                    _test_torch_geometric_reader(nn, activation, pooling)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_gnn_with_fixed_graph():
    for pooling in [global_mean_pool, global_add_pool]:
        nn = GCN_Sequential(ReLU, pooling)
        _test_gnn_with_fixed_graph(nn)
        for aggr in ["sum", "mean"]:
            for root_weight in [False, True]:
                nn = SAGE_Sequential(ReLU, pooling, aggr, root_weight)
                _test_gnn_with_fixed_graph(nn)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_gnn_with_non_fixed_graph():
    for pooling in [global_mean_pool, global_add_pool]:
        for aggr in ["sum"]:
            for root_weight in [False, True]:
                nn = SAGE_Sequential(ReLU, pooling, aggr, root_weight)
                _test_gnn_with_non_fixed_graph(nn)
