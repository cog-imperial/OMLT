import pytest
import numpy as np

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
    from torch_geometric.utils import erdos_renyi_graph
    from omlt.io.torch_geometric import load_torch_geometric_sequential


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def GCN_Sequential(activation, pooling):
    torch.manual_seed(123)
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
    torch.manual_seed(123)
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
def generate_random_inputs(N, F, seed, p):
    torch.manual_seed(seed)
    edges = erdos_renyi_graph(N, p, directed=False)
    A = np.zeros((N, N), dtype=int)
    for k in range(edges.shape[1]):
        u = edges[0, k].numpy()
        v = edges[1, k].numpy()
        A[u, v] = 1
    x = 2.0 * torch.rand((N, F)) - 1.0
    return x, edges, np.squeeze(x.numpy().reshape(1, -1)), A


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_torch_geometric_reader(nn):
    N = 4
    F = 2
    nn.eval()
    for seed in range(10):
        for p in range(10):
            x, edges, x_np, A = generate_random_inputs(N, F, seed, p / 10.0)
            net = load_torch_geometric_sequential(nn, N, A)
            y = nn(x, edges).detach().numpy()
            y_np = x_np
            for layer in net.layers:
                y_np = layer.eval_single_layer(y_np)
            assert abs(y - y_np) < 1e-6


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_torch_geometric_reader():
    for activation in [ReLU, Sigmoid, Tanh]:
        for pooling in [global_mean_pool, global_add_pool]:
            nn = GCN_Sequential(activation, pooling)
            _test_torch_geometric_reader(nn)
            for aggr in ["sum", "mean"]:
                for root_weight in [False, True]:
                    nn = SAGE_Sequential(activation, pooling, aggr, root_weight)
                    _test_torch_geometric_reader(nn)
