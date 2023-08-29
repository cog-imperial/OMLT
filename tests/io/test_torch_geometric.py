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
    from torch_geometric.utils import erdos_renyi_graph
    from omlt.io import (
        load_torch_geometric_sequential,
        gnn_with_fixed_graph,
        gnn_with_non_fixed_graph,
    )


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
    x = 1.0 * torch.randint(1, (N, F))
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
def _test_gnn_with_fixed_graph(nn):
    N = 4
    F = 2
    nn.eval()
    for p in range(10):
        x, edges, x_np, A = generate_random_inputs(N, F, 0, p / 10.0)
        y = nn(x, edges).detach().numpy()
        input_size = [N * F]
        input_bounds = {}
        for i in range(input_size[0]):
            input_bounds[(i)] = (0.0, 1.0)
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        gnn_with_fixed_graph(m.nn, nn, N, A, scaled_input_bounds=input_bounds)
        for i in range(N * F):
            m.nn.inputs[i].fix(x_np[i])
        m.obj = pyo.Objective(expr=m.nn.outputs[0])
        status = pyo.SolverFactory("cbc").solve(m, tee=False)
        assert abs(pyo.value(m.nn.outputs[0]) - y) < 1e-6


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_gnn_with_non_fixed_graph(nn):
    N = 4
    F = 2
    nn.eval()
    for p in range(10):
        x, edges, x_np, A = generate_random_inputs(N, F, 0, p / 10.0)
        y = nn(x, edges).detach().numpy()
        input_size = [N * F]
        input_bounds = {}
        for i in range(input_size[0]):
            input_bounds[(i)] = (-1.0, 1.0)
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        gnn_with_non_fixed_graph(m.nn, nn, N, scaled_input_bounds=input_bounds)
        for i in range(N * F):
            m.nn.inputs[i].fix(x_np[i])
        for u in range(N):
            for v in range(N):
                if u != v:
                    m.nn.A[u, v].fix(A[u, v])
        m.obj = pyo.Objective(expr=m.nn.outputs[0])
        status = pyo.SolverFactory("cbc").solve(m, tee=False)
        assert abs(pyo.value(m.nn.outputs[0]) - y) < 1e-6


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
