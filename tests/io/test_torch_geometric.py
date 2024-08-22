import numpy as np
import pyomo.environ as pyo
import pytest

from omlt import OmltBlock
from omlt.dependencies import (
    torch_available,
    torch_geometric_available,
)

if torch_available and torch_geometric_available:
    from torch.nn import Linear, ReLU, Sigmoid, Tanh
    from torch_geometric.nn import (
        GCNConv,
        SAGEConv,
        Sequential,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )

    from omlt.io.torch_geometric import (
        gnn_with_fixed_graph,
        gnn_with_non_fixed_graph,
        load_torch_geometric_sequential,
    )


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def gcn_sequential(activation, pooling):
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
def sage_sequential(activation, pooling, aggr, root_weight):
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
    assert len(layers) == 7  # noqa: PLR2004
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
    assert m.nvariables() == 282  # noqa: PLR2004
    assert m.nconstraints() == 614  # noqa: PLR2004


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
    assert m.nvariables() == 282  # noqa: PLR2004
    assert m.nconstraints() == 620  # noqa: PLR2004


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_torch_geometric_reader():
    for activation in [ReLU, Sigmoid, Tanh]:
        for pooling in [global_mean_pool, global_add_pool]:
            nn = gcn_sequential(activation, pooling)
            _test_torch_geometric_reader(nn, activation, pooling)
            for aggr in ["sum", "mean"]:
                for root_weight in [False, True]:
                    nn = sage_sequential(activation, pooling, aggr, root_weight)
                    _test_torch_geometric_reader(nn, activation, pooling)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_gnn_with_fixed_graph():
    for pooling in [global_mean_pool, global_add_pool]:
        nn = gcn_sequential(ReLU, pooling)
        _test_gnn_with_fixed_graph(nn)
        for aggr in ["sum", "mean"]:
            for root_weight in [False, True]:
                nn = sage_sequential(ReLU, pooling, aggr, root_weight)
                _test_gnn_with_fixed_graph(nn)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_gnn_with_non_fixed_graph():
    for pooling in [global_mean_pool, global_add_pool]:
        for aggr in ["sum"]:
            for root_weight in [False, True]:
                nn = sage_sequential(ReLU, pooling, aggr, root_weight)
                _test_gnn_with_non_fixed_graph(nn)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def _test_gnn_value_error(nn, error_info, error_type="ValueError"):
    N = 4
    F = 2

    input_size = [N * F]
    input_bounds = {}
    for i in range(input_size[0]):
        input_bounds[(i)] = (-1.0, 1.0)
    if error_type == "ValueError":
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011
            load_torch_geometric_sequential(
                nn=nn,
                N=N,
                A=None,
                scaled_input_bounds=input_bounds,
            )
        assert str(excinfo.value) == error_info
    elif error_type == "warns":
        with pytest.warns() as record:
            load_torch_geometric_sequential(
                nn=nn,
                N=N,
                A=None,
                scaled_input_bounds=input_bounds,
            )
        assert str(record[0].message) == error_info


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="Test only valid when torch and torch_geometric are available",
)
def test_gnn_value_error():
    nn = sage_sequential(ReLU, global_max_pool, "mean", root_weight=True)
    _test_gnn_value_error(nn, "Operation global_max_pool is not supported.")

    nn = sage_sequential(Sigmoid, global_mean_pool, "sum", root_weight=True)
    _test_gnn_value_error(nn, "nonlinear activation results in a MINLP", "warns")

    nn = sage_sequential(ReLU, global_mean_pool, "mean", root_weight=True)
    _test_gnn_value_error(
        nn, "this aggregation is not supported when the graph is not fixed"
    )

    nn = gcn_sequential(ReLU, global_mean_pool)
    _test_gnn_value_error(
        nn, "this layer is not supported when the graph is not fixed."
    )
