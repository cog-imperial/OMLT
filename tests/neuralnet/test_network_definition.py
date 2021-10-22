import pytest

from omlt.neuralnet.network_definition import NetworkDefinition


# ToDo: Add tests for teh scaling object
def test_network_definition():
    """
    Test of the following model:

           1           
     (0) ------------ (2) ----\
                \              \ 2
                 \              \
                  \ -2           (4)
                   \            /
                    \          / -3
          -1         \        /
     (1) ------------ (3) ---/
    """

    n_inputs = 2
    n_hidden = 2
    n_outputs = 1
    w = {2: {0: 1.0}, 3: {0: -2.0, 1: -1.0}, 4: {2: 2.0, 3: -3}}
    b = {2: 1, 3: 2, 4: 3}
    a = {2: "linear", 3: "linear", 4: "linear"}

    nd = NetworkDefinition(n_inputs, n_hidden, n_outputs, w, b, a)
    assert nd.n_inputs == n_inputs
    assert nd.n_hidden == n_hidden
    assert nd.n_outputs == n_outputs
    for j in w.keys():
        for i in w[j].keys():
            assert w[j][i] == nd.weights[j][i]
        assert b[j] == nd.biases[j]
        assert a[j] == nd.activations[j]

    assert nd.scaling_object is None
    assert nd.input_node_ids() == [0, 1]
    assert nd.hidden_node_ids() == [2, 3]
    assert nd.output_node_ids() == [4]

    with pytest.warns(
        UserWarning,
        match="No input bounds were provided. This may lead to extrapolation outside of the training data",
    ):
        nd = NetworkDefinition(n_inputs, n_hidden, n_outputs, w, b, a)

    input_bounds = [(0, 2), (-1, 1)]
    nd = NetworkDefinition(
        n_inputs, n_hidden, n_outputs, w, b, a, input_bounds=input_bounds
    )
    assert nd.input_bounds == input_bounds

    with pytest.raises(ValueError):
        input_bounds = [
            (0, 2),
        ]
        nd = NetworkDefinition(
            n_inputs, n_hidden, n_outputs, w, b, a, input_bounds=input_bounds
        )

    with pytest.raises(ValueError):
        input_bounds = [(0, 2), (3,)]
        nd = NetworkDefinition(
            n_inputs, n_hidden, n_outputs, w, b, a, input_bounds=input_bounds
        )

    # Todo: this should be able to throw an error
    nd = NetworkDefinition(1, n_hidden, n_outputs, w, b, a)

    with pytest.raises(ValueError):
        nd = NetworkDefinition(n_inputs, 1, n_outputs, w, b, a)
