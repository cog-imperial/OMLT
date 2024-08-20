import pytest

from omlt.dependencies import onnx, onnx_available

NUM_LAYERS_131 = 3
NUM_LAYERS_GEMM = 4
NUM_LAYERS_MAXPOOL = 4
NUM_LAYERS_BIG = 5

MAXPOOL_KERNEL_DEPTH = 3

NEAR_EQUAL = 1e-05

if onnx_available:
    from omlt.io.onnx import load_onnx_neural_network
    from omlt.io.onnx_parser import NetworkParser


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    for layer in layers:
        assert layer.activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131_relu(datadir):
    model = onnx.load(datadir.file("keras_linear_131_relu.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    assert layers[1].activation == "relu"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131_sigmoid(datadir):
    model = onnx.load(datadir.file("keras_linear_131_sigmoid.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_131
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_gemm(datadir):
    model = onnx.load(datadir.file("gemm.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_GEMM
    assert layers[1].weights.shape == (784, 75)
    assert layers[2].weights.shape == (75, 75)
    assert layers[3].weights.shape == (75, 10)
    assert layers[1].activation == "relu"
    assert layers[2].activation == "relu"
    assert layers[3].activation == "logsoftmax"


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_gemm_trans_b(datadir):
    model = onnx.load(datadir.file("gemm_not_transB.onnx"))
    model_transB = onnx.load(datadir.file("gemm_transB.onnx"))
    net = load_onnx_neural_network(model)
    net_transB = load_onnx_neural_network(model_transB)
    layers = list(net.layers)
    layers_transB = list(net_transB.layers)
    assert len(layers) == len(layers_transB)
    assert layers[1].weights.shape == layers_transB[1].weights.shape
    assert abs(layers[1].weights[0][0] - layers_transB[1].weights[0][0]) < NEAR_EQUAL
    assert abs(layers[1].weights[0][1] - layers_transB[1].weights[1][0]) < NEAR_EQUAL
    assert abs(layers[1].weights[1][0] - layers_transB[1].weights[0][1]) < NEAR_EQUAL
    assert abs(layers[1].weights[1][1] - layers_transB[1].weights[1][1]) < NEAR_EQUAL


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_conv(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_GEMM
    assert layers[1].activation == "linear"
    assert layers[2].activation == "linear"
    assert layers[3].activation == "relu"
    assert layers[1].strides == [1, 1]
    assert layers[1].kernel_shape == (2, 2)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_maxpool(datadir):
    model = onnx.load(datadir.file("maxpool_2d.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == NUM_LAYERS_MAXPOOL
    assert layers[1].activation == "relu"
    assert layers[2].activation == "linear"
    assert layers[3].activation == "linear"
    assert layers[1].kernel_shape == (2, 3)
    assert layers[2].kernel_shape == (1, 2)
    assert layers[3].kernel_shape == (4, 2)
    assert layers[1].strides == [1, 1]
    assert layers[2].strides == [1, 2]
    assert layers[3].strides == [3, 1]
    assert layers[1].output_size == [3, 5, 5]
    assert layers[2].output_size == [3, 5, 2]
    assert layers[3].output_size == [3, 2, 1]
    for layer in layers[1:]:
        assert layer.kernel_depth == MAXPOOL_KERNEL_DEPTH


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_input_tensor_invalid_dims(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 0
    parser = NetworkParser()
    with pytest.raises(
        ValueError, match='All dimensions in graph "tf2onnx" input tensor have 0 value.'
    ):
        parser.parse_network(model.graph, None, None)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_no_input_layers(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    model.graph.input.remove(model.graph.input[0])
    parser = NetworkParser()
    with pytest.raises(
        ValueError, match='No valid input layer found in graph "tf2onnx".'
    ):
        parser.parse_network(model.graph, None, None)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_node_no_inputs(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    while len(model.graph.node[0].input) > 0:
        model.graph.node[0].input.pop()
    parser = NetworkParser()
    expected_msg = (
        'Nodes must have inputs or have op_type "Constant". Node '
        '"StatefulPartitionedCall/keras_linear_131/dense/MatMul" has'
        ' no inputs and op_type "MatMul".'
    )
    with pytest.raises(ValueError, match=expected_msg):
        parser.parse_network(model.graph, None, None)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_wrong_node_type(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)

    expected_msg_dense = (
        "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, "
        "but the parsing method for MatMul nodes was called. This could indicate "
        "changes in the network being parsed."
    )
    with pytest.raises(ValueError, match=expected_msg_dense):
        parser._consume_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_gemm = (
        "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, "
        "but the parsing method for Gemm nodes was called. This could indicate "
        "changes in the network being parsed."
    )
    with pytest.raises(ValueError, match=expected_msg_gemm):
        parser._consume_gemm_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_conv = (
        "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, "
        "but the parsing method for Conv nodes was called. This could indicate "
        "changes in the network being parsed."
    )
    with pytest.raises(ValueError, match=expected_msg_conv):
        parser._consume_conv_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_reshape = (
        "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, "
        "but the parsing method for Reshape nodes was called. This could indicate "
        "changes in the network being parsed."
    )
    with pytest.raises(ValueError, match=expected_msg_reshape):
        parser._consume_reshape_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_pool = (
        "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, "
        "but the parsing method for MaxPool nodes was called. This could indicate "
        "changes in the network being parsed."
    )
    with pytest.raises(ValueError, match=expected_msg_pool):
        parser._consume_pool_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_dense_wrong_dims(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)

    parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][
        1
    ].input.append("abcd")
    expected_msg_dense = (
        "StatefulPartitionedCall/keras_linear_131/dense/MatMul input has 3 dimensions, "
        "only nodes with 2 input dimensions can be used as starting points for parsing."
    )
    with pytest.raises(ValueError, match=expected_msg_dense):
        parser._consume_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][2],
        )


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_gemm_wrong_dims(datadir):
    model = onnx.load(datadir.file("gemm.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Gemm_0"][1].input.append("abcd")
    expected_msg_gemm = (
        "Gemm_0 input has 4 dimensions, only nodes with 3 input dimensions "
        "can be used as starting points for parsing."
    )
    with pytest.raises(ValueError, match=expected_msg_gemm):
        parser._consume_gemm_dense_nodes(
            parser._nodes["Gemm_0"][1], parser._nodes["Gemm_0"][2]
        )


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_conv_wrong_dims(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Conv_0"][1].input.append("abcd")
    expected_msg_conv = (
        "Conv_0 input has 4 dimensions, only nodes with 2 or 3 input"
        " dimensions can be used as starting points for parsing."
    )
    with pytest.raises(ValueError, match=expected_msg_conv):
        parser._consume_conv_nodes(
            parser._nodes["Conv_0"][1], parser._nodes["Conv_0"][2]
        )


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_reshape_wrong_dims(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Reshape_2"][1].input.append("abcd")
    expected_msg_reshape = (
        "Reshape_2 input has 3 dimensions, only nodes with 2 input"
        " dimensions can be used as starting points for parsing."
    )
    with pytest.raises(ValueError, match=expected_msg_reshape):
        parser._consume_reshape_nodes(
            parser._nodes["Reshape_2"][1], parser._nodes["Reshape_2"][2]
        )


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_maxpool_wrong_dims(datadir):
    model = onnx.load(datadir.file("maxpool_2d.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["node1"][1].input.append("abcd")
    expected_msg_maxpool = (
        "node1 input has 2 dimensions, only nodes with 1 input "
        "dimension can be used as starting points for parsing."
    )
    with pytest.raises(ValueError, match=expected_msg_maxpool):
        parser._consume_pool_nodes(parser._nodes["node1"][1], parser._nodes["node1"][2])
