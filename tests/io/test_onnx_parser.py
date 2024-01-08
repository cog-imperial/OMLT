import pytest

from omlt.dependencies import onnx, onnx_available

if onnx_available:
    from omlt.io.onnx import load_onnx_neural_network
    from omlt.io.onnx_parser import NetworkParser
    from onnx import numpy_helper
    from numpy import array


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 3
    for layer in layers:
        assert layer.activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131_relu(datadir):
    model = onnx.load(datadir.file("keras_linear_131_relu.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 3
    assert layers[1].activation == "relu"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_linear_131_sigmoid(datadir):
    model = onnx.load(datadir.file("keras_linear_131_sigmoid.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 3
    assert layers[1].activation == "sigmoid"
    assert layers[2].activation == "linear"
    assert layers[1].weights.shape == (1, 3)
    assert layers[2].weights.shape == (3, 1)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_gemm(datadir):
    model = onnx.load(datadir.file("gemm.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 4
    assert layers[1].weights.shape == (784, 75)
    assert layers[2].weights.shape == (75, 75)
    assert layers[3].weights.shape == (75, 10)
    assert layers[1].activation == "relu"
    assert layers[2].activation == "relu"
    assert layers[3].activation == "logsoftmax"


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_gemm_transB(datadir):
    model = onnx.load(datadir.file("gemm_not_transB.onnx"))
    model_transB = onnx.load(datadir.file("gemm_transB.onnx"))
    net = load_onnx_neural_network(model)
    net_transB = load_onnx_neural_network(model_transB)
    layers = list(net.layers)
    layers_transB = list(net_transB.layers)
    assert len(layers) == len(layers_transB)
    assert layers[1].weights.shape == layers_transB[1].weights.shape
    assert abs(layers[1].weights[0][0] - layers_transB[1].weights[0][0]) < 1e-05
    assert abs(layers[1].weights[0][1] - layers_transB[1].weights[1][0]) < 1e-05
    assert abs(layers[1].weights[1][0] - layers_transB[1].weights[0][1]) < 1e-05
    assert abs(layers[1].weights[1][1] - layers_transB[1].weights[1][1]) < 1e-05


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_conv(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    for attr in model.graph.node[0].attribute:
        if attr.name == "dilations":
            del attr
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 4
    assert layers[1].activation == "linear"
    assert layers[2].activation == "linear"
    assert layers[3].activation == "relu"
    assert layers[1].strides == [1, 1]
    assert layers[1].kernel_shape == (2, 2)
    assert layers[1].dilations == [1, 1]
    assert layers[1].pads == [0, 0, 0, 0]


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_conv_dilations(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    for attr in model.graph.node[0].attribute:
        if attr.name == "dilations":
            del attr.ints[:]
            attr.ints.extend([2, 2])
        if attr.name == "pads":
            del attr.ints[:]
            attr.ints.extend([1, 2, 1, 0])
    model.graph.node[1].attribute[0].t.raw_data = numpy_helper.from_array(
        array([-1, 128])
    ).raw_data
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert layers[1].dilations == [2, 2]
    assert (
        layers[1].dilated_kernel[0][0].round(8)
        == array(
            [[-0.00886667, 0, 0.18750042], [0, 0, 0], [-0.11404419, 0, -0.02588665]]
        )
    ).all()
    assert (
        layers[1].dilated_kernel[1][0].round(8)
        == array([[-0.07554907, 0, -0.05939162], [0, 0, 0], [0.2217437, 0, 0.14637864]])
    ).all()
    assert layers[1].pads == [1, 2, 1, 0]


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_maxpool(datadir):
    model = onnx.load(datadir.file("maxpool_2d.onnx"))
    net = load_onnx_neural_network(model)
    layers = list(net.layers)
    assert len(layers) == 4
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
        assert layer.kernel_depth == 3


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_input_tensor_invalid_dims(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 0
    parser = NetworkParser()
    with pytest.raises(ValueError) as excinfo:
        parser.parse_network(model.graph, None, None)
    expected_msg = 'All dimensions in graph "tf2onnx" input tensor have 0 value.'
    assert str(excinfo.value) == expected_msg


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_no_input_layers(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    model.graph.input.remove(model.graph.input[0])
    parser = NetworkParser()
    with pytest.raises(ValueError) as excinfo:
        parser.parse_network(model.graph, None, None)
    expected_msg = 'No valid input layer found in graph "tf2onnx".'
    assert str(excinfo.value) == expected_msg


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_node_no_inputs(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    while len(model.graph.node[0].input) > 0:
        model.graph.node[0].input.pop()
    parser = NetworkParser()
    with pytest.raises(ValueError) as excinfo:
        parser.parse_network(model.graph, None, None)
    expected_msg = """Nodes must have inputs or have op_type \"Constant\". Node \"StatefulPartitionedCall/keras_linear_131/dense/MatMul\" has no inputs and op_type \"MatMul\"."""
    assert str(excinfo.value) == expected_msg


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_wrong_node_type(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)

    with pytest.raises(ValueError) as excinfo:
        parser._consume_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_dense = "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, but the method for parsing MatMul nodes was invoked."
    assert str(excinfo.value) == expected_msg_dense

    with pytest.raises(ValueError) as excinfo:
        parser._consume_gemm_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_gemm = "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, but the method for parsing Gemm nodes was invoked."
    assert str(excinfo.value) == expected_msg_gemm

    with pytest.raises(ValueError) as excinfo:
        parser._consume_conv_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_conv = "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, but the method for parsing Conv nodes was invoked."
    assert str(excinfo.value) == expected_msg_conv

    with pytest.raises(ValueError) as excinfo:
        parser._consume_reshape_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_reshape = "StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, but the method for parsing Reshape nodes was invoked."
    assert str(excinfo.value) == expected_msg_reshape

    with pytest.raises(ValueError) as excinfo:
        parser._consume_pool_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/BiasAdd"][2],
        )
    expected_msg_pool = """StatefulPartitionedCall/keras_linear_131/dense/BiasAdd is a Add node, but the method for parsing MaxPool nodes was invoked."""
    assert str(excinfo.value) == expected_msg_pool


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_dense_wrong_dims(datadir):
    model = onnx.load(datadir.file("keras_linear_131.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)

    parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][
        1
    ].input.append("abcd")
    with pytest.raises(ValueError) as excinfo:
        parser._consume_dense_nodes(
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][1],
            parser._nodes["StatefulPartitionedCall/keras_linear_131/dense/MatMul"][2],
        )
    expected_msg_dense = "StatefulPartitionedCall/keras_linear_131/dense/MatMul input has 3 dimensions, but the parser requires the starting node to have 2 input dimensions."
    assert str(excinfo.value) == expected_msg_dense


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_gemm_wrong_dims(datadir):
    model = onnx.load(datadir.file("gemm.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Gemm_0"][1].input.append("abcd")
    with pytest.raises(ValueError) as excinfo:
        parser._consume_gemm_dense_nodes(
            parser._nodes["Gemm_0"][1], parser._nodes["Gemm_0"][2]
        )
    expected_msg_gemm = "Gemm_0 input has 4 dimensions, but the parser requires the starting node to have 3 input dimensions."
    assert str(excinfo.value) == expected_msg_gemm


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_conv_wrong_dims(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Conv_0"][1].input.append("abcd")
    with pytest.raises(ValueError) as excinfo:
        parser._consume_conv_nodes(
            parser._nodes["Conv_0"][1], parser._nodes["Conv_0"][2]
        )
    expected_msg_conv = "Conv_0 input has 4 dimensions, but the parser requires the starting node to have 2 or 3 input dimensions."
    assert str(excinfo.value) == expected_msg_conv


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_reshape_wrong_dims(datadir):
    model = onnx.load(datadir.file("convx1_gemmx1.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["Reshape_2"][1].input.append("abcd")
    with pytest.raises(ValueError) as excinfo:
        parser._consume_reshape_nodes(
            parser._nodes["Reshape_2"][1], parser._nodes["Reshape_2"][2]
        )
    expected_msg_reshape = """Reshape_2 input has 3 dimensions, but the parser requires the starting node to have 2 input dimensions."""
    assert str(excinfo.value) == expected_msg_reshape


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_consume_maxpool_wrong_dims(datadir):
    model = onnx.load(datadir.file("maxpool_2d.onnx"))
    parser = NetworkParser()
    parser.parse_network(model.graph, None, None)
    parser._nodes["node1"][1].input.append("abcd")
    with pytest.raises(ValueError) as excinfo:
        parser._consume_pool_nodes(parser._nodes["node1"][1], parser._nodes["node1"][2])
    expected_msg_maxpool = """node1 input has 2 dimensions, but the parser requires the starting node to have 1 input dimension."""
    assert str(excinfo.value) == expected_msg_maxpool
