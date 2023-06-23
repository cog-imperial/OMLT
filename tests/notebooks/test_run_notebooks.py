import os

import nbformat
import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook

from omlt.dependencies import keras_available, onnx_available


# return testbook for given notebook
def open_book(folder, notebook_fname, **kwargs):
    execute = kwargs.get("execute", True)
    os.chdir(os.path.join(this_file_dir(), "..", "..", "docs", "notebooks", folder))
    book = testbook(notebook_fname, execute=execute, timeout=300)
    return book


# checks that the number of executed cells matches the expected
def check_cell_execution(tb, notebook_fname, **kwargs):
    injections = kwargs.get("injections", 0)
    assert (
        tb.code_cells_executed
        == cell_counter(notebook_fname, only_code_cells=True) + injections
    )


def check_layers(tb, activations, network):
    tb.inject(
        f"""
                    activations = {activations}
                    for layer_id, layer in enumerate({network}):
                        assert activations[layer_id] in str(layer.activation)
                """
    )


# counting number of cells
def cell_counter(notebook_fname, **kwargs):
    only_code_cells = kwargs.get("only_code_cells", False)
    nb = nbformat.read(notebook_fname, as_version=4)
    nb = nbformat.validator.normalize(nb)[1]
    if only_code_cells:
        total = 0
        for cell in nb.cells:
            if cell["cell_type"] == "code":
                total += 1
        return total
    else:
        return len(nb.cells)


def mnist_stats(tb, fname):
    total_cells = cell_counter(fname)
    tb.inject("test(model, test_loader)")
    model_stats = tb.cell_output_text(total_cells)
    model_stats = model_stats.split(" ")
    loss = float(model_stats[4][:-1])
    accuracy = int(model_stats[-2][:-6])
    return (loss, accuracy)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    notebook_fname = "auto-thermal-reformer-relu.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # check loss of model
        model_loss = tb.ref("nn.evaluate(x, y)")
        assert model_loss == pytest.approx(0.000389626, abs=0.00028)

        # check layers of model
        layers = ["relu", "relu", "relu", "relu", "linear"]
        check_layers(tb, layers, "nn.layers")

        # check final values
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == 0.1
        assert ngRatio == pytest.approx(1.12, abs=0.05)
        assert h2Conc == pytest.approx(0.33, abs=0.03)
        assert n2Conc == pytest.approx(0.34, abs=0.01)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_reformer():
    notebook_fname = "auto-thermal-reformer.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # check loss of model
        model_loss = tb.ref("nn.evaluate(x, y)")
        assert model_loss == pytest.approx(0.00015207, abs=0.00016)

        # check layers of model
        layers = ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "linear"]
        check_layers(tb, layers, "nn.layers")

        # check final values
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == pytest.approx(0.1, abs=0.001)
        assert ngRatio == pytest.approx(1.12, abs=0.03)
        assert h2Conc == pytest.approx(0.33, abs=0.01)
        assert n2Conc == pytest.approx(0.34, abs=0.01)


def test_build_network():
    notebook_fname = "build_network.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # check for correct layers
        layers = ["linear", "linear", "relu"]
        check_layers(tb, layers, "list(net.layers)")

        m_layers = tb.ref("list(m.neural_net.layer)")
        assert len(m_layers) == 3

        # check eval function
        eval_ex = list(tb.ref("x"))
        assert eval_ex[0] == pytest.approx(2.15)


@pytest.mark.skipif(
    (not onnx_available) or (not keras_available),
    reason="onnx and keras needed for this notebook",
)
def test_import_network():
    notebook_fname = "import_network.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=False)

    with book as tb:
        # inject cell that reads in loss and accuracy of keras model
        # TODO: add something that checks where to inject code cell instead of hardcoding
        tb.inject(
            "keras_loss, keras_accuracy = model.evaluate(X, Y)", before=25, run=False
        )
        tb.execute()

        check_cell_execution(tb, notebook_fname, injections=1)

        # check input bounds
        input_bounds = tb.ref("input_bounds")
        assert input_bounds == [
            [0.0, 17.0],
            [0.0, 199.0],
            [0.0, 122.0],
            [0.0, 99.0],
            [0.0, 846.0],
            [0.0, 67.1],
            [0.078, 2.42],
            [21.0, 81.0],
        ]

        # checking accuracy and loss of keras model
        keras_loss, keras_accuracy = tb.ref("keras_loss"), tb.ref("keras_accuracy")
        assert keras_loss == pytest.approx(5.4, abs=4.8)
        assert keras_accuracy == pytest.approx(0.48, abs=0.21)

        # checking loss of pytorch model
        pytorch_loss = tb.ref("loss.item()")
        assert pytorch_loss == pytest.approx(0.25, abs=0.1)

        # checking the model that was imported
        imported_input_bounds = tb.ref("network_definition.scaled_input_bounds")
        assert imported_input_bounds == {
            "0": [0.0, 17.0],
            "1": [0.0, 199.0],
            "2": [0.0, 122.0],
            "3": [0.0, 99.0],
            "4": [0.0, 846.0],
            "5": [0.0, 67.1],
            "6": [0.078, 2.42],
            "7": [21.0, 81.0],
        }

        # checking the imported layers
        layers = ["linear", "relu", "relu", "linear"]
        check_layers(tb, layers, "network_definition.layers")


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_convolutional():
    notebook_fname = "mnist_example_convolutional.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # checking training accuracy
        loss, accuracy = mnist_stats(tb, notebook_fname)
        assert loss == pytest.approx(0.3, abs=0.15)
        assert accuracy / 10000 == pytest.approx(0.93, abs=0.07)

        # checking the imported layers
        layers = ["linear", "relu", "relu", "relu", "linear"]
        check_layers(tb, layers, "network_definition.layers")

        # checking optimal solution
        optimal_sol = tb.ref(
            "-(pyo.value(m.nn.outputs[0,adversary]-m.nn.outputs[0,label]))"
        )
        assert optimal_sol == pytest.approx(11, abs=6.6)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_dense():
    notebook_fname = "mnist_example_dense.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # checking training accuracy
        loss, accuracy = mnist_stats(tb, notebook_fname)
        assert loss == pytest.approx(0.0867, abs=0.03)
        assert accuracy / 10000 == pytest.approx(0.95, abs=0.05)

        # checking the imported layers
        layers = ["linear", "relu", "relu", "linear"]
        check_layers(tb, layers, "network_definition.layers")

        # checking optimal solution
        optimal_sol = tb.ref(
            "-(pyo.value(m.nn.outputs[adversary]-m.nn.outputs[label]))"
        )
        assert optimal_sol == pytest.approx(5, abs=2.7)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_neural_network_formulations():
    notebook_fname = "neural_network_formulations.ipynb"
    book = open_book("neuralnet", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # checking loss of keras models
        losses = [
            tb.ref(f"nn{x + 1}.evaluate(x=df['x_scaled'], y=df['y_scaled'])")
            for x in range(3)
        ]
        assert losses[0] == pytest.approx(0.000534, abs=0.0003)
        assert losses[1] == pytest.approx(0.000691, abs=0.0003)
        assert losses[2] == pytest.approx(0.0024, abs=0.001)

        # checking scaled input bounds
        scaled_input = tb.ref("input_bounds[0]")
        assert scaled_input[0] == pytest.approx(-1.73179)
        assert scaled_input[1] == pytest.approx(1.73179)

        # checking optimal solution
        # TODO: make a helper function for all of these
        x1_reduced = tb.ref("solution_1_reduced[0]")
        y1_reduced = tb.ref("solution_1_reduced[1]")
        assert x1_reduced == pytest.approx(-0.8, abs=0.75)
        assert y1_reduced == pytest.approx(0.8, abs=0.75)

        x1_full = tb.ref("solution_1_full[0]")
        y1_full = tb.ref("solution_1_full[1]")
        assert x1_full == pytest.approx(-0.27382, abs=0.3)
        assert y1_full == pytest.approx(-0.86490, abs=0.3)

        x2_comp = tb.ref("solution_2_comp[0]")
        y2_comp = tb.ref("solution_2_comp[1]")
        assert x2_comp == pytest.approx(-0.29967, abs=0.3)
        assert y2_comp == pytest.approx(-0.84415, abs=0.3)

        x2_bigm = tb.ref("solution_2_bigm[0]")
        y2_bigm = tb.ref("solution_2_bigm[1]")
        assert x2_bigm == pytest.approx(-0.29967, abs=0.3)
        assert y2_bigm == pytest.approx(-0.84414, abs=0.3)

        x3 = tb.ref("solution_3_mixed[0]")
        y3 = tb.ref("solution_3_mixed[1]")
        assert x3 == pytest.approx(-0.23955, abs=0.3)
        assert y3 == pytest.approx(-0.90598, abs=0.3)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_bo_with_trees():
    notebook_fname = "bo_with_trees.ipynb"
    book = open_book("", notebook_fname)

    with book as tb:
        check_cell_execution(tb, notebook_fname)

        # not sure what to put here...
