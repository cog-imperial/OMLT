import os
import nbformat
import numpy as np
import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook

from omlt.dependencies import keras_available, onnx_available, lightgbm_available


# return testbook for given notebook
def open_book(folder, notebook_fname, execute):
    # changes directory to the notebooks folder
    os.chdir(os.path.join(this_file_dir(), "..", "..", "docs", "notebooks", folder))
    book = testbook(notebook_fname, execute=execute, timeout=360)
    return book


# checks that the number of executed cells matches the expected
def check_cell_execution(tb, notebook_fname, injections):
    assert (
        tb.code_cells_executed
        == get_cell_count(notebook_fname, only_code_cells=True) + injections
    )


# checks for correct type and number of layers in a model
def inject_activation_check(tb, activations, network):
    # does the assertion within the notebook through injection
    tb.inject(
        f"""
                    activations = {activations}
                    for layer_id, layer in enumerate({network}):
                        assert activations[layer_id] in str(layer.activation)
                """
    )


# counting number of cells
def get_cell_count(notebook_fname, only_code_cells):
    # reads in the notebooks data
    nb = nbformat.read(notebook_fname, as_version=4)
    nb = nbformat.validator.normalize(nb)[1]
    if only_code_cells:
        # checks that cell is a a code cell and not empty
        return len(
            [
                cell
                for cell in nb.cells
                if cell["cell_type"] == "code" and len(cell["source"]) != 0
            ]
        )
    else:
        return len(nb.cells)


# gets model stats for mnist notebooks
def get_mnist_stats(tb, fname):
    total_cells = get_cell_count(fname, only_code_cells=False)

    # injects a cell at the end of the notebook that contains
    # the models final loss and accuracy values
    tb.inject("test(model, test_loader)")

    model_stats = tb.cell_output_text(total_cells)
    model_stats = model_stats.split(" ")
    loss = float(model_stats[4][:-1])
    accuracy = int(model_stats[-2][:-6])
    return (loss, accuracy)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    notebook_fname = "auto-thermal-reformer-relu.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # check loss of model
        model_loss = tb.ref("nn.evaluate(x, y)")
        assert model_loss < 0.1

        # check final values after optimization
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == pytest.approx(0.1, abs=0.01)
        assert ngRatio == pytest.approx(1.12, abs=0.1)
        assert h2Conc == pytest.approx(0.33, abs=0.05)
        assert n2Conc == pytest.approx(0.34, abs=0.05)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_reformer():
    notebook_fname = "auto-thermal-reformer.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # check loss of model
        model_loss = tb.ref("nn.evaluate(x, y)")
        assert model_loss < 0.1

        # check final values after optimization
        bypassFraction = tb.ref("pyo.value(m.reformer.inputs[0])")
        ngRatio = tb.ref("pyo.value(m.reformer.inputs[1])")
        h2Conc = tb.ref("pyo.value(m.reformer.outputs[h2_idx])")
        n2Conc = tb.ref("pyo.value(m.reformer.outputs[n2_idx])")

        assert bypassFraction == pytest.approx(0.1, abs=0.01)
        assert ngRatio == pytest.approx(1.12, abs=0.1)
        assert h2Conc == pytest.approx(0.33, abs=0.05)
        assert n2Conc == pytest.approx(0.34, abs=0.05)


def test_build_network():
    notebook_fname = "build_network.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # makes sure that there are three layers in the network
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
        # TODO: add something that checks where to inject code cell
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

        # checking that the model was imported correctly
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
        inject_activation_check(tb, layers, "network_definition.layers")


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_convolutional():
    notebook_fname = "mnist_example_convolutional.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # checking training accuracy
        loss, accuracy = get_mnist_stats(tb, notebook_fname)
        assert loss < 1
        assert accuracy / 10000 > 0.9

        # checking the imported layers
        layers = ["linear", "relu", "relu", "relu", "linear"]
        inject_activation_check(tb, layers, "network_definition.layers")

        # checking optimal solution
        optimal_sol = tb.ref(
            "-(pyo.value(m.nn.outputs[0,adversary]-m.nn.outputs[0,label]))"
        )
        assert optimal_sol == pytest.approx(10.5, abs=0.1)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_dense():
    notebook_fname = "mnist_example_dense.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # checking training accuracy
        loss, accuracy = get_mnist_stats(tb, notebook_fname)
        assert loss < 1
        assert accuracy / 10000 < 1

        # checking the imported layers
        layers = ["linear", "relu", "relu", "linear"]
        inject_activation_check(tb, layers, "network_definition.layers")

        # checking optimal solution
        optimal_sol = tb.ref(
            "-(pyo.value(m.nn.outputs[adversary]-m.nn.outputs[label]))"
        )
        assert optimal_sol == pytest.approx(5, abs=3.3)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_neural_network_formulations():
    notebook_fname = "neural_network_formulations.ipynb"
    book = open_book("neuralnet", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # checking loss of keras models
        losses = np.asarray(
            [
                tb.ref(f"nn{x + 1}.evaluate(x=df['x_scaled'], y=df['y_scaled'])")
                for x in range(3)
            ]
        )
        assert np.all(losses <= 0.1)

        # checking scaled input bounds
        scaled_input = tb.ref("input_bounds[0]")
        assert scaled_input[0] == pytest.approx(-1.73179, abs=0.01)
        assert scaled_input[1] == pytest.approx(1.73179, abs=0.01)

        # now let's compare our results against the possible solutions
        # of the original function - the first one is the global
        possible_solutions = [
            (-0.290839, -0.908622),
            (-1.447314, 1.279338),
            (0.871281, -0.178173),
            (2.000000, 3.455979),
        ]
        global_solution = possible_solutions[0]

        def matches_one_of(x, y, solutions, abs_tolerance=0.15):
            for s in solutions:
                if abs(x - s[0]) < abs_tolerance and abs(y - s[1]) < abs_tolerance:
                    return True
            # doesn't match
            print("*** not matching ***")
            print(x, y)
            print(solutions)
            return False

        assert matches_one_of(
            tb.ref("solution_1_reduced[0]"),
            tb.ref("solution_1_reduced[1]"),
            possible_solutions,
        )
        assert matches_one_of(
            tb.ref("solution_1_full[0]"),
            tb.ref("solution_1_full[1]"),
            possible_solutions,
        )
        assert matches_one_of(
            tb.ref("solution_2_comp[0]"),
            tb.ref("solution_2_comp[1]"),
            possible_solutions,
        )
        assert matches_one_of(
            tb.ref("solution_2_bigm[0]"),
            tb.ref("solution_2_bigm[1]"),
            [global_solution],
        )
        # Skipping partition tests since they are not working right now
        # assert matches_one_of(tb.ref("solution_2_partition[0]"),
        #                       tb.ref("solution_2_partition[1]"),
        #                       [global_solution])
        assert matches_one_of(
            tb.ref("solution_3_mixed[0]"),
            tb.ref("solution_3_mixed[1]"),
            possible_solutions,
        )


@pytest.mark.skipif(
    not onnx_available or not lightgbm_available,
    reason="onnx and lightgbm needed for this notebook",
)
def test_bo_with_trees():
    notebook_fname = "bo_with_trees.ipynb"
    book = open_book("", notebook_fname, execute=True)

    with book as tb:
        check_cell_execution(tb, notebook_fname, injections=0)

        # TODO: Add stronger test to verify correct output
        y_min = tb.ref("min(data['y'])")
        assert y_min < 10
