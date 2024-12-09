import os
from pathlib import Path

import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook

from omlt.dependencies import (
    keras_available,
    onnx_available,
    torch_available,
    torch_geometric_available,
)


def _test_run_notebook(folder, notebook_fname, n_cells):
    # Change to notebook directory to allow for testing
    cwd = Path.cwd()
    os.chdir(Path(this_file_dir()) / ".." / ".." / "docs" / "notebooks" / folder)
    with testbook(notebook_fname, timeout=500, execute=True) as tb:
        assert tb.code_cells_executed == n_cells
    os.chdir(cwd)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_relu_notebook():
    _test_run_notebook("neuralnet", "auto-thermal-reformer-relu.ipynb", 13)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_autothermal_reformer():
    _test_run_notebook("neuralnet", "auto-thermal-reformer.ipynb", 13)


def test_build_network():
    _test_run_notebook("neuralnet", "build_network.ipynb", 37)


@pytest.mark.skipif(
    (not onnx_available) or (not keras_available),
    reason="onnx and keras needed for this notebook",
)
def test_import_network():
    _test_run_notebook("neuralnet", "import_network.ipynb", 16)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_convolutional():
    _test_run_notebook("neuralnet", "mnist_example_convolutional.ipynb", 13)


@pytest.mark.skipif(not onnx_available, reason="onnx needed for this notebook")
def test_mnist_example_dense():
    _test_run_notebook("neuralnet", "mnist_example_dense.ipynb", 13)


@pytest.mark.skipif(not keras_available, reason="keras needed for this notebook")
def test_neural_network_formulations():
    _test_run_notebook("neuralnet", "neural_network_formulations.ipynb", 21)


@pytest.mark.skipif(
    not (torch_available and torch_geometric_available),
    reason="torch and torch_geometric needed for this notebook",
)
def test_graph_neural_network_formulation():
    _test_run_notebook("neuralnet", "graph_neural_network_formulation.ipynb", 8)


def test_index_handling():
    _test_run_notebook("neuralnet", "index_handling.ipynb", 6)
