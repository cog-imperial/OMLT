import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook
import os 

# TODO: These will be replaced with stronger tests using testbook soon
def _test_run_notebook(folder, notebook_fname, n_cells):
    # change to notebook directory to allow testing
    cwd = os.getcwd()
    os.chdir(os.path.join(this_file_dir(), '..', '..', 'docs', 'notebooks', folder))
    with testbook(notebook_fname, execute=True) as tb:
        assert tb.code_cells_executed == n_cells
    os.chdir(cwd)

def test_autothermal_relu_notebook():
    _test_run_notebook('neuralnet', "auto-thermal-reformer-relu.ipynb", 13)

def test_autothermal_reformer():
    _test_run_notebook('neuralnet', "auto-thermal-reformer.ipynb", 13)

def test_build_network():
    _test_run_notebook('neuralnet', 'build_network.ipynb', 37)

def test_import_network():
    _test_run_notebook('neuralnet', 'import_network.ipynb', 16)

def test_mnist_example_convolutional():
    _test_run_notebook('neuralnet', 'mnist_example_convolutional.ipynb', 13)
    
def test_mnist_example_dense():
    _test_run_notebook('neuralnet', 'mnist_example_dense.ipynb', 13)

def test_neural_network_formulations():
    _test_run_notebook('neuralnet', 'neural_network_formulations.ipynb', 21)
    
