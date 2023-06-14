import pytest
from pyomo.common.fileutils import this_file_dir
from testbook import testbook
import os

from omlt.dependencies import keras_available, onnx_available

#returns the executed notebook
def openBook(folder, notebook_fname):
    cwd = os.getcwd()
    os.chdir(os.path.join(this_file_dir(), '..', '..', 'docs', 'notebooks', folder))
    with testbook(notebook_fname, execute=True, timeout=300) as tb:
        os.chdir(cwd)
        return tb

def trainingAccuracy(trainingList):
    final_output = trainingList[-1].split(",")
    final_output = [x.split(":") for x in final_output]
    avg_loss = float(final_output[0][-1])
    final_accuracy = int(final_output[1][-1][:5])/10000
    return (avg_loss, final_accuracy)

#asserts that every cell was succesfully executed
def _test_run_notebook(tb, n_cells):
    assert tb.code_cells_executed == n_cells

@pytest.mark.skipif(not keras_available, reason='keras needed for this notebook')
def test_autothermal_relu_notebook():
    tb = openBook('neuralnet', 'auto-thermal-reformer-relu.ipynb')
    _test_run_notebook(tb, 13)

    #grab the output and format
    output = (tb.cell_output_text(16)).splitlines()

    #split each element by colon, then keep only numbers
    output = [x.split(": ") for x in output]
    output = [float(x[1]) for x in output]

    #testing the output (with some tolerance)
    assert output[0] == pytest.approx(0.1)
    assert output[1] == pytest.approx(1.1250305, abs=3e-2)
    assert output[2] == pytest.approx(0.33191865, abs=1.8e-2)
    assert output[3] == pytest.approx(0.34, abs=1.8e-2)

@pytest.mark.skipif(not keras_available, reason='keras needed for this notebook')
def test_autothermal_reformer():
    tb = openBook('neuralnet', "auto-thermal-reformer.ipynb")
    _test_run_notebook(tb, 13)

    #grab the output and format
    output = (tb.cell_output_text(16)).splitlines()

    #split each element by colon, then keep only numbers
    output = [x.split(": ") for x in output]
    output = [float(x[1]) for x in output]

    #testing the output (with some tolerance)
    assert output[0] == pytest.approx(0.1, abs=1e-4)
    assert output[1] == pytest.approx(1.1250305, abs=1.8e-2)
    assert output[2] == pytest.approx(0.33191865, abs=1.8e-2)
    assert output[3] == pytest.approx(0.34, abs=1.e-2)

def test_build_network():
    tb = openBook('neuralnet', 'build_network.ipynb')
    _test_run_notebook(tb, 37)

    #checking the last outputs of the notebook
    outputs = []
    for cell in [71, 73, 75, 77]:
        cellOutput = tb.cell_output_text(cell)[5:].replace(" ", "")
        outputs.append(cellOutput)
    assert outputs[0] == '([[1.3,0.3],\n[0.3,0.4]])'
    assert outputs[1] == '([[1.3,0.3],\n[0.3,0.4]])'
    assert outputs[2] == '([[1.25],\n[0.35]])'
    assert outputs[3] == '([2.15])'

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_import_network():
    tb = openBook('neuralnet', 'import_network.ipynb',)
    _test_run_notebook(tb, 16)

    #checking that the final layers are correct
    layer_descr = tb.cell_output_text(37).splitlines()
    assert layer_descr[0] == "0\tInputLayer(input_size=[8], output_size=[8])\tlinear"
    assert layer_descr[1] == "1\tDenseLayer(input_size=[8], output_size=[12])\trelu"
    assert layer_descr[2] == "2\tDenseLayer(input_size=[12], output_size=[1])\trelu"
    assert layer_descr[3] == "3\tDenseLayer(input_size=[1], output_size=[1])\tlinear"

    #checking that the input bounds are correct
    correct_bounds = []
    #('{0: (0.0, 17.0),\n 1: (0.0, 199.0),\n 2: (0.0, 122.0),\n 3: (0.0, 99.0),\n 4: (0.0, 846.0),\n 5: (0.0, 67.1),\n 6: (0.078, 2.42),\n 7: (21.0, 81.0)}'
    scaled_inp_bounds = tb.cell_output_text(35).replace(" ", "")
    scaled_inp_bounds = scaled_inp_bounds.splitlines()
    assert scaled_inp_bounds[0] == '{0:(0.0,17.0),'
    assert scaled_inp_bounds[1] == '1:(0.0,199.0),'
    assert scaled_inp_bounds[2] == '2:(0.0,122.0),'
    assert scaled_inp_bounds[3] == '3:(0.0,99.0),'
    assert scaled_inp_bounds[4] == '4:(0.0,846.0),'
    assert scaled_inp_bounds[5] == '5:(0.0,67.1),'
    assert scaled_inp_bounds[6] == "6:(0.078,2.42),"
    assert scaled_inp_bounds[7] == '7:(21.0,81.0)}'

    #checking that the final loss is around 0.25
    epoch_loss = tb.cell_output_text(28).splitlines()
    assert (float(epoch_loss[-1][25:]) == pytest.approx(0.251, abs=1e-2))

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_mnist_example_convolutional():
    tb = openBook('neuralnet', 'mnist_example_convolutional.ipynb')
    _test_run_notebook(tb, 13)

    #check the model
    layers = tb.cell_output_text(19).splitlines()
    assert layers[0] == '0\tInputLayer(input_size=[1, 28, 28], output_size=[1, 28, 28])\tlinear'
    assert layers[1] == '1\tConvLayer(input_size=[1, 28, 28], output_size=[2, 13, 13], strides=[2, 2], kernel_shape=(4, 4))\trelu'
    assert layers[2] == '2\tConvLayer(input_size=[2, 13, 13], output_size=[2, 5, 5], strides=[2, 2], kernel_shape=(4, 4))\trelu'
    assert layers[3] == '3\tDenseLayer(input_size=[1, 50], output_size=[1, 10])\trelu'
    assert layers[4] == '4\tDenseLayer(input_size=[1, 10], output_size=[1, 10])\tlinear'

    #check accuracy of the training
    training_output = tb.cell_output_text(10).splitlines()
    results = trainingAccuracy(training_output)
    avg_loss, final_accuracy = results[0], results[1]
    assert avg_loss == pytest.approx(0.25, abs=0.05)
    assert final_accuracy == pytest.approx(0.92, abs = 4e-2)

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_mnist_example_dense():
    tb = openBook('neuralnet', 'mnist_example_dense.ipynb')
    _test_run_notebook(tb, 13)

    #check accuracy of the training
    training_output = tb.cell_output_text(10).splitlines()
    results = trainingAccuracy(training_output)
    avg_loss, final_accuracy = results[0], results[1]
    assert avg_loss == pytest.approx(0.0879, abs=0.01)
    assert final_accuracy == pytest.approx(0.95, abs=0.05)

@pytest.mark.skipif(not keras_available, reason='keras needed for this notebook')
def test_neural_network_formulations():
    tb = openBook('neuralnet', 'neural_network_formulations.ipynb',)
    _test_run_notebook(tb, 21)

@pytest.mark.skipif(not onnx_available, reason='onnx needed for this notebook')
def test_bo_with_trees():
    tb = openBook('', 'bo_with_trees.ipynb',)
    _test_run_notebook(tb, 10)

