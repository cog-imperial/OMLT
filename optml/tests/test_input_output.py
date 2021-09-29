import pytest
import pyomo.environ as pyo
from optml.block import _BaseInputOutputBlock

def test_input_output_auto_creation():
    m  = pyo.ConcreteModel()
    m.b = _BaseInputOutputBlock()
    m.b._setup_inputs_outputs(n_inputs=3, n_outputs=2)
    assert len(m.b.inputs) == 3
    assert len(m.b.outputs) == 2
    assert len(m.b.inputs_list) == 3
    assert len(m.b.outputs_list) == 2
    assert m.b.inputs_list[0] is m.b.inputs[0]
    assert m.b.inputs_list[1] is m.b.inputs[1]
    assert m.b.inputs_list[2] is m.b.inputs[2]
    assert m.b.outputs_list[0] is m.b.outputs[0]
    assert m.b.outputs_list[1] is m.b.outputs[1]

    m.b2 = _BaseInputOutputBlock()
    m.b2._setup_inputs_outputs(n_inputs=1, n_outputs=1)
    assert len(m.b2.inputs) == 1
    assert len(m.b2.outputs) == 1
    assert len(m.b2.inputs_list) == 1
    assert len(m.b2.outputs_list) == 1
    assert m.b2.inputs_list[0] is m.b2.inputs[0]
    assert m.b2.outputs_list[0] is m.b2.outputs[0]
    
    m.b3 = _BaseInputOutputBlock()
    with pytest.raises(ValueError):
        m.b3._setup_inputs_outputs(n_inputs=0, n_outputs=0)

def test_provided_inputs_outputs():
    m = pyo.ConcreteModel()
    m.cin = pyo.Var(['A', 'B', 'C'])
    m.cout = pyo.Var([1,2])
    

    # test inputs provided, outputs auto
    m.b = _BaseInputOutputBlock()
    m.b._setup_inputs_outputs(n_inputs=3, input_vars=[m.cin], n_outputs=2)
    assert hasattr(m.b, 'inputs') == False
    assert len(m.b.outputs) == 2
    assert len(m.b.inputs_list) == 3
    assert len(m.b.outputs_list) == 2
    assert m.b.inputs_list[0] is m.cin['A']
    assert m.b.inputs_list[1] is m.cin['B']
    assert m.b.inputs_list[2] is m.cin['C']
    assert m.b.outputs_list[0] is m.b.outputs[0]
    assert m.b.outputs_list[1] is m.b.outputs[1]

    # test outputs provided, inputs auto
    m.b2 = _BaseInputOutputBlock()
    m.b2._setup_inputs_outputs(n_inputs=2, n_outputs=2, output_vars=[m.cout])
    assert len(m.b2.inputs) == 2
    assert hasattr(m.b2, 'outputs') == False
    assert len(m.b2.inputs_list) == 2
    assert len(m.b2.outputs_list) == 2
    assert m.b2.inputs_list[0] is m.b2.inputs[0]
    assert m.b2.inputs_list[1] is m.b2.inputs[1]
    assert m.b2.outputs_list[0] is m.cout[1]
    assert m.b2.outputs_list[1] is m.cout[2]

    # test both provided
    m.b3 = _BaseInputOutputBlock()
    m.b3._setup_inputs_outputs(n_inputs=3, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout])
    assert hasattr(m.b3, 'inputs') == False
    assert hasattr(m.b3, 'outputs') == False
    assert len(m.b3.inputs_list) == 3
    assert len(m.b3.outputs_list) == 2
    assert m.b3.inputs_list[0] is m.cin['A']
    assert m.b3.inputs_list[1] is m.cin['B']
    assert m.b3.inputs_list[2] is m.cin['C']
    assert m.b3.outputs_list[0] is m.cout[1]
    assert m.b3.outputs_list[1] is m.cout[2]

    # test both provided with data objects reordered
    m.b4 = _BaseInputOutputBlock()
    m.b4._setup_inputs_outputs(n_inputs=2, input_vars=[m.cin['C'], m.cin['A']], n_outputs=2, output_vars=[m.cout[1], m.cout[2]])
    assert hasattr(m.b4, 'inputs') == False
    assert hasattr(m.b4, 'outputs') == False
    assert len(m.b4.inputs_list) == 2
    assert len(m.b4.outputs_list) == 2
    assert m.b4.inputs_list[0] is m.cin['C']
    assert m.b4.inputs_list[1] is m.cin['A']
    assert m.b4.outputs_list[0] is m.cout[1]
    assert m.b4.outputs_list[1] is m.cout[2]

    # test errors with mismatched lengths
    m.b5 = _BaseInputOutputBlock()
    with pytest.raises(ValueError):
        m.b5._setup_inputs_outputs(n_inputs=2, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout])

    m.b6 = _BaseInputOutputBlock()
    with pytest.raises(ValueError):
        m.b6._setup_inputs_outputs(n_inputs=3, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout[1]])
