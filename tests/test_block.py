import pytest
import pyomo.environ as pyo
from omlt import OmltBlock

class dummy_formulation(object):
    def __init__(self):
        self.input_indexes=['A', 'C', 'D']
        self.output_indexes=[(0,0), (0,1), (1,0), (1,1)] 

    def _set_block(self, blk):
        pass

    def _build_formulation(self):
        pass

def test_block():
    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    m.b._setup_inputs_outputs(input_indexes=['A', 'B', 'C'], output_indexes=[1,4])
    m.b2 = OmltBlock()
    m.b2._setup_inputs_outputs(input_indexes=[(1,3),(42,1975),(13,2)], output_indexes=[(0,0), (0,1), (1,0), (1,1)])

    assert [k for k in m.b.inputs] == ['A', 'B', 'C']
    assert [k for k in m.b.outputs] == [1,4]
    assert [k for k in m.b2.inputs] == [(1,3),(42,1975),(13,2)]
    assert [k for k in m.b2.outputs] == [(0,0), (0,1), (1,0), (1,1)]

    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    formulation = dummy_formulation()
    m.b.build_formulation(formulation)
    print(dir(m.b))
    assert m.b._OmltBlockData__formulation is formulation
    assert [k for k in m.b.inputs] == ['A', 'C', 'D']
    assert [k for k in m.b.outputs] == [(0,0), (0,1), (1,0), (1,1)]
    
def test_input_output_auto_creation():
    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    m.b._setup_inputs_outputs(
        input_indexes=range(3),
        output_indexes=range(2)
    )
    assert len(m.b.inputs) == 3
    assert len(m.b.outputs) == 2

    m.b2 = OmltBlock()
    m.b2._setup_inputs_outputs(
        input_indexes=[0],
        output_indexes=[0],
    )
    assert len(m.b2.inputs) == 1
    assert len(m.b2.outputs) == 1

    m.b3 = OmltBlock()
    with pytest.raises(ValueError):
        m.b3._setup_inputs_outputs(
            input_indexes=[],
            output_indexes=[],
        )
