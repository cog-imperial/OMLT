import pyomo.environ as pyo
import pytest
from omlt import OmltBlock

INPUTS_LENGTH = 3
OUTPUTS_LENGTH = 2


class DummyFormulation:
    def __init__(self):
        self.input_indexes = ["A", "C", "D"]
        self.output_indexes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def _set_block(self, blk):
        pass

    def _build_formulation(self):
        pass


def test_block():
    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    m.b._setup_inputs_outputs(input_indexes=["A", "B", "C"], output_indexes=[1, 4])
    m.b2 = OmltBlock()
    m.b2._setup_inputs_outputs(
        input_indexes=[(1, 3), (42, 1975), (13, 2)],
        output_indexes=[(0, 0), (0, 1), (1, 0), (1, 1)],
    )

    assert list(m.b.inputs) == ["A", "B", "C"]
    assert list(m.b.outputs) == [1, 4]
    assert list(m.b2.inputs) == [(1, 3), (42, 1975), (13, 2)]
    assert list(m.b2.outputs) == [(0, 0), (0, 1), (1, 0), (1, 1)]

    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    formulation = DummyFormulation()
    m.b.build_formulation(formulation)

    assert m.b._OmltBlockData__formulation is formulation
    assert list(m.b.inputs) == ["A", "C", "D"]
    assert list(m.b.outputs) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_input_output_auto_creation():
    m = pyo.ConcreteModel()
    m.b = OmltBlock()
    m.b._setup_inputs_outputs(
        input_indexes=range(INPUTS_LENGTH), output_indexes=range(OUTPUTS_LENGTH)
    )
    assert len(m.b.inputs) == INPUTS_LENGTH
    assert len(m.b.outputs) == OUTPUTS_LENGTH

    m.b2 = OmltBlock()
    m.b2._setup_inputs_outputs(
        input_indexes=[0],
        output_indexes=[0],
    )
    assert len(m.b2.inputs) == 1
    assert len(m.b2.outputs) == 1

    m.b3 = OmltBlock()
    expected_msg = "OmltBlock must have at least one input and at least one output."
    with pytest.raises(ValueError, match=expected_msg):
        m.b3._setup_inputs_outputs(
            input_indexes=[],
            output_indexes=[],
        )
