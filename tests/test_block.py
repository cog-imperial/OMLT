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

    def _clear_inputs(self):
        self.input_indexes = []

    def _clear_outputs(self):
        self.output_indexes = []


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
    formulation1 = DummyFormulation()
    formulation1._clear_inputs()
    expected_msg = (
        "OmltBlock must have at least one input to build a formulation. "
        f"{formulation1} has no inputs."
    )
    with pytest.raises(ValueError, match=expected_msg):
        m.b3.build_formulation(formulation1)

    formulation2 = DummyFormulation()
    formulation2._clear_outputs()
    expected_msg = (
        "OmltBlock must have at least one output to build a formulation. "
        f"{formulation2} has no outputs."
    )
    with pytest.raises(ValueError, match=expected_msg):
        m.b3.build_formulation(formulation2)
