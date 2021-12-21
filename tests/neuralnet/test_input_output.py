import pyomo
import pyomo.environ as pyo
import pytest

from omlt.block import OmltBlock
from omlt.scaling import OffsetScaling


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
