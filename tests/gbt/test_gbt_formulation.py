from pathlib import Path

import pyomo.environ as pe
import pytest

from omlt import OmltBlock
from omlt.dependencies import onnx, onnx_available
from omlt.gbt.gbt_formulation import GBTBigMFormulation
from omlt.gbt.model import GradientBoostedTreeModel

TOTAL_CONSTRAINTS = 423
Y_VARS = 42
Z_L_VARS = 160
SINGLE_LEAVES = 20
SPLITS = 140


@pytest.mark.skip("Francesco and Alex need to check this test")
def test_formulation_with_continuous_variables():
    model = onnx.load(Path(__file__).parent / "continuous_model.onnx")

    m = pe.ConcreteModel()

    m.x = pe.Var(range(4), bounds=(-2.0, 2.0))
    m.x[3].setlb(0.0)
    m.x[3].setub(1.0)

    m.z = pe.Var()

    m.gbt = OmltBlock()
    m.gbt.build_formulation(GBTBigMFormulation(GradientBoostedTreeModel(model)))

    assert (
        len(list(m.gbt.component_data_objects(pe.Var))) == 202 + 10
    )  # our auto-created variables

    assert len(list(m.gbt.component_data_objects(pe.Constraint))) == TOTAL_CONSTRAINTS

    assert len(m.gbt.z_l) == Z_L_VARS
    assert len(m.gbt.y) == Y_VARS

    assert len(m.gbt.single_leaf) == SINGLE_LEAVES
    assert len(m.gbt.left_split) == SPLITS
    assert len(m.gbt.right_split) == SPLITS
    assert len(m.gbt.categorical) == 0
    assert len(m.gbt.var_lower) == Y_VARS
    assert len(m.gbt.var_upper) == Y_VARS


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_big_m_formulation_block():
    onnx_model = onnx.load(Path(__file__).parent / "continuous_model.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = GBTBigMFormulation(model)
    m.mod.build_formulation(formulation)

    m.obj = pe.Objective(expr=0)


@pytest.mark.skipif(not onnx_available, reason="Need ONNX for this test")
def test_big_m_formulation_block_with_dimension_subset():
    onnx_model = onnx.load(Path(__file__).parent / "dimension_subset.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = GBTBigMFormulation(model)
    # if it can build the formulation it means it is handling the lack
    # of all dimension
    m.mod.build_formulation(formulation)
