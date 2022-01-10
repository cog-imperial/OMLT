from pathlib import Path

import onnx
import pyomo.environ as pe
import pytest

from omlt import OmltBlock
from omlt.gbt.gbt_formulation import GBTBigMFormulation, add_formulation_to_block
from omlt.gbt.model import GradientBoostedTreeModel


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

    assert len(list(m.gbt.component_data_objects(pe.Var))) == 202+10 # our auto-created variables
    assert len(list(m.gbt.component_data_objects(pe.Constraint))) == 423 # TODO: fix?

    assert len(m.gbt.z_l) == 160
    assert len(m.gbt.y) == 42

    assert len(m.gbt.single_leaf) == 20
    assert len(m.gbt.left_split) == 140
    assert len(m.gbt.right_split) == 140
    assert len(m.gbt.categorical) == 0
    assert len(m.gbt.var_lower) == 42
    assert len(m.gbt.var_upper) == 42


# TODO: did we remove categorical variables intentionally?
# def test_formulation_with_categorical_variables():
#     model = onnx.load(Path(__file__).parent / "categorical_model.onnx")

#     m = pe.ConcreteModel()

#     m.x = pe.Var(range(3), bounds=(-2.0, 2.0))
#     # categorical variable
#     m.y = pe.Var(bounds=(0, 1), domain=pe.Integers)

#     m.z = pe.Var()

#     m.gbt = pe.Block()
#     add_formulation_to_block(
#         m.gbt, model, input_vars=[m.x[0], m.x[1], m.x[2], m.y], output_vars=[m.z]
#     )

#     assert len(list(m.gbt.component_data_objects(pe.Var))) == 193
#     # there are 28 * 2 constraints missing
#     # related to categorical variables
#     assert len(list(m.gbt.component_data_objects(pe.Constraint))) == 391

#     assert len(m.gbt.z_l) == 160
#     assert len(m.gbt.y) == 31

#     assert len(m.gbt.single_leaf) == 20
#     assert len(m.gbt.left_split) == 140
#     assert len(m.gbt.right_split) == 140
#     assert len(m.gbt.categorical) == 1
#     assert len(m.gbt.var_lower) == 31
#     assert len(m.gbt.var_upper) == 31


def test_big_m_formulation_block():
    onnx_model = onnx.load(Path(__file__).parent / "continuous_model.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = GBTBigMFormulation(model)
    m.mod.build_formulation(formulation)

    m.obj = pe.Objective(expr=0)


def test_big_m_formulation_block_with_dimension_subset():
    onnx_model = onnx.load(Path(__file__).parent / "dimension_subset.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = GBTBigMFormulation(model)
    # if it can build the formulation it means it is handling the lack
    # of all dimension
    m.mod.build_formulation(formulation)
