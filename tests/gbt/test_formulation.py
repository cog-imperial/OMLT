from pathlib import Path

import onnx
import pyomo.environ as pe
import pytest

from omlt import OmltBlock
from omlt.gbt.formulation import BigMFormulation, add_formulation_to_block
from omlt.gbt.model import GradientBoostedTreeModel


def test_formulation_with_continuous_variables():
    model = onnx.load(Path(__file__).parent / "continuous_model.onnx")

    m = pe.ConcreteModel()

    m.x = pe.Var(range(4), bounds=(-2.0, 2.0))
    m.x[3].setlb(0.0)
    m.x[3].setub(1.0)

    m.z = pe.Var()

    m.gbt = pe.Block()
    add_formulation_to_block(
        m.gbt, model, input_vars=[m.x[i] for i in range(4)], output_vars=[m.z]
    )

    assert len(list(m.gbt.component_data_objects(pe.Var))) == 202
    assert len(list(m.gbt.component_data_objects(pe.Constraint))) == 423

    assert len(m.gbt.z_l) == 160
    assert len(m.gbt.y) == 42

    assert len(m.gbt.single_leaf) == 20
    assert len(m.gbt.left_split) == 140
    assert len(m.gbt.right_split) == 140
    assert len(m.gbt.categorical) == 0
    assert len(m.gbt.var_lower) == 42
    assert len(m.gbt.var_upper) == 42


def test_big_m_formulation_block():
    onnx_model = onnx.load(Path(__file__).parent / "continuous_model.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = BigMFormulation(model)
    m.mod.build_formulation(formulation)

    m.obj = pe.Objective(expr=0)


def test_big_m_formulation_block_with_dimension_subset():
    onnx_model = onnx.load(Path(__file__).parent / "dimension_subset.onnx")
    model = GradientBoostedTreeModel(onnx_model)

    m = pe.ConcreteModel()
    m.mod = OmltBlock()
    formulation = BigMFormulation(model)
    # if it can build the formulation it means it is handling the lack
    # of all dimension
    m.mod.build_formulation(formulation)
