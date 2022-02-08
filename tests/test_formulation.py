import pytest
from omlt.formulation import _setup_scaled_inputs_outputs
from omlt.block import OmltBlock
from omlt.scaling import OffsetScaling
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, value

def test_scaled_inputs_outputs():
    m = ConcreteModel()
    xoffset = {(0,i): float(i) for i in range(3)}
    xfactor = {(0,i): 0.5*(i+1) for i in range(3)}
    yoffset = {(1,i): -0.25*i for i in range(2)}
    yfactor = {(1,i): 0.125*(i+1) for i in range(2)}

    scaler = OffsetScaling(
        offset_inputs=xoffset,
        factor_inputs=xfactor,
        offset_outputs=yoffset,
        factor_outputs=yfactor
    )

    scaled_input_bounds = {(0,0): (0, 5), (0,1):(-2, 2), (0,2):(0, 1)}

    m.b1 = OmltBlock()
    m.b1._setup_inputs_outputs(
        input_indexes=[(0,0), (0,1), (0,2)],
        output_indexes=[(1,0), (1,1)]
    )
    _setup_scaled_inputs_outputs(m.b1, scaler=scaler, scaled_input_bounds=scaled_input_bounds)
    m.obj = Objective(expr=1)
    m.b1.inputs.fix(2)
    m.b1.outputs.fix(1)
    status = SolverFactory('ipopt').solve(m)

    assert value(m.b1.scaled_inputs[(0,0)]) == pytest.approx(4.0)
    assert value(m.b1.scaled_inputs[(0,1)]) == pytest.approx(1.0)
    assert value(m.b1.scaled_inputs[(0,2)]) == pytest.approx(0.0)
    assert value(m.b1.scaled_outputs[(1,0)]) == pytest.approx(8)
    assert value(m.b1.scaled_outputs[(1,1)]) == pytest.approx(5)

    assert m.b1.inputs[(0,0)].lb == pytest.approx(0.0)
    assert m.b1.inputs[(0,0)].ub == pytest.approx(2.5)
    assert m.b1.inputs[(0,1)].lb == pytest.approx(-1.0)
    assert m.b1.inputs[(0,1)].ub == pytest.approx(3.0)
    assert m.b1.inputs[(0,2)].lb == pytest.approx(2.0)
    assert m.b1.inputs[(0,2)].ub == pytest.approx(3.5)

    m = ConcreteModel()
    xoffset = {i: float(i) for i in range(3)}
    xfactor = {i: 0.5*(i+1) for i in range(3)}
    yoffset = {i: -0.25*i for i in range(2)}
    yfactor = {i: 0.125*(i+1) for i in range(2)}

    scaler = OffsetScaling(
        offset_inputs=xoffset,
        factor_inputs=xfactor,
        offset_outputs=yoffset,
        factor_outputs=yfactor
    )

    input_bounds = {0: (0, 5), 1:(-2, 2), 2:(0, 1)}

    m.b1 = OmltBlock()
    m.b1._setup_inputs_outputs(
        input_indexes=[0, 1, 2],
        output_indexes=[0, 1]
    )
    _setup_scaled_inputs_outputs(m.b1, scaler=scaler, scaled_input_bounds=input_bounds)
    m.obj = Objective(expr=1)
    m.b1.inputs.fix(2)
    m.b1.outputs.fix(1)
    status = SolverFactory('ipopt').solve(m)
    assert value(m.b1.scaled_inputs[0]) == pytest.approx(4.0)
    assert value(m.b1.scaled_inputs[1]) == pytest.approx(1.0)
    assert value(m.b1.scaled_inputs[2]) == pytest.approx(0.0)
    assert value(m.b1.scaled_outputs[0]) == pytest.approx(8)
    assert value(m.b1.scaled_outputs[1]) == pytest.approx(5)

    assert m.b1.inputs[0].lb == pytest.approx(0.0)
    assert m.b1.inputs[0].ub == pytest.approx(2.5)
    assert m.b1.inputs[1].lb == pytest.approx(-1.0)
    assert m.b1.inputs[1].ub == pytest.approx(3.0)
    assert m.b1.inputs[2].lb == pytest.approx(2.0)
    assert m.b1.inputs[2].ub == pytest.approx(3.5)
