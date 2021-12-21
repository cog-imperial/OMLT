import pytest
from omlt.formulation import _setup_scaled_inputs_outputs
from omlt.block import OmltBlock
from omlt.scaling import OffsetScaling
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory

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

    input_bounds = {(0,0): (0, 5), (0,1):(-2, 2), (0,2):(0, 1)}

    m.b1 = OmltBlock()
    m.b1._setup_inputs_outputs(
        input_indexes=[(0,0), (0,1), (0,2)],
        output_indexes=[(1,0), (1,1)]
    )
    _setup_scaled_inputs_outputs(m.b1, scaler=scaler, scaled_input_bounds=input_bounds)
    m.obj = Objective(expr=1)
    m.b1.inputs.fix(1)
    status = SolverFactory('ipopt').solve(m)

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
    m.b1.inputs.fix(1)
    status = SolverFactory('ipopt').solve(m)
    m.pprint()

    # TODO: complete this test

    # m.b2 = OmltBlock()
    # m.b2._setup_inputs_outputs(
    #     n_inputs=3, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout]
    # )
    # m.b2._setup_scaled_inputs_outputs(
    #     scaling_object=scaler, input_bounds=None, use_scaling_expressions=True
    # )
    # assert isinstance(
    #     m.b2.scaled_inputs_list[0], pyomo.core.expr.numeric_expr.DivisionExpression
    # )
    # assert isinstance(
    #     m.b2.scaled_outputs_list[0], pyomo.core.expr.numeric_expr.DivisionExpression
    # )

    # m.b3 = OmltBlock()
    # m.b3._setup_inputs_outputs(
    #     n_inputs=3, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout]
    # )
    # m.b3._setup_scaled_inputs_outputs(
    #     scaling_object=scaler, input_bounds=None, use_scaling_expressions=False
    # )
    # assert m.nvariables() == 10
    # assert m.nconstraints() == 5

    # m.b4 = OmltBlock()
    # m.b4._setup_inputs_outputs(
    #     n_inputs=3, input_vars=[m.cin], n_outputs=2, output_vars=[m.cout]
    # )
    # m.b4._setup_scaled_inputs_outputs(
    #     scaling_object=scaler, input_bounds=input_bounds, use_scaling_expressions=False
    # )
    # assert m.b4.scaled_inputs_list[0].lb == -2
    # assert m.b4.scaled_inputs_list[0].ub == 8
