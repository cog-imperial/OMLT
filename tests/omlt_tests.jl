using PythonCall
using JuMP
using Ipopt
using HiGHS
using Test

omlt = pyimport("omlt")
omlt_io = pyimport("omlt.io")
omlt_nn = pyimport("omlt.neuralnet")
omlt_julia = pyimport("omlt.base.julia")

onnx_py = pyimport("onnx")

@testset "OMLT variable" begin

    v = omlt_julia.OmltScalarJuMP(bounds=(0,5), initialize=(3,))

    @test pyconvert(Nothing, v.name) == nothing
    @test pyconvert(Bool, pyeq(v.lb, 0))
    @test pyconvert(Bool, pyeq(v.ub, 5))

    v.value = 4
    @test pyconvert(Bool, pyeq(v.value, 4))
end

@testset "Linear expressions" begin

    jump_block = omlt_julia.OmltBlockJuMP()
    jump_block.v1 = omlt_julia.OmltScalarJuMP(initialize=2)
    jump_block.v2 = omlt_julia.OmltScalarJuMP(initialize=3)
    element2 = jump_block.v2._var

    var_plus_three = pyadd(jump_block.v1, 3)
    @test pyisinstance(var_plus_three, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, var_plus_three() == 5)

    three_minus_var = 3 - jump_block.v1
    @test pyisinstance(three_minus_var, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, pyeq(three_minus_var(), 1))

    three_times_var = 3 * jump_block.v1
    @test pyisinstance(three_times_var, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, three_times_var() == 6)

    var_times_three = jump_block.v1 * 3
    @test pyisinstance(var_times_three, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, var_times_three() == 6)

    expr_sum = var_plus_three + var_times_three
    @test pyisinstance(expr_sum, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, expr_sum() == 11)

    var_minus_expr = element2 - three_minus_var
    @test pyisinstance(var_minus_expr, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, var_minus_expr() == 2)

    expr_minus_expr = var_plus_three - three_minus_var
    @test pyisinstance(expr_minus_expr, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, expr_minus_expr() == 4)

    expr_div_int = var_plus_three / 5
    @test pyisinstance(expr_div_int, omlt_julia.OmltExprJuMP)
    @test pyconvert(Bool, expr_div_int() == 1)

    constraint = var_minus_expr == expr_div_int
    @test pyisinstance(constraint, omlt_julia.OmltConstraintScalarJuMP)
end

@testset "Nonlinear expression" begin
    jump_block = omlt_julia.OmltBlockJuMP()
    jump_block.v1 = omlt_julia.OmltScalarJuMP(initialize=2)
    element1 = jump_block.v1._var

    expe = element1.exp()
    loge = element1.log()
    hypt = element1.tanh()

    @test pyisinstance(expe, omlt_julia.OmltExprJuMP)
    @test pyisinstance(loge, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt + 3, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt - 3, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt - element1, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt * 3, omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt.log(), omlt_julia.OmltExprJuMP)
    @test pyisinstance(hypt.tanh(), omlt_julia.OmltExprJuMP)
end

@testset "Bad expression definition messages" begin
    expected_msg1 = """
    Python: ValueError: \
        ('Tried to create an OmltExprJuMP with an invalid expression. Expressions \
        must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where \
        d is exp, log, or tanh. %s was provided', 'invalid')"""
    @test_throws expected_msg1 omlt_julia.OmltExprJuMP("invalid")

    expected_msg2 = """
    Python: ValueError: \
        ('Tried to create an OmltExprJuMP with an invalid expression. Expressions \
        must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where \
        d is exp, log, or tanh. %s was provided', ('invalid', 'pair'))"""
    @test_throws expected_msg2 omlt_julia.OmltExprJuMP(("invalid","pair"))

    expected_msg3 = """
    Python: ValueError: \
        ('Tried to create an OmltExprJuMP with an invalid expression. Expressions \
        must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where \
        d is exp, log, or tanh. %s was provided', ('invalid', 'triple', 'expression'))"""
    
    @test_throws expected_msg3 omlt_julia.OmltExprJuMP(("invalid","triple","expression"))
end

@testset "Bad expression arithmetic messages" begin
    v = omlt_julia.OmltScalarJuMP()
    expected_msg1 = """
    ('Unrecognized types for addition, %s, %s', \
        <class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"""
    @test_throws expected_msg1 pyadd(v, "invalid")

    expected_msg2 = """
    ('Unrecognized types for subtraction, %s, %s', \
    <class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"""
    @test_throws expected_msg2 pysub(v, "invalid")

    expected_msg3 = """
    ('Unrecognized types for multiplication, %s, %s', \
        <class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"""
    @test_throws expected_msg3 pymul(v, "invalid")

    expected_msg4 = """
    ('Unrecognized types for division, %s, %s', \
        <class 'omlt.base.julia.OmltScalarJuMP'>, <class 'str'>)"""
    @test_throws expected_msg4 pytruediv(v, "invalid")
end

@testset "Full-space with sigmoid activation" begin
    fs_model = omlt_julia.OmltBlockJuMP()
    fs_model.set_optimizer(Ipopt.Optimizer)

    jump_model = pyconvert(Model, fs_model.get_model())

    @variable(jump_model, x)
    @variable(jump_model, y)
    @objective(jump_model, Min, y)
    scale_x = (1, 0.5)
    scale_y = (-0.25, 0.125)
    
    scaler = omlt.OffsetScaling(
        offset_inputs=[scale_x[1]],
        factor_inputs=[scale_x[2]],
        offset_outputs=[scale_y[1]],
        factor_outputs=[scale_y[2]]
    )
    scaled_input_bounds = Dict(0 => (0,5))
    
    path = "/workspaces/OMLT/tests/models/keras_linear_131_sigmoid.onnx"
    
    py_model = onnx_py.load(path)
    net = omlt_io.load_onnx_neural_network(py_model, scaler, scaled_input_bounds)
    formulation = omlt_nn.FullSpaceSmoothNNFormulation(net)

    fs_model.build_formulation(formulation)

    @constraint(jump_model, x == pyconvert(VariableRef, fs_model._varrefs["inputs_0"]))
    @constraint(jump_model, y == pyconvert(VariableRef, fs_model._varrefs["outputs_0"]))

    optimize!(jump_model)

    @test isapprox(value(x), 1)
    @test isapprox(value(y), -0.253, atol=0.001)
end

@testset "Reduced space with linear activation" begin
    rs_model = omlt_julia.OmltBlockJuMP()
    rs_model.set_optimizer(HiGHS.Optimizer)

    jump_model = pyconvert(Model, rs_model.get_model())

    @variable(jump_model, x)
    @variable(jump_model, y)
    @objective(jump_model, Min, y)
    scale_x = (1, 0.5)
    scale_y = (-0.25, 0.125)
    
    scaler = omlt.OffsetScaling(
        offset_inputs=[scale_x[1]],
        factor_inputs=[scale_x[2]],
        offset_outputs=[scale_y[1]],
        factor_outputs=[scale_y[2]]
    )
    scaled_input_bounds = Dict(0 => (0,5))
    
    path = "/workspaces/OMLT/tests/models/keras_linear_131.onnx"
    
    py_model = onnx_py.load(path)
    net = omlt_io.load_onnx_neural_network(py_model, scaler, scaled_input_bounds)
    formulation = omlt_nn.FullSpaceSmoothNNFormulation(net)

    rs_model.build_formulation(formulation)

    @constraint(jump_model, x == pyconvert(VariableRef, rs_model._varrefs["inputs_0"]))
    @constraint(jump_model, y == pyconvert(VariableRef, rs_model._varrefs["outputs_0"]))

    optimize!(jump_model)

    @test isapprox(value(x), 1)
    @test isapprox(value(y), -0.250, atol=0.001)
end