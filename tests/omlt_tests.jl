using Pkg
Pkg.add(name="CondaPkg", version="0.2.23")
Pkg.add(name="PythonCall", version="0.9.23")
Pkg.add(name="IJulia", version="1.25.0")
Pkg.add(["Test", "JuMP", "Ipopt", "HiGHS"])

using CondaPkg
CondaPkg.add_pip("omlt", version="@file:///home/runner/work/OMLT/OMLT/")

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

