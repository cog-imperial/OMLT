using Pkg
Pkg.add(name="CondaPkg", version="0.2.23")
Pkg.add(name="PythonCall", version="0.9.23")
Pkg.add(name="IJulia", version="1.25.0")
Pkg.add(["Test", "JuMP", "Ipopt", "HiGHS"])

println("====================")
println(Pkg.status())
println("------")
using CondaPkg
CondaPkg.add("pyomo", version="6.8.0")
CondaPkg.add_pip("omlt", version="@file:///home/runner/work/OMLT/OMLT/")
println("====================")
println(CondaPkg.status())
println("------")

CondaPkg.withenv() do
    run(`pip list`)
  end
