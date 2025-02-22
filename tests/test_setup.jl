using Pkg
Pkg.add(["Test", "CondaPkg", "PythonCall", "IJulia", "JuMP", "Ipopt", "HiGHS"])

using CondaPkg
CondaPkg.add("pyomo", version="==6.8.0")
CondaPkg.add_pip("omlt", version="@file:///home/runner/work/OMLT/OMLT/")
