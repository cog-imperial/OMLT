using Pkg
Pkg.add(name="CondaPkg", version="==0.2.24")
Pkg.add(["Test", "PythonCall", "JuMP", "Ipopt", "HiGHS"])

using CondaPkg
CondaPkg.add("pyomo", version="==6.8.0")
CondaPkg.add_pip("omlt", version="@file:///home/runner/work/OMLT/OMLT/")
