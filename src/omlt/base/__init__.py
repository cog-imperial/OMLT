DEFAULT_MODELING_LANGUAGE = "pyomo"

from omlt.dependencies import julia_available

if julia_available:
    from omlt.base.julia import jl, jump

from omlt.base.var import OmltVar
from omlt.base.expression import OmltExpr

# from omlt.base.constraint import OmltConstraint
