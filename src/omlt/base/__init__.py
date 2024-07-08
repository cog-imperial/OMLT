DEFAULT_MODELING_LANGUAGE = "pyomo"

from omlt.dependencies import julia_available

if julia_available:
    from omlt.base.julia import jl, jump

from omlt.base.constraint import OmltConstraint
from omlt.base.expression import OmltExpr
from omlt.base.pyomo import *
from omlt.base.var import OmltVar

__all__ = [
    "julia_available",
    "jl",
    "jump",
    "OmltExpr",
    "OmltVar",
    "OmltConstraint",
]
