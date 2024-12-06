from omlt.base.language import DEFAULT_MODELING_LANGUAGE
# from omlt.dependencies import julia_available

from omlt.base.constraint import (
    OmltConstraint,
    OmltConstraintFactory,
    OmltConstraintIndexed,
    OmltConstraintScalar,
)
from omlt.base.expression import OmltExpr, OmltExprFactory
from omlt.base.pyomo import (
    OmltConstraintIndexedPyomo,
    OmltConstraintScalarPyomo,
    OmltExprScalarPyomo,
    OmltIndexedPyomo,
    OmltScalarPyomo,
)
from omlt.base.var import OmltIndexed, OmltScalar, OmltVar, OmltVarFactory

# if julia_available:
#     from omlt.base.julia import (
#         OmltConstraintIndexedJuMP,
#         OmltConstraintScalarJuMP,
#         OmltExprJuMP,
#         OmltIndexedJuMP,
#         OmltScalarJuMP,
#         Jl,
#         jump,
#     )

__all__ = [
    "DEFAULT_MODELING_LANGUAGE",
    "julia_available",
    "Jl",
    "jump",
    "OmltExpr",
    "OmltExprFactory",
    "OmltScalar",
    "OmltIndexed",
    "OmltVar",
    "OmltVarFactory",
    "OmltConstraintIndexed",
    "OmltConstraintScalar",
    "OmltConstraint",
    "OmltConstraintFactory",
    "OmltConstraintIndexedPyomo",
    "OmltConstraintScalarPyomo",
    "OmltExprScalarPyomo",
    "OmltIndexedPyomo",
    "OmltScalarPyomo",
    "OmltConstraintIndexedJuMP",
    "OmltConstraintScalarJuMP",
    "OmltExprJuMP",
    "OmltIndexedJuMP",
    "OmltScalarJuMP",
]
