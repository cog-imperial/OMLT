from omlt.dependencies import julia_available, moi_available
from omlt.base.expression import OmltExpression

if julia_available and moi_available:
    from juliacall import Main as jl
    from juliacall import Base

    jl_err = Base.error
    jl.seval("import MathOptInterface")
    moi = jl.MathOptInterface
    jl.seval("import JuMP")
    jump = jl.JuMP


class JuMPVarInfo:
    def __init__(
        self,
        lower_bound=None,
        upper_bound=None,
        fixed_value=None,
        start_value=None,
        binary=False,
        integer=False,
    ):
        self.has_lb = lower_bound is not None
        self.lb = lower_bound
        self.has_ub = upper_bound is not None
        self.ub = upper_bound
        self.has_fix = fixed_value is not None
        self.fixed_value = fixed_value
        self.has_start = start_value is not None
        self.start_value = start_value
        self.binary = binary
        self.integer = integer

    @property
    def lower_bound(self):
        return self.lb

    @lower_bound.setter
    def lower_bound(self, value=None):
        self.lb = value
        self.has_lb = value is not None

    def setlb(self, value):
        self.lower_bound = value

    @property
    def upper_bound(self):
        return self.ub

    @upper_bound.setter
    def upper_bound(self, value=None):
        self.ub = value
        self.has_ub = value is not None

    def setub(self, value):
        self.upper_bound = value

    def to_jump(self):
        return jump.VariableInfo(
            self.has_lb,
            self.lower_bound,
            self.has_ub,
            self.upper_bound,
            self.has_fix,
            self.fixed_value,
            self.has_start,
            self.start_value,
            self.binary,
            self.integer,
        )


class JumpVar:
    def __init__(self, varinfo: JuMPVarInfo, name):
        self.info = varinfo
        self.name = name
        self.construct()

    def __str__(self):
        return self.name

    def setlb(self, value):
        self.info.setlb(value)
        self.construct()

    def setub(self, value):
        self.info.setlb(value)
        self.construct()

    def construct(self):
        self.var = jump.build_variable(Base.error, self.info.to_jump())

    @property
    def value(self):
        return self.var.info.start

    def add_to_model(self, model, name=None):
        if name is None:
            name = self._name
        jump.add_variable(model, self.var, name)

    def to_jump(self):
        return self.var

    def __sub__(self, other):
        return OmltExpression(expr=(self, "-", other), format="jump")

    def __mul__(self, other):
        return OmltExpression(expr=(self, "*", other), format="jump")

    def __eq__(self, other):
        return OmltExpression(expr=(self, "==", other), format="jump")
