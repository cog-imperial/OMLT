from typing import Any

from numpy import float32
from pyomo.core.base import ParamData

from omlt.base.constraint import OmltConstraintIndexed, OmltConstraintScalar
from omlt.base.expression import OmltExpr
from omlt.base.var import OmltElement, OmltIndexed, OmltScalar
from omlt.block import OmltBlockCore
from omlt.dependencies import julia_available

if julia_available:
    from juliacall import Base, convert
    from juliacall import Main as Jl

    jl_err = Base.error
    Jl.seval("import Pkg")
    Jl.Pkg.add("JuMP")
    Jl.seval("import JuMP")
    jump = Jl.JuMP

PAIR = 2
EXPR_TWO = 2
EXPR_THREE = 3

# Elements


class JuMPVarInfo:
    def __init__( # noqa: PLR0913
        self,
        lower_bound=None,
        upper_bound=None,
        fixed_value=None,
        start_value=None,
        *,
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


class JumpVar(OmltElement):
    def __init__(self, varinfo: JuMPVarInfo, name):
        self.info = varinfo
        self.name = name
        self.omltvar = None
        self.index = None
        self.construct()

    def setlb(self, value):
        self.info.setlb(value)
        self.construct()

    def setub(self, value):
        self.info.setub(value)
        self.construct()

    def construct(self):
        self.var = jump.build_variable(jl_err, self.info.to_jump())

    @property
    def lb(self):
        return self.info.lb

    @property
    def ub(self):
        return self.info.ub

    @property
    def bounds(self):
        return self.info.lb, self.info.ub

    @property
    def value(self):
        return self.var.info.start

    @value.setter
    def value(self, val):
        self.info.start_value = val
        self.info.has_start = val is not None
        self.construct()

    def add_to_model(self, model, name=None):
        self.varref = jump.add_variable(model, self.var, name)
        return self.varref

    def to_jump(self):
        return self.var

    def __neg__(self):
        return OmltExprJuMP((self, "*", -1))

    def __add__(self, other):
        return OmltExprJuMP((self, "+", other))

    def __sub__(self, other):
        return OmltExprJuMP((self, "-", other))

    def __mul__(self, other):
        return OmltExprJuMP((self, "*", other))

    def __radd__(self, other):
        return OmltExprJuMP((self, "+", other))

    def __rsub__(self, other):
        return OmltExprJuMP((other, "-", self))

    def __rmul__(self, other):
        return OmltExprJuMP((self, "*", other))

    def __eq__(self, other):
        return OmltConstraintScalarJuMP(lhs=self, rhs=other, sense="==")

    def __ge__(self, other):
        return OmltConstraintScalarJuMP(lhs=self, rhs=other, sense=">=")

    def exp(self):
        return OmltExprJuMP(("exp", self))

    def log(self):
        return OmltExprJuMP(("log", self))

    def tanh(self):
        return OmltExprJuMP(("tanh", self))


# Variables


class OmltScalarJuMP(OmltScalar):
    format = "jump"

    def __init__(self, *, binary=False, **kwargs: Any):
        super().__init__()

        self._bounds = kwargs.pop("bounds", None)
        if isinstance(self._bounds, tuple) and len(self._bounds) == PAIR:
            _lb = self._bounds[0]
            _ub = self._bounds[1]
        elif self._bounds is None:
            _lb = None
            _ub = None
        else:
            msg = ("Bounds must be given as a tuple.", self._bounds)
            raise ValueError(msg)

        _domain = kwargs.pop("domain", None)
        _within = kwargs.pop("within", None)

        if _domain:
            self._domain = _domain
        elif _within:
            self._domain = _within
        else:
            self._domain = None

        self.binary = binary

        _initialize = kwargs.pop("initialize", None)

        if _initialize:
            if isinstance(_initialize, (int, float)):
                self._value = _initialize
            elif len(_initialize) == 1 and isinstance(_initialize[0], (int, float)):
                self._value = _initialize[0]
            else:
                msg = (
                    "Initial value for JuMP variables must be an int"
                    f" or float, but {type(_initialize)} was provided."
                )
                raise ValueError(msg)
        else:
            self._value = None

        self._varinfo = JuMPVarInfo(
            _lb,
            _ub,
            None,  # fix value
            self._value,
            binary=self.binary,
            integer=False,
        )
        self._constructed = False
        self._parent = None
        self._name = None

        self._var = JumpVar(self._varinfo, self._name)
        self._var.omltvar = self

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, val):
        self._domain = val

    @property
    def varref(self):
        return self._varref

    @varref.setter
    def varref(self, value):
        self._varref = value
        self._var.varref = value

    @property
    def lb(self):
        return self._varinfo.lower_bound

    @lb.setter
    def lb(self, val):
        self._varinfo.setlb(val)
        self._var.setlb(val)

    @property
    def ub(self):
        return self._varinfo.upper_bound

    @ub.setter
    def ub(self, val):
        self._varinfo.setub(val)
        self._var.setub(val)

    @property
    def value(self):
        return self._var.value


    @value.setter
    def value(self, val):
        self._var.value = val


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._var.name = value

    def to_jumpvar(self):
        return self._var.to_jump()


class OmltIndexedJuMP(OmltIndexed):
    format = "jump"

    def __init__(self, *indexes: Any, binary: bool = False, **kwargs: Any):
        index_set = indexes[0]
        i_dict = dict(enumerate(index_set))
        self._index_set = tuple(i_dict[i] for i in range(len(index_set)))

        self._bounds = kwargs.pop("bounds", None)

        if isinstance(self._bounds, dict) and len(self._bounds) == len(self._index_set):
            _lb = {k: v[0] for k, v in self._bounds.items()}
            _ub = {k: v[1] for k, v in self._bounds.items()}
        elif isinstance(self._bounds, tuple) and len(self._bounds) == PAIR:
            _lb = {i: self._bounds[0] for i in self._index_set}
            _ub = {i: self._bounds[1] for i in self._index_set}
        elif self._bounds is None:
            _lb = {i: None for i in self._index_set}
            _ub = {i: None for i in self._index_set}
        else:
            msg = (
                "Bounds must be given as a tuple," " but %s was given.",
                self._bounds,
            )
            raise TypeError(msg)

        self.binary = binary

        self._value = {i: None for i in self._index_set}

        self._varinfo = {}
        self._vars = {}
        for idx in self._index_set:
            self._varinfo[idx] = JuMPVarInfo(
                _lb[idx],
                _ub[idx],
                None,  # fix value
                self._value[idx],
                binary=self.binary,
                integer=False,
            )
            self._vars[idx] = JumpVar(self._varinfo[idx], str(idx))
            self._vars[idx].omltvar = self
            self._vars[idx].index = idx
        self._varrefs = {}
        self._constructed = False
        self._parent = None
        self._name = None

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, val):
        self._domain = val

    def __getitem__(self, item):
        if item in self._index_set:
            return self._vars[item]
        if isinstance(item, tuple) and len(item) == 1:
            return self._vars[item[0]]
        return self._vars[item]

    def __setitem__(self, item, value):
        if isinstance(item, tuple) and len(item) == 1:
            self._varinfo[item[0]] = value
        else:
            self._varinfo[item] = value
        if self._constructed:
            self.construct()

    def keys(self):
        if self._parent is not None:
            return self._varrefs.keys()
        return self._vars.keys()

    def values(self):
        if self._parent is not None:
            return self._varrefs.values()
        return self._vars.values()

    def items(self):
        if self._parent is not None:
            return self._varrefs.items()
        return self._vars.items()

    def __len__(self):
        """Return the number of component data objects stored by this component."""
        return len(self._vars)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary."""
        return idx in self._vars

    def __iter__(self):
        """Return an iterator of the component data keys."""
        return self._vars.__iter__()

    def construct(self, *, data=None): # noqa: ARG002
        for idx in self._index_set:
            if isinstance(idx, int):
                name = str(self.name) + "[" + str(idx) + "]"
            else:
                name = str(self.name) + str(list(idx)).replace(" ", "")
            self._vars[idx] = JumpVar(self._varinfo[idx], name)
            self._vars[idx].omltvar = self
            self._vars[idx].index = idx
            if self._parent is not None:
                block = self._parent()
                if block._format == "jump" and block._jumpmodel is not None:
                    self._varrefs[idx] = self._vars[idx].add_to_model(block._jumpmodel)

        self._constructed = True

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.construct()


# Constraints
class OmltConstraintScalarJuMP(OmltConstraintScalar):
    format = "jump"

    def __init__(self, **kwargs: Any):
        super().__init__(lang="jump", **kwargs)
        if self.sense == "==":
            self._jumpcon = jump.build_constraint(
                jl_err, (self.lhs - self.rhs)._jumpexpr, jump.Zeros()
            )
        if self.sense == "<=":
            self._jumpcon = jump.build_constraint(
                jl_err, (self.lhs - self.rhs)._jumpexpr, jump.Nonpositives()
            )
        if self.sense == ">=":
            self._jumpcon = jump.build_constraint(
                jl_err, (self.lhs - self.rhs)._jumpexpr, jump.Nonnegatives()
            )


class OmltConstraintIndexedJuMP(OmltConstraintIndexed):
    format = "jump"

    def __init__(self, *indexes: Any, **kwargs: Any):
        self._index_set = indexes

        self.model = kwargs.pop("model", None)
        self._parent = None
        self.name = None
        self.format = "jump"
        self._jumpcons = {idx: None for idx in self._index_set[0]}

    def keys(self, *, sort=False): # noqa: ARG002
        yield from self._index_set

    def __setitem__(self, label, item):
        self._jumpcons[label] = item
        if self.model is not None and self.name is not None:
            jump.add_constraint(
                self.model._jumpmodel, item._jumpcon, self.name + "[" + str(label) + "]"
            )

    def __getitem__(self, label):
        return self._jumpcons[label]


# Expressions


class OmltExprJuMP(OmltExpr):
    format = "jump"

    def __init__(self, expr):
        """Build an expression from a tuple."""
        msg = (
            "Tried to create an OmltExprJuMP with an invalid expression. Expressions "
            "must be tuples (a, b, c) where b is +, -, *, or /, or tuples (d, e) where "
            "d is exp, log, or tanh. %s was provided",
            expr,
        )
        if len(expr) == EXPR_THREE:
            if expr[1] == "+":
                self._jumpexpr = self.add(expr[0], expr[2])
            elif expr[1] == "-":
                self._jumpexpr = self.subtract(expr[0], expr[2])
            elif expr[1] == "*":
                self._jumpexpr = self.multiply(expr[0], expr[2])
            elif expr[1] == "/":
                self._jumpexpr = self.divide(expr[0], expr[2])
            else:
                raise ValueError(msg)
        elif len(expr) == EXPR_TWO:
            if expr[0] == "exp":
                self._jumpexpr = self.exponent(expr[1])
            elif expr[0] == "log":
                self._jumpexpr = self.logarithm(expr[1])
            elif expr[0] == "tanh":
                self._jumpexpr = self.hyptangent(expr[1])
            else:
                raise ValueError(msg)
        else:
            raise ValueError(msg)

    def add(self, a, b):
        if isinstance(a, (JumpVar, OmltScalarJuMP)) and isinstance(b, (int, float)):
            return jump.AffExpr(b, jump.OrderedDict([(a.varref, 1)]))

        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, (int, float, float32))
        ):
            return jump.AffExpr(
                b + a._jumpexpr.constant,
                jump.OrderedDict(
                    [(var, a._jumpexpr.terms[var]) for var in a._jumpexpr.terms]
                ),
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, OmltExprJuMP)
            and Jl.typeof(b._jumpexpr) == jump.AffExpr
        ):
            return a._jumpexpr + b._jumpexpr
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.NonlinearExpr
            and isinstance(b, (int, float, float32))
        ):
            return jump.NonlinearExpr(
                convert(Jl.Symbol, "+"), convert(Jl.Vector, [a._jumpexpr, b])
            )
        msg = ("Unrecognized types for addition, %s, %s", type(a), type(b))
        raise TypeError(msg)

    def subtract(self, a, b): # noqa: PLR0911
        if isinstance(a, (int, float)) and isinstance(b, (JumpVar, OmltScalarJuMP)):
            return jump.AffExpr(a, jump.OrderedDict([(b.varref, -1)]))
        if isinstance(a, JumpVar) and isinstance(b, (int, float)):
            return jump.AffExpr(-b, jump.OrderedDict([(a.varref, 1)]))
        if isinstance(a, JumpVar) and isinstance(b, JumpVar):
            return jump.AffExpr(0, jump.OrderedDict([(a.varref, 1), (b.varref, -1)]))
        if isinstance(a, JumpVar) and isinstance(b, OmltExprJuMP):
            return self.multiply(b - a, -1)
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, (int, float))
        ):
            return jump.AffExpr(
                a._jumpexpr.constant - b,
                jump.OrderedDict(
                    [(var, a._jumpexpr.terms[var]) for var in a._jumpexpr.terms]
                ),
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, JumpVar)
            and b.varref not in a._jumpexpr.terms
        ):
            return jump.AffExpr(
                a._jumpexpr.constant,
                jump.OrderedDict(
                    [(var, a._jumpexpr.terms[var]) for var in a._jumpexpr.terms]
                    + [(b.varref, -1)]
                ),
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, OmltExprJuMP)
        ):
            return a._jumpexpr - b._jumpexpr
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.NonlinearExpr
            and isinstance(b, (int, float))
        ):
            return jump.NonlinearExpr(
                convert(Jl.Symbol, "-"), convert(Jl.Vector, (a._jumpexpr, b))
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.NonlinearExpr
            and isinstance(b, JumpVar)
        ):
            return a._jumpexpr - b.varref

        msg = ("Unrecognized types for subtraction, %s, %s", type(a), type(b))
        raise TypeError(msg)

    def multiply(self, a, b):
        if isinstance(a, (int, float)) and isinstance(b, (JumpVar, OmltScalarJuMP)):
            return jump.AffExpr(0, jump.OrderedDict([(b.varref, a)]))
        if isinstance(a, (JumpVar, OmltScalarJuMP)):
            if isinstance(b, (int, float, float32)):
                return jump.AffExpr(0, jump.OrderedDict([(a.varref, b)]))
            if isinstance(b, ParamData):
                return jump.AffExpr(0, jump.OrderedDict([(a.varref, b.value)]))
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, (int, float, float32))
        ):
            return jump.AffExpr(
                a._jumpexpr.constant * b,
                jump.OrderedDict(
                    [(var, a._jumpexpr.terms[var] * b) for var in a._jumpexpr.terms]
                ),
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, ParamData)
        ):
            return jump.AffExpr(
                a._jumpexpr.constant * b.value,
                jump.OrderedDict(
                    [
                        (var, a._jumpexpr.terms[var] * b.value)
                        for var in a._jumpexpr.terms
                    ]
                ),
            )
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.NonlinearExpr
            and isinstance(b, (int, float, float32))
        ):
            return jump.NonlinearExpr(
                convert(Jl.Symbol, "*"), convert(Jl.Vector, (a._jumpexpr, b))
            )

        msg = ("Unrecognized types for multiplication, %s, %s", type(a), type(b))
        raise TypeError(msg)

    def divide(self, a, b):
        if (
            isinstance(a, OmltExprJuMP)
            and Jl.typeof(a._jumpexpr) == jump.AffExpr
            and isinstance(b, (int, float))
        ):
            return jump.AffExpr(
                a._jumpexpr.constant / b,
                jump.OrderedDict(
                    [(var, a._jumpexpr.terms[var] / b) for var in a._jumpexpr.terms]
                ),
            )
        if isinstance(b, OmltExprJuMP):
            return jump.NonlinearExpr(
                convert(Jl.Symbol, "/"), convert(Jl.Vector, (a, b._jumpexpr))
            )
        msg = ("Unrecognized types for division, %s, %s", type(a), type(b))
        raise TypeError(msg)

    def exponent(self, a):
        if isinstance(a, OmltExprJuMP):
            return jump.NonlinearExpr(convert(Jl.Symbol, "exp"), a._jumpexpr)
        if isinstance(a, (JumpVar, OmltScalarJuMP)):
            return jump.NonlinearExpr(convert(Jl.Symbol, "exp"), a.varref)
        raise NotImplementedError

    def logarithm(self, a):
        if isinstance(a, OmltExprJuMP):
            return jump.NonlinearExpr(convert(Jl.Symbol, "log"), a._jumpexpr)
        if isinstance(a, (JumpVar, OmltScalarJuMP)):
            return jump.NonlinearExpr(convert(Jl.Symbol, "log"), a.varref)
        raise NotImplementedError

    def hyptangent(self, a):
        if isinstance(a, OmltExprJuMP):
            return jump.NonlinearExpr(convert(Jl.Symbol, "tanh"), a._jumpexpr)
        if isinstance(a, (JumpVar, OmltScalarJuMP)):
            return jump.NonlinearExpr(convert(Jl.Symbol, "tanh"), a.varref)
        raise NotImplementedError

    def exp(self):
        return OmltExprJuMP(("exp", self))

    def log(self):
        return OmltExprJuMP(("log", self))

    def tanh(self):
        return OmltExprJuMP(("tanh", self))

    def __call__(self):
        """Return the current value of the expression."""
        return self._jumpexpr.constant + sum(
            [
                self._jumpexpr.terms[v] * jump.start_value(v)
                for v in self._jumpexpr.terms
            ]
        )

    def __add__(self, other):
        return OmltExprJuMP((self, "+", other))

    def __sub__(self, other):
        return OmltExprJuMP((self, "-", other))

    def __mul__(self, other):
        return OmltExprJuMP((self, "*", other))

    def __truediv__(self, other):
        return OmltExprJuMP((self, "/", other))

    def __radd__(self, other):
        return OmltExprJuMP((self, "+", other))

    def __rmul__(self, other):
        return OmltExprJuMP((self, "*", other))

    def __rtruediv__(self, other):
        return OmltExprJuMP((other, "/", self))

    def __eq__(self, other):
        return OmltConstraintScalarJuMP(lhs=self, sense="==", rhs=other)

    def __le__(self, other):
        return OmltConstraintScalarJuMP(lhs=self, sense="<=", rhs=other)

    def __ge__(self, other):
        return OmltConstraintScalarJuMP(lhs=self, sense=">=", rhs=other)


# Block
class OmltBlockJuMP(OmltBlockCore):
    def __init__(self):
        self.__formulation = None
        self.__input_indexes = None
        self.__output_indexes = None
        self._format = "jump"
        self._jumpmodel = jump.Model()
        self._varrefs = {}
        self._conrefs = {}

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if isinstance(value, OmltScalarJuMP):
            value.name = name
            self._varrefs[name] = jump.add_variable(
                self._jumpmodel, value.to_jumpvar(), name
            )
            value.varref = self._varrefs[name]
        elif isinstance(value, OmltIndexedJuMP):
            value.name = name
            for idx, var in value._vars.items():
                varname = name + "_" + str(idx)
                self.__getattribute__(name)[idx].varref = var.add_to_model(
                    self._jumpmodel, name=varname
                )
                self._varrefs[varname] = self.__getattribute__(name)[idx].varref

        elif isinstance(value, OmltConstraintScalarJuMP):
            self._conrefs[name] = jump.add_constraint(
                self._jumpmodel, value._jumpcon, name
            )
        elif isinstance(value, OmltConstraintIndexedJuMP):
            value.model = self
            value.name = name
        elif isinstance(value, OmltBlockCore):
            value.name = name
            value._parent = self
        elif isinstance(value, dict):
            for k, v in value.items():
                v.name = name + "_" + str(k) + "_"
                v._parent = self

    def set_optimizer(self, optimizer):
        jump.set_optimizer(self._jumpmodel, optimizer)

    def get_model(self):
        return self._jumpmodel
