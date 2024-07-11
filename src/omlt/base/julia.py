from omlt.base.var import OmltIndexed, OmltScalar
from omlt.dependencies import julia_available

if julia_available:
    from juliacall import Base
    from juliacall import Main as jl

    jl_err = Base.error
    jl.seval("import JuMP")
    jump = jl.JuMP

# Elements

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
        self.omltvar = None
        self.index = None
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
            name = self.name
        variable_ref = jump.add_variable(model, self.var, name)
        return variable_ref

    def to_jump(self):
        return self.var

    def __add__(self, other):
        return (self.omltvar + other)[self.index]

    def __sub__(self, other):
        return (self.omltvar - other)[self.index]

    def __mul__(self, other):
        return (self.omltvar * other)[self.index]

    def __eq__(self, other):
        return (self.omltvar == other)[self.index]

# Variables

class OmltScalarJuMP(OmltScalar):
    format = "jump"

    # Claim to be a Pyomo Var so blocks will register
    # properly.
    @property
    def __class__(self):
        return pyo.ScalarVar

    def __init__(self, **kwargs: Any):
        self._block = kwargs.pop("block", None)

        self._bounds = kwargs.pop("bounds", None)

        if isinstance(self._bounds, tuple) and len(self._bounds) == 2:
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

        if _domain and _within and _domain != _within:
            msg = (
                "'domain' and 'within' keywords have both "
                "been supplied and do not agree. Please try "
                "with a single keyword for the domain of this "
                "variable."
            )
            raise ValueError(msg)
        if _domain:
            self.domain = _domain
        elif _within:
            self.domain = _within
        else:
            self.domain = None

        if self.domain == pyo.Binary:
            self.binary = True
        else:
            self.binary = False
        if self.domain == pyo.Integers:
            self.integer = True
        else:
            self.integer = False

        _initialize = kwargs.pop("initialize", None)

        if _initialize:
            if isinstance(_initialize, (int, float)):
                self._value = _initialize
            elif len(_initialize) == 1 and isinstance(_initialize[0], (int, float)):
                self._value = _initialize[0]
            else:
                # Pyomo's "scalar" variables can be multidimensional, they're
                # just not indexed. JuMP scalar variables can only be a single
                # dimension. Rewrite this error to be more helpful.
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
            self.binary,
            self.integer,
        )
        self._constructed = False
        self._parent = None
        self._ctype = pyo.ScalarVar
        self._name = None

    def construct(self, data=None):
        self._var = JumpVar(self._varinfo, self._name)
        self._var.omltvar = self
        self._constructed = True
        if self._parent:
            self._blockvar = jump.add_variable(
                self._parent()._jumpmodel, self.to_jumpvar()
            )

    def fix(self, value, *, skip_validation=True):
        self.fixed = True
        self._value = value
        self._varinfo.fixed_value = value
        self._varinfo.has_fix = value is not None
        if self._constructed:
            self.construct()

    @property
    def bounds(self):
        return (self.lb, self.ub)

    @bounds.setter
    def bounds(self, val):
        if val is None:
            self.lb = None
            self.ub = None
        elif len(val) == 2:
            self.lb = val[0]
            self.ub = val[1]

    @property
    def lb(self):
        return self._varinfo.lower_bound

    @lb.setter
    def lb(self, val):
        self._varinfo.setlb(val)
        if self._constructed:
            self.construct()

    @property
    def ub(self):
        return self._varinfo.upper_bound

    @ub.setter
    def ub(self, val):
        self._varinfo.setub(val)
        if self._constructed:
            self.construct()

    @property
    def value(self):
        if self._constructed:
            return self._var.value
        return self._varinfo.start_value

    @value.setter
    def value(self, val):
        if self._constructed:
            self._var.value = val
        else:
            self._varinfo.start_value = val

    @property
    def ctype(self):
        return self._ctype

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def to_jumpvar(self):
        if self._constructed:
            return self._var.to_jump()
        return self._varinfo.to_jump()

    def to_jumpexpr(self):
        return jump.AffExpr(0, jump.OrderedDict([(self._blockvar, 1)]))

class OmltIndexedJuMP(OmltIndexed):
    format = "jump"

    # Claim to be a Pyomo Var so blocks will register
    # properly.
    @property
    def __class__(self):
        return pyo.Var

    def __init__(self, *indexes, **kwargs: Any):
        if len(indexes) == 1:
            index_set = indexes[0]
            i_dict = {}
            for i, val in enumerate(index_set):
                i_dict[i] = val
            self._index_set = tuple(i_dict[i] for i in range(len(index_set)))
        else:
            msg = ("Currently index cross-products are unsupported.")
            raise ValueError(msg)

        self._block = kwargs.pop("block", None)

        self._bounds = kwargs.pop("bounds", None)

        if isinstance(self._bounds, dict) and len(self._bounds) == len(self._index_set):
            _lb = {k: v[0] for k, v in self._bounds.items()}
            _ub = {k: v[1] for k, v in self._bounds.items()}
        elif isinstance(self._bounds, tuple) and len(self._bounds) == 2:
            _lb = {i: self._bounds[0] for i in self._index_set}
            _ub = {i: self._bounds[1] for i in self._index_set}
        elif self._bounds is None:
            _lb = {i: None for i in self._index_set}
            _ub = {i: None for i in self._index_set}
        else:
            msg = (
                "Bounds must be given as a tuple," " but %s was given.", self._bounds
            )
            raise TypeError(msg)

        _domain = kwargs.pop("domain", None)
        _within = kwargs.pop("within", None)

        if _domain and _within and _domain != _within:
            msg = (
                "'domain' and 'within' keywords have both "
                "been supplied and do not agree. Please try "
                "with a single keyword for the domain of this "
                "variable."
            )
            raise ValueError(msg)
        if _domain:
            self.domain = _domain
        elif _within:
            self.domain = _within
        else:
            self.domain = None

        if self.domain == pyo.Binary:
            self.binary = True
        else:
            self.binary = False
        if self.domain == pyo.Integers:
            self.integer = True
        else:
            self.integer = False

        _initialize = kwargs.pop("initialize", None)

        if _initialize:
            # If starting values have same length as index set,
            # take one for each variable in index.
            if len(self._index_set) == len(_initialize):
                self._value = _initialize
            # If there's a single starting value, use it for all
            # variables in index.
            elif len(_initialize) == 1:
                self._value = {i: _initialize[0] for i in self._index_set}
            else:
                msg = (
                    "Index set has length %s, but initializer has length %s.",
                    len(self._index_set),
                    len(_initialize),
                )
                raise ValueError(msg)
        else:
            self._value = {i: None for i in self._index_set}

        self._varinfo = {}
        for idx in self._index_set:
            self._varinfo[idx] = JuMPVarInfo(
                _lb[idx],
                _ub[idx],
                None,  # fix value
                self._value[idx],
                self.binary,
                self.integer,
            )
        self._vars = {}
        self._varrefs = {}
        self._constructed = False
        self._ctype = pyo.Var
        self._parent = None

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 1:
            return self._vars[item[0]]
        return self._vars[item]

    def __setitem__(self, item, value):
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

    def fix(self, value=None):
        self.fixed = True
        if value is not None:
            for vardata in self._varinfo():
                vardata.has_fix = True
                vardata.fixed_value = value
        else:
            for vardata in self._varinfo():
                vardata.has_fix = True

    def __len__(self):
        """Return the number of component data objects stored by this component."""
        return len(self._vars)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary."""
        return idx in self._vars

    # The default implementation is for keys() and __iter__ to be
    # synonyms.  The logic is implemented in keys() so that
    # keys/values/items continue to work for components that implement
    # other definitions for __iter__ (e.g., Set)
    def __iter__(self):
        """Return an iterator of the component data keys."""
        return self._vars.__iter__()

    def construct(self, data=None):
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

    def setub(self, value):
        for idx in self.index_set():
            self._varinfo[idx][2] = True
            self._varinfo[idx][3] = value
        if self._constructed:
            self.construct()

    def setlb(self, value):
        for idx in self.index_set():
            self._varinfo[idx][0] = True
            self._varinfo[idx][1] = value
        if self._constructed:
            self.construct()

    @property
    def ctype(self):
        return self._ctype

    def index_set(self):
        return self._index_set

    @property
    def name(self):
        return self._name

    def to_jumpvar(self):
        if self._constructed:
            return jump.Containers.DenseAxisArray(list(self.values()), self.index_set())
        msg = "Variable must be constructed before exporting to JuMP."
        raise ValueError(msg)

    def to_jumpexpr(self):
        return {k: jump.AffExpr(0, jump.OrderedDict([(v, 1)])) for k, v in self.items()}


# Constraints

# Expressions