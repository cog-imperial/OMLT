"""
Abstraction layer of classes used by OMLT. Underneath these are
objects in a choice of modeling languages: Pyomo (default),
JuMP, or others (not yet implemented - e.g. Smoke, Gurobi).


"""

from abc import ABC, abstractmethod
import pyomo.environ as pyo

from omlt.dependencies import julia_available

from omlt.base import DEFAULT_MODELING_LANGUAGE

if julia_available:
    from omlt.base import jump
from omlt.base.julia import JuMPVarInfo, JumpVar
from omlt.base.expression import OmltExpr, OmltExprIndexed, OmltExprScalar
# from omlt.base.constraint import OmltRelation, OmltRelScalar


class OmltVar(ABC):
    def __new__(cls, *indexes, **kwargs):

        if not indexes:
            instance = OmltScalar.__new__(OmltScalar, **kwargs)
        else:
            instance = OmltIndexed.__new__(OmltIndexed, *indexes, **kwargs)
        return instance

    @abstractmethod
    def construct(self, data):
        pass

    @abstractmethod
    def fix(self, value, skip_validation):
        pass

    @property
    @abstractmethod
    def ctype(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    # Some methods to tell OMLT (and Pyomo components) that this
    # is a variable.
    def is_component_type(self):
        return True

    @abstractmethod
    def is_indexed(self):
        pass

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True


class OmltScalar(OmltVar):
    def __new__(cls, *args, format=DEFAULT_MODELING_LANGUAGE, **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                "Variable format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltVar, subclass).__new__(subclass)

        instance.__init__(*args, **kwargs)
        instance._format = format
        return instance

    def is_indexed(self):
        return False

    # Bound-setting interface for scalar variables:
    @property
    @abstractmethod
    def bounds(self):
        pass

    @bounds.setter
    @abstractmethod
    def bounds(self, val):
        pass

    @property
    @abstractmethod
    def lb(self):
        pass

    @lb.setter
    @abstractmethod
    def lb(self, val):
        pass

    @property
    @abstractmethod
    def ub(self):
        pass

    @ub.setter
    @abstractmethod
    def ub(self, val):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @domain.setter
    @abstractmethod
    def domain(self, val):
        pass

    # Interface for getting/setting value
    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, val):
        pass

    # Interface governing how variables behave in expressions.

    # def __lt__(self, other):
    #     return OmltRelScalar(expr=(self, "<", other))

    # def __gt__(self, other):
    #     return OmltRelScalar(expr=(self, ">", other))

    # def __le__(self, other):
    #     return OmltRelScalar(expr=(self, "<=", other))

    # def __ge__(self, other):
    #     return OmltRelScalar(expr=(self, ">=", other))

    # def __eq__(self, other):
    #     return OmltRelScalar(expr=(self, "==", other))

    def __add__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "+", other))

    def __sub__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "-", other))

    def __mul__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "*", other))

    def __div__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "//", other))

    def __truediv__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "/", other))

    def __pow__(self, other):
        return OmltExprScalar(format=self._format, expr=(self, "**", other))

    def __radd__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "+", self))

    def __rsub__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "-", self))

    def __rmul__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "*", self))

    def __rdiv__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "//", self))

    def __rtruediv__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "/", self))

    def __rpow__(self, other):
        return OmltExprScalar(format=self._format, expr=(other, "**", self))

    def __iadd__(self, other):
        return pyo.NumericValue.__iadd__(self, other)

    def __isub__(self, other):
        return pyo.NumericValue.__isub__(self, other)

    def __imul__(self, other):
        return pyo.NumericValue.__imul__(self, other)

    def __idiv__(self, other):
        return pyo.NumericValue.__idiv__(self, other)

    def __itruediv__(self, other):
        return pyo.NumericValue.__itruediv__(self, other)

    def __ipow__(self, other):
        return pyo.NumericValue.__ipow__(self, other)

    def __neg__(self):
        return pyo.NumericValue.__neg__(self)

    def __pos__(self):
        return pyo.NumericValue.__pos__(self)

    def __abs__(self):
        return pyo.NumericValue.__abs__(self)


class OmltScalarPyomo(OmltScalar, pyo.ScalarVar):
    format = "pyomo"

    def __init__(self, *args, **kwargs):
        kwargs.pop("format", None)
        # pyo.ScalarVar.__init__(self, *args, **kwargs)
        self._pyovar = pyo.ScalarVar(*args, **kwargs)
        self._parent = None
        self._constructed = None

    def construct(self, data=None):
        return self._pyovar.construct(data)

    def fix(self, value, skip_validation):
        return self._pyovar.fix(value, skip_validation)

    @property
    def ctype(self):
        return pyo.ScalarVar

    @property
    def name(self):
        self._pyovar._name = self._name
        return self._pyovar._name

    @property
    def bounds(self):
        return (self._pyovar._lb, self._pyovar._ub)

    @bounds.setter
    def bounds(self, val):
        self._pyovar.lb = val[0]
        self._pyovar.ub = val[1]

    @property
    def lb(self):
        return self._pyovar._lb

    @lb.setter
    def lb(self, val):
        self._pyovar.setlb(val)

    @property
    def ub(self):
        return self._pyovar._ub

    @ub.setter
    def ub(self, val):
        self._pyovar.setub(val)

    @property
    def domain(self):
        return self._pyovar._domain

    @domain.setter
    def domain(self, val):
        self._pyovar._domain = val

    # Interface for getting/setting value
    @property
    def value(self):
        return self._pyovar.value

    @value.setter
    def value(self, val):
        self._pyovar.value = val


class OmltScalarJuMP(OmltScalar):
    format = "jump"

    # Claim to be a Pyomo Var so blocks will register
    # properly.
    @property
    def __class__(self):
        return pyo.ScalarVar

    def __init__(self, *args, **kwargs):

        self._block = kwargs.pop("block", None)

        self._bounds = kwargs.pop("bounds", None)

        if isinstance(self._bounds, tuple) and len(self._bounds) == 2:
            _lb = self._bounds[0]
            _ub = self._bounds[1]
        elif self._bounds is None:
            _lb = None
            _ub = None
        else:
            raise ValueError("Bounds must be given as a tuple")

        _domain = kwargs.pop("domain", None)
        _within = kwargs.pop("within", None)

        if _domain and _within and _domain != _within:
            raise ValueError(
                "'domain' and 'within' keywords have both "
                "been supplied and do not agree. Please try "
                "with a single keyword for the domain of this "
                "variable."
            )
        elif _domain:
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
                raise ValueError(
                    "Initial value for JuMP variables must be an int"
                    f" or float, but {type(_initialize)} was provided."
                )
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

    def fix(self, value, skip_validation):
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
        else:
            return self._varinfo.start_value

    @value.setter
    def value(self, val):
        if self._constructed:
            self._var.value = val
        else:
            self._varinfo.start_value = val
            self

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
        else:
            return self._varinfo.to_jump()

    def to_jumpexpr(self):
        return jump.AffExpr(0, jump.OrderedDict([(self._blockvar, 1)]))


"""
Future formats to implement.
"""


class OmltScalarSmoke(OmltScalar):
    format = "smoke"

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Storing variables in Smoke format is not currently implemented."
        )


class OmltScalarGurobi(OmltScalar):
    format = "gurobi"

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Storing variables in Gurobi format is not currently implemented."
        )


class OmltIndexed(OmltVar):
    def __new__(cls, *indexes, format=DEFAULT_MODELING_LANGUAGE, **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                "Variable format %s not recognized. Supported formats are 'pyomo'"
                " or 'jump'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltVar, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        instance._format = format
        return instance

    def is_indexed(self):
        return True

    @property
    @abstractmethod
    def index_set(self):
        pass

    # Bound-setting interface for indexed variables:
    @abstractmethod
    def setub(self, value):
        pass

    @abstractmethod
    def setlb(self, value):
        pass

    # Interface: act as a dict for the sub-variables.
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __setitem__(self, item, value):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def values(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __contains__(self, idx):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    # Interface governing how variables behave in expressions.

    # def __lt__(self, other):
    #     return OmltRelation(self.index_set(), expr=(self, "<", other))

    # def __gt__(self, other):
    #     return OmltRelation(self.index_set(), expr=(self, ">", other))

    # def __le__(self, other):
    #     return OmltRelation(self.index_set(), expr=(self, "<=", other))

    # def __ge__(self, other):
    #     return OmltRelation(self.index_set(), expr=(self, ">=", other))

    # def __eq__(self, other):
    #     return OmltRelation(self.index_set(), expr=(self, "==", other))

    def __add__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "+", other))

    def __sub__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "-", other))

    def __mul__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "*", other))

    def __div__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "//", other))

    def __truediv__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "/", other))

    def __pow__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(self, "**", other))

    def __radd__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "+", self))

    def __rsub__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "-", self))

    def __rmul__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "*", self))

    def __rdiv__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "//", self))

    def __rtruediv__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "/", self))

    def __rpow__(self, other):
        return OmltExprIndexed(self.index_set(), expr=(other, "**", self))

    def __iadd__(self, other):
        return pyo.NumericValue.__iadd__(self, other)

    def __isub__(self, other):
        return pyo.NumericValue.__isub__(self, other)

    def __imul__(self, other):
        return pyo.NumericValue.__imul__(self, other)

    def __idiv__(self, other):
        return pyo.NumericValue.__idiv__(self, other)

    def __itruediv__(self, other):
        return pyo.NumericValue.__itruediv__(self, other)

    def __ipow__(self, other):
        return pyo.NumericValue.__ipow__(self, other)

    def __neg__(self):
        return pyo.NumericValue.__neg__(self)

    def __pos__(self):
        return pyo.NumericValue.__pos__(self)

    def __abs__(self):
        return pyo.NumericValue.__abs__(self)


class OmltIndexedPyomo(pyo.Var, OmltIndexed):
    format = "pyomo"

    def __init__(self, *indexes, **kwargs):
        kwargs.pop("format", None)
        super().__init__(*indexes, **kwargs)

    def fix(self, value=None, skip_validation=False):
        self.fixed = True
        if value is None:
            for vardata in self.values():
                vardata.fix(skip_validation)
        else:
            for vardata in self.values():
                vardata.fix(value, skip_validation)

    def setub(self, value):
        for vardata in self.values():
            vardata.ub = value

    def setlb(self, value):
        for vardata in self.values():
            vardata.lb = value


class OmltIndexedJuMP(OmltIndexed):
    format = "jump"

    # Claim to be a Pyomo Var so blocks will register
    # properly.
    @property
    def __class__(self):
        return pyo.Var

    def __init__(self, *indexes, **kwargs):
        if len(indexes) == 1:
            index_set = indexes[0]
            i_dict = {}
            for i, val in enumerate(index_set):
                i_dict[i] = val
            self._index_set = tuple(i_dict[i] for i in range(len(index_set)))
        else:
            raise ValueError("Currently index cross-products are unsupported.")

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
            raise ValueError(
                "Bounds must be given as a tuple," " but %s was given.", self._bounds
            )

        _domain = kwargs.pop("domain", None)
        _within = kwargs.pop("within", None)

        if _domain and _within and _domain != _within:
            raise ValueError(
                "'domain' and 'within' keywords have both "
                "been supplied and do not agree. Please try "
                "with a single keyword for the domain of this "
                "variable."
            )
        elif _domain:
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
                raise ValueError(
                    "Index set has length %s, but initializer has length %s.",
                    len(self._index_set),
                    len(_initialize),
                )
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
        else:
            return self._vars[item]

    def __setitem__(self, item, value):
        self._varinfo[item] = value
        if self._constructed:
            self.construct()

    def keys(self):
        if self._parent is not None:
            return self._varrefs.keys()
        else:
            return self._vars.keys()

    def values(self):
        if self._parent is not None:
            return self._varrefs.values()
        else:
            return self._vars.values()

    def items(self):
        if self._parent is not None:
            return self._varrefs.items()
        else:
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
        """
        Return the number of component data objects stored by this
        component.
        """
        return len(self._vars)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary"""
        return idx in self._vars

    # The default implementation is for keys() and __iter__ to be
    # synonyms.  The logic is implemented in keys() so that
    # keys/values/items continue to work for components that implement
    # other definitions for __iter__ (e.g., Set)
    def __iter__(self):
        """Return an iterator of the component data keys"""
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

    def to_jumpexpr(self):
        return {k: jump.AffExpr(0, jump.OrderedDict([(v, 1)])) for k, v in self.items()}


"""
Future formats to implement.
"""


class OmltIndexedSmoke(OmltIndexed):
    format = "smoke"

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Storing variables in Smoke format is not currently implemented."
        )


class OmltIndexedGurobi(OmltIndexed):
    format = "gurobi"

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "Storing variables in Gurobi format is not currently implemented."
        )
