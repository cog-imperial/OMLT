"""
Abstraction layer of classes used by OMLT. Underneath these are
objects in a choice of modeling languages: Pyomo (default),
MathOptInterface, or Smoke (not yet implemented).


"""

from abc import ABC, abstractmethod
import pyomo.environ as pyo

from omlt.dependencies import julia_available, moi_available

if julia_available and moi_available:
    from juliacall import Main as jl
    from juliacall import Base

    jl.seval("import MathOptInterface")
    moi = jl.MathOptInterface
    jl.seval("import JuMP")
    jump = jl.JuMP


class OmltVar(ABC):
    def __new__(cls, *indexes, **kwargs):

        if not indexes:
            instance = OmltScalar.__new__(OmltScalar, **kwargs)
        else:
            instance = OmltIndexed.__new__(OmltIndexed, *indexes, **kwargs)
        return instance


class OmltScalar(OmltVar):
    def __new__(cls, *args, format="pyomo", **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                f"Variable format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltVar, subclass).__new__(subclass)

        instance.__init__(*args, **kwargs)
        return instance

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def construct(self, data):
        pass

    @abstractmethod
    def fix(self, value, skip_validation):
        pass

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

    def is_component_type(self):
        return True

    def is_indexed(self):
        return False

    # @abstractmethod
    # def __mul__(self, other):
    #     pass

    # @abstractmethod
    # def __rmul__(self, other):
    #     pass


class OmltScalarPyomo(pyo.ScalarVar, OmltScalar):
    format = "pyomo"

    def __init__(self, *args, **kwargs):
        pyo.ScalarVar.__init__(self, *args, **kwargs)

    def construct(self, data):
        super().construct(data)

    def fix(self, value=None, skip_validation=False):
        self.fixed = True
        if value is None:
            super().fix(skip_validation)
        else:
            super().fix(value, skip_validation)

    @property
    def bounds(self):
        return super().bounds

    @bounds.setter
    def bounds(self, val):
        super().bounds = val

    @property
    def ub(self):
        return super().ub

    @ub.setter
    def ub(self, val):
        super().ub = val

    @property
    def lb(self):
        return super().__get__(self.lb)

    @lb.setter
    def lb(self, val):
        super().__setattr__(self.lb, val)

    def __lt__(self, other):
        return pyo.NumericValue.__lt__(self, other)

    def __gt__(self, other):
        return pyo.NumericValue.__gt__(self, other)

    def __le__(self, other):
        return pyo.NumericValue.__le__(self, other)

    def __ge__(self, other):
        return pyo.NumericValue.__ge__(self, other)

    def __eq__(self, other):
        return pyo.NumericValue.__eq__(self, other)

    def __add__(self, other):
        return pyo.NumericValue.__add__(self, other)

    def __sub__(self, other):
        return pyo.NumericValue.__sub__(self, other)

    # def __mul__(self,other):
    #     return pyo.NumericValue.__mul__(self,other)

    def __div__(self, other):
        return pyo.NumericValue.__div__(self, other)

    def __truediv__(self, other):
        return pyo.NumericValue.__truediv__(self, other)

    def __pow__(self, other):
        return pyo.NumericValue.__pow__(self, other)

    def __radd__(self, other):
        return pyo.NumericValue.__radd__(self, other)

    def __rsub__(self, other):
        return pyo.NumericValue.__rsub__(self, other)

    # def __rmul__(self,other):
    #     return self._ComponentDataClass.__rmul__(self,other)

    def __rdiv__(self, other):
        return pyo.NumericValue.__rdiv__(self, other)

    def __rtruediv__(self, other):
        return pyo.NumericValue.__rtruediv__(self, other)

    def __rpow__(self, other):
        return pyo.NumericValue.__rpow__(self, other)

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
            _has_lb = _lb is not None
            _ub = self._bounds[1]
            _has_ub = _ub is not None
        elif self._bounds is None:
            _has_lb = False
            _lb = None
            _has_ub = False
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
            self._value = _initialize
        else:
            self._value = None

        self._jumpvarinfo = jump.VariableInfo(
            _has_lb,
            _lb,
            _has_ub,
            _ub,
            False,  # is fixed
            None,  # fixed value
            _initialize is not None,
            self._value,
            self.binary,
            self.integer,
        )
        self._constructed = False
        self._parent = None
        self._ctype = pyo.ScalarVar

    def construct(self, data):
        if self._block:
            self._jumpvar = jump.add_variable(self._block, self._jumpvarinfo)
        else:
            self._jumpvar = jump.build_variable(Base.error, self._jumpvarinfo)
        self._constructed = True

    def fix(self, value, skip_validation):
        self.fixed = True
        self._value = value

    @property
    def bounds(self):
        pass

    @bounds.setter
    def bounds(self, val):
        pass

    @property
    def lb(self):
        return self._jumpvar.info.lower_bound

    @lb.setter
    def lb(self, val):
        jump.set_upper_bound(self._jumpvar, val)

    @property
    def ub(self):
        return self._jumpvar.info.upper_bound

    @ub.setter
    def ub(self, val):
        jump.set_upper_bound(self._jumpvar, val)

    def to_jump(self):
        if self._constructed:
            return self._jumpvar


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
    def __new__(cls, *indexes, format="pyomo", **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                f"Variable format %s not recognized. Supported formats are 'pyomo'"
                " or 'moi'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltVar, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        return instance

    @abstractmethod
    def fix(self, value=None, skip_validation=False):
        pass

    @abstractmethod
    def setub(self, value):
        pass

    @abstractmethod
    def setlb(self, value):
        pass

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True


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
        self._varinfo = {}
        for idx in self._index_set:
            self._varinfo[idx] = jump.VariableInfo(
                False,  # _has_lb,
                None,  # _lb,
                False,  # _has_ub,
                None,  # _ub,
                False,  # is fixed
                None,  # fix value
                False,  # _initialize is not None,
                None,  # self._value,
                False,  # self.binary,
                False,  # self.integer
            )
        self._vars = {}
        self._constructed = False
        self._ctype = pyo.Var
        self._parent = None

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 1:
            return self._vars[item[0]]
        else:
            return self._vars[item]

    def __setitem__(self, item, value):
        self._vars[item] = value

    def keys(self):
        return self._vars.keys()

    def values(self):
        return self._vars.values()

    def items(self):
        return self._vars.items()

    def fix(self, value=None, skip_validation=False):
        self.fixed = True
        if value is None:
            for vardata in self.values():
                vardata.fix(skip_validation)
        else:
            for vardata in self.values():
                vardata.fix(value, skip_validation)

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
            self._vars[idx] = jump.build_variable(Base.error, self._varinfo[idx])
        self._constructed = True

    def setub(self, value):
        if self._constructed:
            for idx in self.index_set():
                self._varinfo[idx].has_ub = True
                self._varinfo[idx].upper_bound = value
                self._vars[idx].info.has_ub = True
                self._vars[idx].info.upper_bound = value
        else:
            for idx in self.index_set():
                self._varinfo[idx].has_ub = True
                self._varinfo[idx].upper_bound = value

    def setlb(self, value):
        if self._constructed:
            for idx in self.index_set():
                self._varinfo[idx].has_lb = True
                self._varinfo[idx].lower_bound = value
                self._vars[idx].info.has_lb = True
                self._vars[idx].info.lower_bound = value
        else:
            for idx in self.index_set():
                self._varinfo[idx].has_lb = True
                self._varinfo[idx].lower_bound = value

    @property
    def ctype(self):
        return self._ctype

    def index_set(self):
        return self._index_set

    @property
    def name(self):
        return self._name

    def to_jump(self):
        if self._constructed:
            return jump.Containers.DenseAxisArray(
                list(self._vars.values()), self.index_set()
            )


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


class OmltSet:
    def __init__(self):
        pass


class OmltExpression:
    def __init__(self):
        pass
