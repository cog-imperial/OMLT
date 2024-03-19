"""
Abstraction layer of classes used by OMLT. Underneath these are
objects in a choice of modeling languages: Pyomo (default),
MathOptInterface, or Smoke (not yet implemented).


"""

from abc import ABC, abstractmethod
import pyomo.environ as pyo


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
                "are 'pyomo' or 'moi'.",
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


"""
Future formats to implement.
"""


class OmltScalarMOI(OmltScalar):
    format = "moi"


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


class OmltIndexedPyomo(pyo.Var, OmltIndexed):
    format = "pyomo"

    def __init__(self, *indexes, **kwargs):
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


"""
Future formats to implement.
"""


class OmltIndexedMOI(OmltIndexed):
    format = "moi"


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
