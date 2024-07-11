"""Abstraction layer of classes used by OMLT.

Underneath these are
objects in a choice of modeling languages: Pyomo (default),
JuMP, or others (not yet implemented - e.g. Smoke, Gurobi).
"""

from abc import ABC, abstractmethod
from typing import Any

import pyomo.environ as pyo

from omlt.base import DEFAULT_MODELING_LANGUAGE, expression


class OmltVar(ABC):
    def __new__(cls, *indexes, **kwargs: Any):
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
    def __new__(cls, *args, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Variable format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super(OmltVar, subclass).__new__(subclass)

        instance.__init__(*args, **kwargs)
        instance._format = lang
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

    def __add__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "+", other))

    def __sub__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "-", other))

    def __mul__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "*", other))

    def __div__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "//", other))

    def __truediv__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "/", other))

    def __pow__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(self, "**", other))

    def __radd__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "+", self))

    def __rsub__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "-", self))

    def __rmul__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "*", self))

    def __rdiv__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "//", self))

    def __rtruediv__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "/", self))

    def __rpow__(self, other):
        return expression.OmltExprScalar(lang=self._format, expr=(other, "**", self))

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


class OmltIndexed(OmltVar):
    def __new__(cls, *indexes, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Variable format %s not recognized. Supported formats are 'pyomo'"
                " or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super(OmltVar, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        instance._format = lang
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

    def __add__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "+", other))

    def __sub__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "-", other))

    def __mul__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "*", other))

    def __div__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "//", other))

    def __truediv__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "/", other))

    def __pow__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(self, "**", other))

    def __radd__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "+", self))

    def __rsub__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "-", self))

    def __rmul__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "*", self))

    def __rdiv__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "//", self))

    def __rtruediv__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "/", self))

    def __rpow__(self, other):
        return expression.OmltExprIndexed(self.index_set(), expr=(other, "**", self))

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
