"""Abstraction layer of classes used by OMLT.

Underneath these are
objects in a choice of modeling languages: Pyomo (default),
JuMP, or others (not yet implemented - e.g. Smoke, Gurobi).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omlt.base import DEFAULT_MODELING_LANGUAGE, expression


class OmltVar(ABC):
    @abstractmethod
    def construct(self, data):
        """Construct the variable."""

    @abstractmethod
    def fix(self, value, *, skip_validation=False):
        """Fix the value of the variable."""

    @property
    @abstractmethod
    def ctype(self):
        """Return the type of the variable."""

    @property
    @abstractmethod
    def name(self):
        """Return the name of the variable."""

    # Some methods to tell OMLT (and Pyomo components) that this
    # is a variable.
    def is_component_type(self):
        return True

    @abstractmethod
    def is_indexed(self):
        """Return False for a scalar variable, True for an indexed variable."""

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True


class OmltScalar(OmltVar):
    format: str | None = None

    def __init__(self):
        self.expr_factory = expression.OmltExprFactory()

    def is_indexed(self):
        return False

    # Bound-setting interface for scalar variables:
    @property
    @abstractmethod
    def bounds(self):
        """Return a tuple with the lower and upper bounds."""

    @bounds.setter
    @abstractmethod
    def bounds(self, val):
        """Set lower and upper bounds to the given tuple."""

    @property
    @abstractmethod
    def lb(self):
        """Return the lower bound of the variable."""

    @lb.setter
    @abstractmethod
    def lb(self, val):
        """Set lower bound to the given value."""

    @property
    @abstractmethod
    def ub(self):
        """Return the upper bound of the variable."""

    @ub.setter
    @abstractmethod
    def ub(self, val):
        """Set upper bound to the given value."""

    @property
    @abstractmethod
    def domain(self):
        """Return the set of allowable values."""

    @domain.setter
    @abstractmethod
    def domain(self, val):
        """Set the allowable values to the given set."""

    # Interface for getting/setting value
    @property
    @abstractmethod
    def value(self):
        """Return the current value of the variable."""

    @value.setter
    @abstractmethod
    def value(self, val):
        """Set the current value of the variable."""

    # Interface governing how variables behave in expressions.

    def __add__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(self, "+", other)
        )

    def __sub__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(self, "-", other)
        )

    def __mul__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(self, "*", other)
        )

    def __truediv__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(self, "/", other)
        )

    def __radd__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(other, "+", self)
        )

    def __rsub__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(other, "-", self)
        )

    def __rmul__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(other, "*", self)
        )

    def __rtruediv__(self, other):
        return self.expr_factory.new_expression(
            lang=self.format, expr=(other, "/", self)
        )


class OmltIndexed(OmltVar):
    format: str | None = None

    def is_indexed(self):
        return True

    @property
    @abstractmethod
    def index_set(self):
        """Return the index set for the variable."""

    # Bound-setting interface for indexed variables:
    @abstractmethod
    def setub(self, value):
        """Set upper bounds on all component variables."""

    @abstractmethod
    def setlb(self, value):
        """Set lower bounds on all component variables."""

    # Interface: act as a dict for the sub-variables.
    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def keys(self):
        """Return iterator over the index set."""

    @abstractmethod
    def values(self, sort):
        """Return iterator over the component variables."""

    @abstractmethod
    def items(self):
        """Return iterator over the key-value pairs."""

    @abstractmethod
    def __len__(self):
        """Return size of the index set."""

    @abstractmethod
    def __contains__(self, idx):
        """Return true if idx is in the index set."""


class OmltVarFactory:
    def __init__(self):
        self.scalars = {
            subclass.format: subclass for subclass in OmltScalar.__subclasses__()
        }
        self.indexed = {
            subclass.format: subclass for subclass in OmltIndexed.__subclasses__()
        }

    def register(self, lang, indexed, varclass):
        if lang is None:
            lang = varclass.format
        if indexed:
            if lang in self.indexed:
                msg = ("Indexed variable format %s is already registered.", lang)
                raise KeyError(msg)
            self.indexed[lang] = varclass
        else:
            if lang in self.scalars:
                msg = ("Scalar variable format %s is already registered.", lang)
                raise KeyError(msg)
            self.scalars[lang] = varclass

    def new_var(
        self, *indexes: Any, lang: str = DEFAULT_MODELING_LANGUAGE, **kwargs: Any
    ) -> Any:
        if indexes:
            if lang not in self.indexed:
                msg = (
                    "Variable format %s not recognized. Supported formats are %s",
                    lang,
                    list(self.indexed.keys()),
                )
                raise KeyError(msg)
            return self.indexed[lang](*indexes, **kwargs)
        if lang not in self.scalars:
            msg = (
                "Variable format %s not recognized. Supported formats are %s",
                lang,
                list(self.scalars.keys()),
            )
            raise KeyError(msg)

        return self.scalars[lang](**kwargs)
