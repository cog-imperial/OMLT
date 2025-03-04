"""Abstraction layer of classes used by OMLT.

Underneath these are
objects in a choice of modeling languages: Pyomo (default),
JuMP, or others (not yet implemented - e.g. Smoke, Gurobi).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omlt.base import DEFAULT_MODELING_LANGUAGE, expression


class OmltElement(ABC):
    """Base class for OMLT elements.

    An OMLT element represents a single dimension of a variable. Scalar variables
    have a single OMLT element, indexed variables have an OMLT element for each
    member of the index set.
    """

    @abstractmethod
    def __neg__(self):
        """Return an expression representing the negation of the element."""

    @abstractmethod
    def __sub__(self, other):
        """Return an expression representing the difference."""

    @abstractmethod
    def __mul__(self, other):
        """Return an expression representing the product."""

    @abstractmethod
    def __radd__(self, other):
        """Return an expression representing the sum."""

    @abstractmethod
    def __rsub__(self, other):
        """Return an expression representing the difference."""

    @abstractmethod
    def __rmul__(self, other):
        """Return an expression representing the product."""

    @abstractmethod
    def __eq__(self, other):
        """Return an equality constraint between this element and the other."""

    @abstractmethod
    def __ge__(self, other):
        """Return an inequality constraint between this element and the other."""

    @abstractmethod
    def exp(self):
        """Return an expression representing the exponent of the element."""

    @abstractmethod
    def log(self):
        """Return an expression representing the logarithm of the element."""

    @abstractmethod
    def tanh(self):
        """Return an expression representing the hyperbolic tangent of the element."""


class OmltVar(ABC):
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

    def __init__(self, binary=False):  # noqa: ARG002, FBT002
        self.expr_factory = expression.OmltExprFactory()

    def is_indexed(self):
        return False

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

    # Interface: act as a dict for the sub-variables.
    @abstractmethod
    def __getitem__(self, item):
        """Return the element corresponding to the given index."""

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
        self,
        *indexes: Any,
        lang: str = DEFAULT_MODELING_LANGUAGE,
        binary: bool = False,
        **kwargs: Any,
    ) -> OmltVar:
        if indexes:
            if lang not in self.indexed:
                msg = (
                    "Variable format %s not recognized. Supported formats are %s",
                    lang,
                    list(self.indexed.keys()),
                )
                raise KeyError(msg)
            return self.indexed[lang](*indexes, binary=binary, **kwargs)  # type: ignore[abstract, call-arg]
        if lang not in self.scalars:
            msg = (
                "Variable format %s not recognized. Supported formats are %s",
                lang,
                list(self.scalars.keys()),
            )
            raise KeyError(msg)

        return self.scalars[lang](binary=binary, **kwargs)  # type: ignore[abstract]
