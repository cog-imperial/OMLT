from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyomo.environ as pyo

from omlt.base import DEFAULT_MODELING_LANGUAGE


class OmltExpr(ABC):
    format: str | None = None

    @property
    def ctype(self):
        return pyo.Expression

    def is_component_type(self):
        return True

    def is_expression_type(self):
        return True

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    @abstractmethod
    def __call__(self):
        """Return the current value of the expression."""

    @abstractmethod
    def exp(self):
        """Return an expression representing the exponent of this expression."""

    @abstractmethod
    def log(self):
        """Return an expression representing the logarithm of this expression."""

    @abstractmethod
    def tanh(self):
        """Return an expression representing the hyperbolic tangent of expression."""

    @abstractmethod
    def __add__(self, other):
        """Return an expression representing the sum."""

    @abstractmethod
    def __sub__(self, other):
        """Return an expression representing the difference."""

    @abstractmethod
    def __mul__(self, other):
        """Return an expression representing the product."""

    @abstractmethod
    def __truediv__(self, other):
        """Return an expression representing the quotient."""

    @abstractmethod
    def __radd__(self, other):
        """Return an expression representing the sum."""

    @abstractmethod
    def __rmul__(self, other):
        """Return an expression representing the product."""

    @abstractmethod
    def __rtruediv__(self, other):
        """Return an expression representing the quotient."""

    @abstractmethod
    def __eq__(self, other):
        """Return an equality constraint between this expression and the other."""

    @abstractmethod
    def __le__(self, other):
        """Return an inequality constraint between this expression and the other."""

    @abstractmethod
    def __ge__(self, other):
        """Return an inequality constraint between this expression and the other."""


class OmltExprFactory:
    def __init__(self):
        self.exprs = {
            subclass.format: subclass for subclass in OmltExpr.__subclasses__()
        }

    def register(self, lang, varclass):
        if lang is None:
            lang = varclass.format
        if lang in self.exprs:
            msg = ("Expression format %s is already registered.", lang)
            raise KeyError(msg)
        self.exprs[lang] = varclass

    def new_expression(
        self, lang: str | None = DEFAULT_MODELING_LANGUAGE, **kwargs: Any
    ) -> Any:
        if lang not in self.exprs:
            msg = (
                "Expression format %s not recognized. Supported formats are %s",
                lang,
                list(self.exprs.keys()),
            )
            raise KeyError(msg)
        return self.exprs[lang](**kwargs)  # type: ignore[abstract]
