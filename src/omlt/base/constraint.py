from __future__ import annotations

from abc import abstractmethod
from typing import Any

import pyomo.environ as pyo
from pyomo.core.expr import EqualityExpression, InequalityExpression

from omlt.base import DEFAULT_MODELING_LANGUAGE


class OmltConstraint:

    @property
    def ctype(self):
        return pyo.Constraint

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True



class OmltConstraintScalar(OmltConstraint):

    def __init__(self, lang: str = DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        lhs = kwargs.pop("lhs", None)
        if lhs is not None:
            self.lhs = lhs
        sense = kwargs.pop("sense", None)
        if sense is not None:
            self.sense = sense
        rhs = kwargs.pop("rhs", None)
        if rhs is not None:
            self.rhs = rhs
        if not lhs and not sense and not rhs:
            expr = kwargs.pop("expr", None)
            if isinstance(expr, EqualityExpression):
                self.lhs = expr.arg(0)
                self.sense = "=="
                self.rhs = expr.arg(1)
            if isinstance(expr, InequalityExpression):
                self.lhs = expr.arg(0)
                self.sense = "<="
                self.rhs = expr.arg(1)

        self.model = kwargs.pop("model", None)
        self.format = lang
        self._parent = None


    @property
    def args(self):
        """Return an iterator over the arguments of the constraint."""


class OmltConstraintIndexed(OmltConstraint):

    def __init__(
        self, *indexes: Any, lang: str = DEFAULT_MODELING_LANGUAGE, **kwargs: Any
    ):
        self._index_set = indexes

        self.model = kwargs.pop("model", None)
        self._parent = None
        self.name = None
        self.format = lang


    def keys(self, sort=False):
        yield from self._index_set

    @property
    @abstractmethod
    def _constructed(self):
        """Return True if the constraint has been constructed."""

    @property
    @abstractmethod
    def _active(self):
        """Return True if the constraint is active."""

    @property
    @abstractmethod
    def _data(self):
        """Return data from the constraint."""


class OmltConstraintFactory:
    def __init__(self):
        self.scalars = {
            subclass.format: subclass
            for subclass in OmltConstraintScalar.__subclasses__()
        }
        self.indexed = {
            subclass.format: subclass
            for subclass in OmltConstraintIndexed.__subclasses__()
        }

    def register(self, lang, indexed, varclass):
        if lang is None:
            lang = varclass.format
        if indexed:
            if lang in self.indexed:
                msg = ("Indexed constraint format %s is already registered.", lang)
                raise KeyError(msg)
            self.indexed[lang] = varclass
        else:
            if lang in self.scalars:
                msg = ("Scalar constraint format %s is already registered.", lang)
                raise KeyError(msg)
            self.scalars[lang] = varclass

    def new_constraint(
        self, *indexes: Any, lang: str = DEFAULT_MODELING_LANGUAGE, **kwargs: Any
    ) -> Any:
        if indexes:
            if lang not in self.indexed:
                msg = (
                    "Constraint format %s not recognized. Supported formats are %s",
                    lang,
                    list(self.indexed.keys()),
                )
                raise KeyError(msg)
            return self.indexed[lang](*indexes, **kwargs)
        if lang not in self.scalars:
            msg = (
                "Constraint format %s not recognized. Supported formats are %s",
                lang,
                list(self.scalars.keys()),
            )
            raise KeyError(msg)
        return self.scalars[lang](**kwargs)
