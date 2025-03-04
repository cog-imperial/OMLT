from __future__ import annotations

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


class OmltConstraintIndexed(OmltConstraint):
    def __init__(
        self, *indexes: Any, lang: str = DEFAULT_MODELING_LANGUAGE, **kwargs: Any
    ):
        self._index_set = indexes

        self.model = kwargs.pop("model", None)
        self._parent = None
        self.name = None
        self.format = lang

    def __getitem__(self, item):
        """Return the scalar constraint corresponding to the given index."""

    def __setitem__(self, item, value):
        """Add the scalar constraint to the dict, at the given index."""

    def keys(self, sort=False):  # noqa: ARG002, FBT002
        yield from self._index_set


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
    ) -> OmltConstraint:
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
