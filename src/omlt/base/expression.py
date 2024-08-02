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

    @abstractmethod
    def is_indexed(self):
        """Return False for a scalar expression, True for an indexed expression."""


    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    @abstractmethod
    def __call__(self):
        """Return the current value of the expression."""

    @abstractmethod
    def is_potentially_variable(self):
        """Return True if the expression has variable arguments, False if constant."""

    @property
    @abstractmethod
    def args(self):
        """Return a list of the args of the expression."""

    @abstractmethod
    def arg(self, index):
        """Return the arg corresponding to the given index."""

    @abstractmethod
    def nargs(self):
        """Return the number of arguments."""

class OmltExprFactory:
    def __init__(self):
        self.exprs = {
            subclass.format: subclass
            for subclass in OmltExpr.__subclasses__()
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
        return self.exprs[lang](**kwargs)
