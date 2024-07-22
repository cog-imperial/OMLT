from abc import ABC, abstractmethod
from typing import Any

import pyomo.environ as pyo

from omlt.base import DEFAULT_MODELING_LANGUAGE


class OmltExpr(ABC):
    def __new__(cls, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Expression format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super().__new__(subclass)
        instance._format = lang
        return instance

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
