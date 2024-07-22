from abc import ABC, abstractmethod
from typing import Any

import pyomo.environ as pyo
from pyomo.core.expr import EqualityExpression, InequalityExpression

from omlt.base import DEFAULT_MODELING_LANGUAGE


class OmltConstraint(ABC):
    def __new__(cls, *indexes, **kwargs: Any):
        if not indexes:
            instance = OmltConstraintScalar.__new__(OmltConstraintScalar, **kwargs)
        else:
            instance = OmltConstraintIndexed.__new__(
                OmltConstraintIndexed, *indexes, **kwargs
            )
        return instance

    @property
    def ctype(self):
        return pyo.Constraint

    def is_component_type(self):
        return True

    def is_expression_type(self, enum):
        # The Pyomo ExpressionType.RELATIONAL is enum 1.
        return enum.value == 1

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class OmltConstraintScalar(OmltConstraint):
    def __new__(cls, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Constraint format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super(OmltConstraint, subclass).__new__(subclass)
        instance.__init__(**kwargs)
        instance._format = lang
        return instance

    def __init__(self, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
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
            expr_tuple = kwargs.pop("expr_tuple", None)
            if expr_tuple and expr_tuple[1] in {"==", ">=", "<=", ">", "<", "in"}:
                self.lhs = expr_tuple[0]
                self.sense = expr_tuple[1]
                self.rhs = expr_tuple[2]
        if not lhs and not sense and not rhs and not expr_tuple:
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Return the value of the body of the constraint."""

    @property
    def args(self):
        """Return an iterator over the arguments of the constraint."""


class OmltConstraintIndexed(OmltConstraint):
    def __new__(cls, *indexes, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Constraint format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super(OmltConstraint, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        instance._format = lang
        return instance

    def __init__(self, *indexes, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        self._index_set = indexes

        self.model = kwargs.pop("model", None)
        self._parent = None
        self.name = None
        self.format = lang

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

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

    @_active.setter
    @abstractmethod
    def _active(self, val):
        """Set the constraint status to active or inactive."""

    @property
    @abstractmethod
    def _data(self):
        """Return data from the constraint."""
