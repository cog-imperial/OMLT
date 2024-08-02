"""Pyomo-backed objects.

This file contains implementations of the OMLT classes, using
Pyomo objects as the underlying data storage mechanism.
"""

from typing import Any

import pyomo.environ as pyo
from pyomo.core.base.var import _GeneralVarData

from omlt.base.constraint import OmltConstraintIndexed, OmltConstraintScalar
from omlt.base.expression import OmltExpr, OmltExprFactory
from omlt.base.var import OmltIndexed, OmltScalar

# Variables


class OmltScalarPyomo(OmltScalar, pyo.ScalarVar):
    format = "pyomo"

    def __init__(self, *args: Any, **kwargs: Any):
        OmltScalar.__init__(self)
        kwargs.pop("lang", None)
        self._format = "pyomo"
        self._pyovar = pyo.ScalarVar(*args, **kwargs)
        self._name = None
        self._parent = None
        self._constructed = self._pyovar._constructed

    def construct(self, data=None):
        return self._pyovar.construct(data)

    def is_constructed(self):
        return self._pyovar.is_constructed()

    def fix(self, value, *, skip_validation=False):
        self._pyovar.fix(value, skip_validation)

    @property
    def ctype(self):
        return pyo.ScalarVar

    @property
    def name(self):
        self._pyovar._name = self._name
        return self._pyovar._name

    @property
    def bounds(self):
        return (self._pyovar._lb, self._pyovar._ub)

    @bounds.setter
    def bounds(self, val):
        self._pyovar.lb = val[0]
        self._pyovar.ub = val[1]

    @property
    def lb(self):
        return self._pyovar._lb

    @lb.setter
    def lb(self, val):
        self._pyovar.setlb(val)

    @property
    def ub(self):
        return self._pyovar._ub

    @ub.setter
    def ub(self, val):
        self._pyovar.setub(val)

    @property
    def domain(self):
        return self._pyovar._domain

    @domain.setter
    def domain(self, val):
        self._pyovar._domain = val

    # Interface for getting/setting value
    @property
    def value(self):
        return self._pyovar.value

    @value.setter
    def value(self, val):
        self._pyovar.value = val


class OmltIndexedPyomo(pyo.Var, OmltIndexed):
    format = "pyomo"

    def __init__(self, *indexes: Any, **kwargs: Any):
        kwargs.pop("lang", None)
        self._format = "pyomo"
        super().__init__(*indexes, **kwargs)
        self.bounds = (None, None)

    def fix(self, value=None, *, skip_validation=False):
        self.fixed = True
        if value is None:
            for vardata in self.values():
                vardata.fix(skip_validation)
        else:
            for vardata in self.values():
                vardata.fix(value, skip_validation)

    def setub(self, value):
        self.bounds = (self.bounds[0], value)
        for vardata in self.values():
            vardata.ub = value

    def setlb(self, value):
        self.bounds = (value, self.bounds[1])
        for vardata in self.values():
            vardata.lb = value


# Constraints


class OmltConstraintScalarPyomo(OmltConstraintScalar, pyo.Constraint):
    format = "pyomo"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lhs: pyo.Expression = (
            self.lhs._expression
            if isinstance(self.lhs, OmltExprScalarPyomo)
            else self.lhs
        )
        self.rhs: pyo.Expression = (
            self.rhs._expression
            if isinstance(self.rhs, OmltExprScalarPyomo)
            else self.rhs
        )

        if self.sense == "==":
            pyoexpr = self.lhs == self.rhs
        if self.sense == ">=":
            pyoexpr = self.lhs >= self.rhs
        if self.sense == "<=":
            pyoexpr = self.lhs <= self.rhs

        self.constraint = pyo.Constraint(expr=pyoexpr)
        self.constraint._parent = self._parent
        self.constraint.construct()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.constraint.__call__(*args, **kwds)

    @property
    def __class__(self):
        return type(self.constraint.expr)

    @property
    def args(self):
        return self.constraint.expr.args

    @property
    def strict(self):
        return self.constraint.expr._strict

    @property
    def _constructed(self):
        return self.constraint._constructed

    @property
    def _active(self):
        return self.constraint._active

    @property
    def _data(self):
        return self.constraint._data

    def is_indexed(self):
        return False


class OmltConstraintIndexedPyomo(OmltConstraintIndexed, pyo.Constraint):
    format = "pyomo"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        kwargs.pop("model", None)
        kwargs.pop("lang", None)
        kwargs.pop("expr_tuple", None)
        self.constraint = pyo.Constraint(*args, **kwargs)
        self._index_set = self.constraint._index_set

        self.constraint._parent = self._parent
        self.constraint.construct()
        self.model = self.constraint.model

        self.constraints: dict[Any, Any] = {}

    def __setitem__(self, index, expr):
        if index in self._index_set:
            self.constraint[index] = expr
            self.constraints[index] = self.constraint[index]
        else:
            msg = (
                "Couldn't find index %s in index set %s.",
                index,
                list(self._index_set.data()),
            )
            raise KeyError(msg)

    def __getitem__(self, index):
        if index in self.constraint._index_set:
            return self.constraint[index]
        msg = (
            "Couldn't find index %s in index set %s.",
            index,
            list(self._index_set.data()),
        )
        raise KeyError(msg)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.constraint.__call__(*args, **kwds)

    def __len__(self):
        return len(self.constraint)

    @property
    def _constructed(self):
        return self.constraint._constructed

    @property
    def _active(self):
        return self.constraint._active

    @_active.setter
    def _active(self, val):
        self.constraint._active = val

    @property
    def _data(self):
        return self.constraint._data

    @property
    def doc(self):
        return self.constraint.doc


# Expressions


class OmltExprScalarPyomo(OmltExpr, pyo.Expression):
    format = "pyomo"

    def __init__(self, expr=None, **kwargs):
        self._index_set = {}
        if isinstance(expr, OmltExprScalarPyomo):
            self._expression = expr._expression
        elif isinstance(expr, (pyo.Expression, pyo.NumericValue)):
            self._expression = expr
        elif isinstance(expr, tuple):
            self._expression = self._parse_expression_tuple(expr)
        else:
            msg = ("Expression %s type %s not recognized", expr, type(expr))
            raise TypeError(msg)

        self._parent = None
        self.name = None
        self.__class__ = type(self._expression)
        self._args_ = self._expression._args_
        self._format = "pyomo"
        self.expr_factory = OmltExprFactory()

    def _parse_expression_tuple_term(self, term):
        if isinstance(term, tuple):
            return self._parse_expression_tuple(term)
        if isinstance(term, OmltExprScalarPyomo):
            return term._expression
        if isinstance(term, OmltScalarPyomo):
            return term._pyovar
        if isinstance(term, (pyo.Expression, pyo.Var, _GeneralVarData, int, float)):
            return term
        msg = ("Term of expression %s is an unsupported type. %s", term, type(term))
        raise TypeError(msg)

    def _parse_expression_tuple(self, expr):
        lhs = self._parse_expression_tuple_term(expr[0])
        rhs = self._parse_expression_tuple_term(expr[2])

        if expr[1] == "+":
            return lhs + rhs

        if expr[1] == "-":
            return lhs - rhs

        if expr[1] == "*":
            return lhs * rhs

        if expr[1] == "/":
            return lhs / rhs

        msg = ("Expression middle term was {%s}.", expr[1])
        raise ValueError(msg)

    def __class__(self):
        return type(self._expression)

    def is_potentially_variable(self):
        return self._expression.is_potentially_variable()

    def as_numeric(self):
        return self._expression._apply_operation(self._expression.args)

    def construct(self, data=None):
        return self._expression.construct(data)

    @property
    def _constructed(self):
        return self._expression.expr._constructed

    @property
    def const(self):
        return self._expression.const

    @property
    def args(self):
        return self._expression.args

    def arg(self, index):
        return self._expression.arg(index)

    def nargs(self):
        return self._expression.nargs()

    def __len__(self):
        return 1

    def __call__(self):
        return self._expression()

    def is_indexed(self):
        return False

    def __add__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = self._expression + other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression + other
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __sub__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = self._expression - other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression - other
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __mul__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = self._expression * other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression * other
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __div__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = self._expression / other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression / other
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __radd__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = other._expression + self._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = other + self._expression
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __rsub__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = other._expression - self._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = other - self._expression
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __rmul__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            expr = other._expression * self._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = other * self._expression
        return self.expr_factory.new_expression(lang=self._format, expr=expr)

    def __ge__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            rhs = other._expression
        elif isinstance(other, OmltScalarPyomo):
            rhs = other._pyovar
        else:
            rhs = other
        return OmltConstraintScalarPyomo(
            model=self._parent, lang=self._format, lhs=self, sense=">=", rhs=rhs
        )

    def __le__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            rhs = other._expression
        elif isinstance(other, OmltScalarPyomo):
            rhs = other._pyovar
        else:
            rhs = other
        return OmltConstraintScalarPyomo(
            model=self._parent, lang=self._format, lhs=self, sense="<=", rhs=rhs
        )

    def __eq__(self, other):
        if isinstance(other, OmltExprScalarPyomo):
            rhs = other._expression
        elif isinstance(other, OmltScalarPyomo):
            rhs = other._pyovar
        else:
            rhs = other
        return OmltConstraintScalarPyomo(
            model=self._parent, lang=self._format, lhs=self, sense="==", rhs=rhs
        )
