from abc import ABC, abstractmethod
import pyomo.environ as pyo

# from pyomo.core.expr import RelationalExpression

from omlt.base import DEFAULT_MODELING_LANGUAGE
import omlt.base.var as var

# from omlt.dependencies import julia_available

# if julia_available:
#     from omlt.base.julia import jl, jump, JumpVar
#     from juliacall import AnyValue
# relations = {"==", ">=", "<=", ">", "<"}


class OmltExpr(ABC):
    # Claim to be a Pyomo Expression so blocks will register
    # properly.
    @property
    def __class__(self):
        return pyo.Expression

    def __new__(cls, *indexes, **kwargs):
        if not indexes:
            instance = super(OmltExpr, cls).__new__(OmltExprScalar)
            instance.__init__(**kwargs)
        else:
            instance = super(OmltExpr, cls).__new__(OmltExprIndexed)
            instance.__init__(*indexes, **kwargs)
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
        pass

    def valid_model_component(self):
        """Return True if this can be used as a model component."""
        return True

    @property
    @abstractmethod
    def args(self):
        pass

    @abstractmethod
    def arg(self, index):
        pass

    @abstractmethod
    def nargs(self):
        pass


class OmltExprScalar(OmltExpr):
    def __new__(cls, *args, format=DEFAULT_MODELING_LANGUAGE, **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                "Expression format %s not recognized. Supported formats "
                "are 'pyomo' or 'jump'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltExpr, cls).__new__(subclass)
        # instance.__init__(*args, **kwargs)
        instance._format = format
        return instance

    def __mul__(self, other):
        pass


class OmltExprScalarPyomo(OmltExprScalar, pyo.Expression):
    format = "pyomo"

    def __init__(self, *args, expr=None, **kwargs):
        self._index_set = {}
        if isinstance(expr, (pyo.Expression, pyo.NumericValue)):
            self._expression = expr
        elif isinstance(expr, OmltExprScalarPyomo):
            self._expression = expr._expression
        elif isinstance(expr, tuple):
            self._expression = self._parse_expression_tuple(expr)
        else:
            print("expression not recognized", expr, type(expr))

        self._parent = None
        self.name = None

    def _parse_expression_tuple_term(self, term):
        if isinstance(term, tuple):
            return self._parse_expression_tuple(term)
        elif isinstance(term, OmltExprScalarPyomo):
            return term._expression
        elif isinstance(term, var.OmltVar):
            return term._pyovar
        elif isinstance(term, (pyo.Expression, pyo.Var, int, float)):
            return term
        else:
            raise TypeError(
                "Term of expression is an unsupported type. "
                "Write a better error message."
            )

    def _parse_expression_tuple(self, expr):
        lhs = self._parse_expression_tuple_term(expr[0])
        rhs = self._parse_expression_tuple_term(expr[2])

        if expr[1] == "+":
            return lhs + rhs

        elif expr[1] == "-":
            return lhs - rhs

        elif expr[1] == "*":
            return lhs * rhs

        elif expr[1] == "/":
            return lhs / rhs

        else:
            raise ValueError("Expression middle term was {%s}.", expr[1])

    def __repr__(self):
        return repr(self._expression.arg(0))

    def is_indexed(self):
        return False

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

    def __call__(self):
        return self._expression()

    def __add__(self, other):
        if isinstance(other, OmltExpr):
            expr = self._expression + other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression + other
        return OmltExpr(format=self._format, expr=expr)

    # def __sub__(self, other):
    #     expr = (self, "-", other)
    #     return OmltExpression(format=self._format, expr=expr)

    def __mul__(self, other):
        if isinstance(other, OmltExpr):
            expr = self._expression * other._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = self._expression * other
        return OmltExprScalar(format=self._format, expr=expr)

    def __div__(self, other):
        expr = (self, "/", other)
        return OmltExpr(format=self._format, expr=expr)

    def __truediv__(self, other):
        expr = (self, "//", other)
        return OmltExpr(format=self._format, expr=expr)

    def __radd__(self, other):
        if isinstance(other, OmltExpr):
            expr = other._expression + self._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = other + self._expression
        return OmltExpr(format=self._format, expr=expr)

    def __rsub__(self, other):
        if isinstance(other, OmltExpr):
            expr = other._expression - self._expression
        elif isinstance(other, (int, float, pyo.Expression)):
            expr = other - self._expression
        return OmltExpr(format=self._format, expr=expr)

    def __rmul__(self, other):
        expr = (other, "*", self)
        return OmltExpr(format=self._format, expr=expr)

    def __ge__(self, other):
        expr = self._expression >= other
        return expr
        # return constraint.OmltRelScalar(format=self._format, expr_tuple=expr)

    def __le__(self, other):
        expr = self._expression <= other
        return expr
        # return constraint.OmltRelScalar(format=self._format, expr_tuple=expr)

    def __eq__(self, other):
        expr = self._expression == other
        return pyo.Expression(expr=expr)
        # return constraint.OmltRelScalar(format=self._format, expr_tuple=expr)


class OmltExprIndexed(OmltExpr):
    def __new__(cls, *indexes, format=DEFAULT_MODELING_LANGUAGE, **kwargs):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if format not in subclass_map:
            raise ValueError(
                "Variable format %s not recognized. Supported formats are 'pyomo'"
                " or 'jump'.",
                format,
            )
        subclass = subclass_map[format]
        instance = super(OmltExpr, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        instance._format = format
        return instance


class OmltExprIndexedPyomo(OmltExprIndexed, pyo.Expression):
    format = "pyomo"

    def __init__(self, *indexes, expr=None, format=DEFAULT_MODELING_LANGUAGE, **kwargs):
        if len(indexes) == 1:
            index_set = indexes[0]
            i_dict = {}
            for i, val in enumerate(index_set):
                i_dict[i] = val
            self._index_set = tuple(i_dict[i] for i in range(len(index_set)))
        elif len(indexes) > 1:
            raise ValueError("Currently index cross-products are unsupported.")
        else:
            self._index_set = {}
        self._format = format
        self._expression = pyo.Expression(self._index_set, expr=expr)

        # self.pyo.construct()

    def is_indexed(self):
        return True

    def expression_as_dict(self):
        if len(self._index_set) == 1:
            return {self._index_set[0]: self._expression}
        else:
            return {k: self._expression[k] for k in self._index_set}

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 1:
            return self._expression[item[0]]
        else:
            return self._expression[item]

    def __setitem__(self, item, value):
        self._expression[item] = value

    def keys(self):
        return self._expression.keys()

    def values(self):
        return self._expression.values()

    def items(self):
        return self._expression.items()

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.
        """
        return len(self._expression)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary"""
        return idx in self._expression

    # The default implementation is for keys() and __iter__ to be
    # synonyms.  The logic is implemented in keys() so that
    # keys/values/items continue to work for components that implement
    # other definitions for __iter__ (e.g., Set)
    def __iter__(self):
        """Return an iterator of the component data keys"""
        return self._expression.__iter__()

    @property
    def args(self):
        return self._expression.args()

    def arg(self, index):
        return self._expression.arg(index)

    def nargs(self):
        return self._expression.nargs()

    def __call__(self):
        return self._expression()

    # # def __str__(self):
    # #     return parse_expression(self.expr, "").rstrip()

    # def __repr__(self):
    #     if self._expr is not None:
    #         return parse_expression(self._expr, "").rstrip()
    #     else:
    #         return "empty expression"

    # def set_value(self, value):
    #     print("setting value:", value)
    #     self.value = value

    # @property
    # def rule(self):
    #     return self._expr

    def __add__(self, other):
        expr = (self, "+", other)
        return OmltExpr(self._index_set, format=self._format, expr=expr)

    # def __sub__(self, other):
    #     expr = (self, "-", other)
    #     return OmltExpression(format=self._format, expr=expr)

    # def __mul__(self, other):
    #     expr = (self, "*", other)
    #     return OmltExpression(format=self._format, expr=expr)

    def __div__(self, other):
        expr = (self, "/", other)
        return OmltExpr(self._index_set, format=self._format, expr=expr)

    def __truediv__(self, other):
        expr = (self, "//", other)
        return OmltExpr(self._index_set, format=self._format, expr=expr)

    def __eq__(self, other):
        expr = (self, "==", other)
        return pyo.Expression(self._index_set, expr=expr)
        # return constraint.OmltRelation(
        #     self._index_set, format=self._format, expr_tuple=expr
        # )

    def __le__(self, other):
        expr = (self, "<=", other)
        return pyo.Expression(self._index_set, expr=expr)
        # return constraint.OmltRelation(
        #     self._index_set, format=self._format, expr_tuple=expr
        # )

    def __ge__(self, other):
        expr = (self, ">=", other)
        return pyo.Expression(self._index_set, expr=expr)
        # return constraint.OmltRelation(
        #     self._index_set, format=self._format, expr_tuple=expr
        # )


# def parse_expression(expr, string):
#     if expr is not None:
#         for t in expr:
#             if str(t).count(" ") == 2:
#                 string += "(" + str(t) + ") "
#             else:
#                 string += str(t) + " "
#     else:
#         string = expr
#     return string


# def parse_jump_affine(expr_tuple):
#     if expr_tuple is not None:
#         if isinstance(expr_tuple, JumpVar):
#             return jump.AffExpr(0, {expr_tuple.to_jump(): 1})
#         elif isinstance(expr_tuple, (int, float)):
#             return jump.AffExpr(expr_tuple, {})
#         elif isinstance(expr_tuple, OmltExprScalar):
#             print("found a scalar expression")
#             print(expr_tuple)
#             print(expr_tuple._expression)
#             return expr_tuple._expression
#         elif len(expr_tuple) == 1 and isinstance(expr_tuple[0], JumpVar):
#             return jump.AffExpr(0, {expr_tuple[0].to_jump(): 1})
#         elif len(expr_tuple) == 1 and isinstance(expr_tuple[0], (int, float)):
#             return jump.AffExpr(expr_tuple[0], {})
#         elif len(expr_tuple) == 2:
#             print("don't know how to deal with 2-element expressions")
#             print("expr_tuple")
#         elif len(expr_tuple) == 3:
#             print("triplet")
#             if expr_tuple[1] == "+":
#                 return parse_jump_affine(expr_tuple[0]) + parse_jump_affine(
#                     expr_tuple[2]
#                 )
#             elif expr_tuple[1] == "-":
#                 return parse_jump_affine(expr_tuple[0]) - parse_jump_affine(
#                     expr_tuple[2]
#                 )
#             elif expr_tuple[1] == "*":
#                 return parse_jump_affine(expr_tuple[0]) * parse_jump_affine(
#                     expr_tuple[2]
#                 )
#             elif expr_tuple[1] == "/":
#                 return parse_jump_affine(expr_tuple[0]) / parse_jump_affine(
#                     expr_tuple[2]
#                 )
#             elif expr_tuple[1] == "//":
#                 return parse_jump_affine(expr_tuple[0]) // parse_jump_affine(
#                     expr_tuple[2]
#                 )
#             elif expr_tuple[1] == "**":
#                 return parse_jump_affine(expr_tuple[0]) ** parse_jump_affine(
#                     expr_tuple[2]
#                 )


# def dictplus(a, b):
#     c = dict()
#     if a.keys() == b.keys():
#         for k in a.keys():
#             c[k] = a[k] + b[k]
#         return c
#     else:
#         raise ValueError("dicts have non-matching keys")


# def dictminus(a, b):
#     c = dict()
#     if a.keys() == b.keys():
#         for k in a.keys():
#             c[k] = a[k] - b[k]
#         print("dictminus gives:", c)
#         return c
#     else:
#         raise ValueError("dicts have non-matching keys")


# def dicttimes(a, b):
#     c = dict()
#     if a.keys() == b.keys():
#         for k in a.keys():

#             c[k] = a[k] * b[k]
#         return c
#     else:
#         raise ValueError("dicts have non-matching keys")


# def dictover(a, b):
#     c = dict()
#     if a.keys() == b.keys():
#         for k in a.keys():

#             c[k] = jump_divide(a[k], b[k])
#         return c
#     else:
#         raise ValueError("dicts have non-matching keys")


# def jump_divide(a, b):
#     assert isinstance(a, AnyValue)
#     print(b.terms)
#     assert (isinstance(b, AnyValue) and len(b.terms) == 0) or isinstance(
#         b, (int, float)
#     )
#     if isinstance(b, AnyValue):
#         div_by = b.constant
#     else:
#         div_by = b
#     return jump.AffExpr(a.constant / div_by, {})


# def parse_jump_indexed(expr_tuple, index):
#     print("parsing:", expr_tuple)
#     if expr_tuple is not None:
#         if isinstance(expr_tuple, OmltExpr):
#             print("here")
#             return expr_tuple.expression_as_dict()
#         elif isinstance(expr_tuple, var.OmltVar):
#             return expr_tuple.to_jumpexpr()
#         elif isinstance(expr_tuple, (int, float)):
#             return {k: jump.AffExpr(expr_tuple, {}) for k in index}
#         elif len(expr_tuple) == 1 and isinstance(expr_tuple[0], OmltExpr):
#             return expr_tuple[0]._expression
#         elif len(expr_tuple) == 1 and isinstance(expr_tuple[0], var.OmltVar):
#             indexed = {
#                 k: jump.AffExpr(0, jump.OrderedDict([(v, 1)]))
#                 for k, v in expr_tuple[0].items()
#             }
#             return indexed
#         elif len(expr_tuple) == 1 and isinstance(expr_tuple[0], (int, float)):
#             return {k: jump.AffExpr(expr_tuple[0], {}) for k in index}
#         elif len(expr_tuple) == 2:
#             print("don't know how to deal with 2-element expressions")
#             print(expr_tuple)
#         elif len(expr_tuple) == 3:
#             if expr_tuple[1] == "+":
#                 return dictplus(
#                     parse_jump_indexed(expr_tuple[0], index),
#                     parse_jump_indexed(expr_tuple[2], index),
#                 )
#             elif expr_tuple[1] == "-":
#                 return dictminus(
#                     parse_jump_indexed(expr_tuple[0], index),
#                     parse_jump_indexed(expr_tuple[2], index),
#                 )
#             elif expr_tuple[1] == "*":
#                 return dicttimes(
#                     parse_jump_indexed(expr_tuple[0], index),
#                     parse_jump_indexed(expr_tuple[2], index),
#                 )
#             elif expr_tuple[1] == "/":
#                 return dictover(
#                     parse_jump_indexed(expr_tuple[0], index),
#                     parse_jump_indexed(expr_tuple[2], index),
#                 )
#             elif expr_tuple[1] == "//":
#                 return dictover(
#                     parse_jump_indexed(expr_tuple[0], index),
#                     parse_jump_indexed(expr_tuple[2], index),
#                 )
#             elif expr_tuple[1] == "**":
#                 return parse_jump_indexed(expr_tuple[0], index) ** parse_jump_indexed(
#                     expr_tuple[2], index
#                 )
#             elif expr_tuple[1] in relations:
#                 cnstrnt = constraint.OmltRelation(
#                     index,
#                     model=None,
#                     lhs=parse_jump_indexed(expr_tuple[0], index),
#                     sense=expr_tuple[1],
#                     rhs=parse_jump_indexed(expr_tuple[2], index),
#                     format="jump",
#                 )
#                 indexed = {k: cnstrnt.lhs[k] - cnstrnt.rhs[k] for k in index}
#                 return indexed
