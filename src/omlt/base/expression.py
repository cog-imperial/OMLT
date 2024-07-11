from abc import ABC, abstractmethod
from typing import Any

import pyomo.environ as pyo

from omlt.base import DEFAULT_MODELING_LANGUAGE


class OmltExpr(ABC):
    def __new__(cls, *indexes, **kwargs: Any):
        if not indexes:
            instance = super().__new__(OmltExprScalar)
            instance.__init__(**kwargs)
        else:
            instance = super().__new__(OmltExprIndexed)
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
        instance = super(OmltExpr, cls).__new__(subclass)
        instance._format = lang
        return instance

    def is_potentially_variable(self):
        pass

    def __mul__(self, other):
        pass


class OmltExprIndexed(OmltExpr):
    def __new__(cls, *indexes, lang=DEFAULT_MODELING_LANGUAGE, **kwargs: Any):
        subclass_map = {subclass.format: subclass for subclass in cls.__subclasses__()}
        if lang not in subclass_map:
            msg = (
                "Variable format %s not recognized. Supported formats are 'pyomo'"
                " or 'jump'.",
                lang,
            )
            raise ValueError(msg)
        subclass = subclass_map[lang]
        instance = super(OmltExpr, subclass).__new__(subclass)
        instance.__init__(*indexes, **kwargs)
        instance._format = lang
        return instance
