import pyomo.environ as pyo
from pyomo.core.base.var import IndexedVar, ScalarVar

pyomo_activations = {
    "tanh": pyo.tanh,
    "sigmoid": lambda x: 1 / (1 + pyo.exp(-x)),
    "softplus": lambda x: pyo.log(pyo.exp(x) + 1),
}


def _extract_var_data(vars):
    if isinstance(vars, ScalarVar):
        return [vars]
    elif isinstance(vars, IndexedVar):
        if vars.indexed_set().is_ordered():
            return list(vars.values())
        raise ValueError(
            "Expected IndexedVar: {} to be indexed over an ordered set.".format(vars)
        )
    elif isinstance(vars, list):
        # Todo: the above if should check if the item supports iteration rather than only list?
        varlist = list()
        for v in vars:
            if v.is_indexed():
                varlist.extend(v.values())
            else:
                varlist.append(v)
        return varlist
    else:
        raise ValueError("Unknown variable type {}".format(vars))
