from .linear import linear_activation_constraint, linear_activation_function
from .relu import ComplementarityReLUActivation, bigm_relu_activation_constraint
from .smooth import (
    sigmoid_activation_constraint,
    sigmoid_activation_function,
    softplus_activation_constraint,
    softplus_activation_function,
    tanh_activation_constraint,
    tanh_activation_function,
)

ACTIVATION_FUNCTION_MAP = {
    "linear": linear_activation_function,
    #    "relu": relu_max_activation
    "sigmoid": sigmoid_activation_function,
    "softplus": softplus_activation_function,
    "tanh": tanh_activation_function,
}

NON_INCREASING_ACTIVATIONS = []
