<<<<<<< HEAD
from omlt.neuralnet.activations.linear import (
    linear_activation_constraint,
    linear_activation_function,
)
from omlt.neuralnet.activations.relu import (
    ComplementarityReLUActivation,
    bigm_relu_activation_constraint,
)
from omlt.neuralnet.activations.smooth import (
=======
r"""
Since all activation functions are element-wised, we only consider how to formulate activation functions for a single neuron, where :math:`x` denotes pre-activation variable, and :math:`y` denotes post-activation variable.

"""
from .linear import linear_activation_constraint, linear_activation_function
from .relu import ComplementarityReLUActivation, bigm_relu_activation_constraint
from .smooth import (
>>>>>>> a3d128d569b9b7cbf13315ea6be813f076d56b22
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
