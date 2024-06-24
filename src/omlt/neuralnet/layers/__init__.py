r"""Neural network layers.

Since OMLT builds layer and activation functions in layer level, we ignore the layer
index and use the following notations to describe the :math:`l`-th layer:

.. math::

    \begin{align*}
        F_{in}  &:= \text{number of input features}\\
        F_{out} &:= \text{number of output features}\\
        x_i     &:= \text{the $i$-th input, $0\le i<F_{in}$}\\
        y_j     &:= \text{the $j$-th output, $0\le j<F_{out}$}\\
        w_{ij}  &:= \text{weight from $x_i$ to $y_j$, $0\le i<F_{in}, 0\le j<F_{out}$}\\
        b_j     &:= \text{bias for $y_j$, $0\le j<F_{out}$}
    \end{align*}

"""

from .full_space import full_space_conv2d_layer, full_space_dense_layer
from .reduced_space import reduced_space_dense_layer

__all__ = [
    "full_space_conv2d_layer",
    "full_space_dense_layer",
    "reduced_space_dense_layer",
]
