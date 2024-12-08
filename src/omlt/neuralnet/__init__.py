r"""omlt.neuralnet.

The basic pipeline in source code of OMLT is:

.. math::

    \begin{align*}
        \mathbf z^{(0)}
        \xrightarrow[\text{Constraints}]{\text{Layer 1}} \hat{\mathbf z}^{(1)}
        \xrightarrow[\text{Activations}]{\text{Layer 1}} \mathbf z^{(1)}
        \xrightarrow[\text{Constraints}]{\text{Layer 2}} \hat{\mathbf z}^{(2)}
        \xrightarrow[\text{Activations}]{\text{Layer 2}} \mathbf z^{(2)}
        \xrightarrow[\text{Constraints}]{\text{Layer 3}}\cdots
    \end{align*}

where
:math:`\mathbf z^{(0)}` is the output of `InputLayer`,
:math:`\hat{\mathbf z}^{(l)}` is the pre-activation output of :math:`l`-th layer,
:math:`\mathbf z^{(l)}` is the post-activation output of :math:`l`-th layer.

"""

from omlt.neuralnet.network_definition import NetworkDefinition
from omlt.neuralnet.nn_formulation import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    ReducedSpaceNNFormulation,
    ReducedSpaceSmoothNNFormulation,
    ReluBigMFormulation,
    ReluComplementarityFormulation,
    ReluPartitionFormulation,
)

__all__ = [
    "FullSpaceNNFormulation",
    "FullSpaceSmoothNNFormulation",
    "NetworkDefinition",
    "ReducedSpaceNNFormulation",
    "ReducedSpaceSmoothNNFormulation",
    "ReluBigMFormulation",
    "ReluComplementarityFormulation",
    "ReluPartitionFormulation",
]
