r"""
We use the following notation to describe layer and activation functions:

.. math::

    \begin{align*}
    N &:= \text{Set of nodes (i.e. neurons in the neural network)}\\
    M_i &:= \text{Number of inputs to node $i$}\\
    \hat z_i &:= \text{pre-activation value on node $i$}\\
    z_i &:= \text{post-activation value on node $i$}\\
    w_{ij} &:= \text{weight from input $j$ to node $i$}\\
    b_i &:= \text{bias value for node $i$}
    \end{align*}
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
