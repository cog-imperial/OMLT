r"""
There are multiple formulations for representing linear model decision trees.

Please see the following reference:
    * Ammari et al. (2023) Linear Model Decision Trees as Surrogates in Optimization
      of Engineering Applications. Computers & Chemical Engineering

We utilize the following common nomenclature in the formulations:

.. math::
    \begin{align*}
        L  &:= \text{Set of leaves} \\
        z_{\ell} &:= \text{Binary variable indicating which leaf is selected} \\
        x &:= \text{Vector of input variables to the decision tree}  \\
        d &:= \text{Output variable from the decision tree} \\
        a_{\ell} &:= \text{Vector of slopes learned by the tree for leaf } \ell \in L\\
        b_{\ell} &:= \text{Bias term learned by the tree for leaf } \ell \in L\\
    \end{align*}
"""

from omlt.linear_tree.lt_definition import LinearTreeDefinition
from omlt.linear_tree.lt_formulation import (
    LinearTreeGDPFormulation,
    LinearTreeHybridBigMFormulation,
)
from omlt.linear_tree.gblt_model import EnsembleDefinition
