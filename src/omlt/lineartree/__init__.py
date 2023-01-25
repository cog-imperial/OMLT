r"""
We use the following notation to describe the gradient-boosted trees formulation:

.. math::

    \begin{align*}
    \hat{\mu} &:= \text{Mean prediction of tree ensemble}\\
    T &:= \text{Set of trees in ensemble}\\
    L_t &:= \text{Set of leaves in tree $t$}\\
    z_{t,l} &:= \text{Binary variable indicating if leaf $l$
        in tree $t$ is active}\\
    \text{Left}_{t,s} &:= \text{Set of leaf variables left of split $s$
        in tree $t$}\\
    \text{Right}_{t,s} &:= \text{Set of leaf variables right of split $s$
        in tree $t$}\\
    y_{i(s),j(s)} &:= \text{Binary variable indicating if split $s$ is active}\\
    i(s) &:= \text{feature of split $s$}\\
    j(s) &:= \text{index of split $s$}\\
    V_t &:= \text{Set of splits in tree $t$}\\
    n &:= \text{Index set of input features}\\
    m_i &:= \text{Index set of splits for feature $i$}\\
    F_{t,l} &:= \text{Weight of leaf $l$ in tree $t$}\\
    \end{align*}
"""
from omlt.lineartree.lt_formulation import GBTBigMFormulation
from omlt.lineartree.lt_model import LinearTreeModel
