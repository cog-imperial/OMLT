"""OMLT.

OMLT is a Python package for representing machine learning models (neural networks
and gradient-boosted trees) within the Pyomo optimization environment.
The package provides various optimization formulations for machine learning models
(such as full-space, reduced-space, and MILP) as well as an interface to import
sequential Keras and general ONNX models.

"""

from omlt._version import __version__
from omlt.block import OmltBlock  # type: ignore[attr-defined]
from omlt.scaling import OffsetScaling

__all__ = [
    "OffsetScaling",
    "OmltBlock",
    "__version__",
]
