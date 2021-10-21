"""
    Dummy conftest.py for omlt.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest

import numpy as np


def get_neural_network_data(desc):
    """
    Return input and test data for a neural network.

    Parameters
    ----------
    desc : string
        model name. One of 131 or 2353.
    """
    if desc == "131":
        # build data with 1 input and 1 output and 500 data points
        x = np.random.uniform(-1, 1, 500)
        y = np.sin(x)
        x_test = np.random.uniform(-1, 1, 5)
        return x, y, x_test

    elif desc == "2353":
        # build data with 2 inputs, 3 outputs, and 500 data points
        np.random.seed(42)
        x = np.random.uniform([-1, 2], [1, 3], (500, 2))
        y1 = np.sin(x[:, 0] * x[:, 1])
        y2 = x[:, 0] + x[:, 1]
        y3 = np.cos(x[:, 0] / x[:, 1])
        y = np.column_stack((y1, y2, y3))
        x_test = np.random.uniform([-1, 2], [1, 3], (5, 2))
        return x, y, x_test

    return None