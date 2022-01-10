import numpy as np
import pytest
from omlt import OffsetScaling

def test_offset_scaling():
    x_offset = [42, 65]
    x_factor = [1975, 1964]
    y_offset = [-4, 2, 1.784]
    y_factor = [2, 1.5, 1.3]

    x = {0: 1, 1: 2}
    y = {0: -1, 1: 2, 2: 3}
    x_scal = {k: (v - x_offset[k]) / x_factor[k] for (k, v) in x.items()}
    y_scal = {k: (v - y_offset[k]) / y_factor[k] for (k, v) in y.items()}

    scaling = OffsetScaling(
        offset_inputs=x_offset,
        factor_inputs=x_factor,
        offset_outputs=y_offset,
        factor_outputs=y_factor,
    )

    test_x_scal = scaling.get_scaled_input_expressions(x)
    test_y_unscal = scaling.get_unscaled_output_expressions(y_scal)
    np.testing.assert_almost_equal(list(test_x_scal.values()), list(x_scal.values()))
    np.testing.assert_almost_equal(list(test_y_unscal.values()), list(y.values()))

def test_negative_offsets():
    x_offset = [42, 65]
    x_factor = [-1975, 1964]
    y_offset = [-4, 2, 1.784]
    y_factor = [2, 1.5, 1.3]

    with pytest.raises(ValueError) as excinfo:
        scaling = OffsetScaling(
            offset_inputs=x_offset,
            factor_inputs=x_factor,
            offset_outputs=y_offset,
            factor_outputs=y_factor,
        )
    assert str(excinfo.value) == "OffsetScaling only accepts positive values" \
       " for factor_inputs. Negative value found at" \
       " index 0."

    x_offset = [42, 65]
    x_factor = [1975, 1964]
    y_offset = [-4, 2, 1.784]
    y_factor = [2, -1.5, 1.3]

    with pytest.raises(ValueError) as excinfo:
        scaling = OffsetScaling(
            offset_inputs=x_offset,
            factor_inputs=x_factor,
            offset_outputs=y_offset,
            factor_outputs=y_factor,
        )
    assert str(excinfo.value) == "OffsetScaling only accepts positive values" \
       " for factor_outputs. Negative value found at" \
       " index 1."

    
