import numpy as np
import pytest
from omlt import OffsetScaling
from omlt.scaling import convert_to_dict

def test_convert_to_dict():
    x = ['a', 'b']
    x = convert_to_dict(x)
    assert sorted(x.keys()) == [0, 1]
    assert x[0] == 'a'
    assert x[1] == 'b'

    x = {2: 'a', 1: 'b'}
    x = convert_to_dict(x)
    assert sorted(x.keys()) == [1, 2]
    assert x[2] == 'a'
    assert x[1] == 'b'


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
    test_x = scaling.get_unscaled_input_expressions(test_x_scal)
    test_y = scaling.get_unscaled_output_expressions(y_scal)
    test_y_scal = scaling.get_scaled_output_expressions(y)
    np.testing.assert_almost_equal(list(test_x_scal.values()), list(x_scal.values()))
    np.testing.assert_almost_equal(list(test_x.values()), list(x.values()))
    np.testing.assert_almost_equal(list(test_y.values()), list(y.values()))
    np.testing.assert_almost_equal(list(test_y_scal.values()), list(y_scal.values()))
    
def test_incorrect_keys():
    x_offset = {1: 42, 42: 65}
    x_factor = {1: 1975, 42: 1964}
    y_offset = {7: -4, 9: 2, 11: 1.784}
    y_factor = {7: 2, 9: 1.5, 11: 1.3}

    x = {1: 1, 42: 2}
    y = {7: -1, 9: 2, 11: 3}
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

    x = {1: 42, 2: 65}
    with pytest.raises(ValueError) as excinfo:
        test_x_scal = scaling.get_scaled_input_expressions(x)
    expected_msg = "get_scaled_input_expressions called with input_vars that " \
        "do not have the same indices as offset_inputs or factor_inputs.\nKeys " \
        "in input_vars: [1, 2].\nKeys in offset_inputs: [1, 42].\nKeys in "\
        "offset_factor: [1, 42]."
    assert str(excinfo.value) == expected_msg

    y = {7: -1, 19: 2, 11: 3}
    with pytest.raises(ValueError) as excinfo:
        test_y_scal = scaling.get_scaled_output_expressions(y)
    expected_msg = "get_scaled_output_expressions called with output_vars that " \
        "do not have the same indices as offset_outputs or factor_outputs.\nKeys " \
        "in output_vars: [7, 11, 19]\nKeys in offset_outputs: [7, 9, 11]\nKeys in " \
        "offset_factor: [7, 9, 11]"
    assert str(excinfo.value) == expected_msg

    x_scal = {1: 42, 2: 65}
    with pytest.raises(ValueError) as excinfo:
        test_x_unscal = scaling.get_unscaled_input_expressions(x_scal)
    expected_msg = "get_scaled_input_expressions called with input_vars that " \
        "do not have the same indices as offset_inputs or factor_inputs.\nKeys " \
        "in input_vars: [1, 2]\nKeys in offset_inputs: [1, 42]\nKeys in " \
        "offset_factor: [1, 42]"
    assert str(excinfo.value) == expected_msg
    
    y_scal = {7: -1, 8: 2, 11: 3}
    with pytest.raises(ValueError) as excinfo:
        test_y_unscal = scaling.get_unscaled_output_expressions(y_scal)
    expected_msg = "get_scaled_output_expressions called with output_vars that do " \
        "not have the same indices as offset_outputs or factor_outputs.\nKeys in " \
        "output_vars: [7, 8, 11]\nKeys in offset_outputs: [7, 9, 11]\nKeys in " \
        "offset_factor: [7, 9, 11]"
    assert str(excinfo.value) == expected_msg
    
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

    
if __name__ == '__main__':
    test_incorrect_keys()
