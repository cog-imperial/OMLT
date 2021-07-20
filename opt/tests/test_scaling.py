import pytest
import numpy as np
from pyoml.opt.scaling import OffsetScaling

def test_offset_scaling():
    xdata = np.random.normal(1,0.5,(2,10))
    ydata = np.random.normal(-1,0.2,(3,10))
    x_offset = np.mean(xdata, axis=-1)
    x_factor = np.std(xdata, axis=-1)
    y_offset = np.mean(ydata, axis=-1)
    y_factor = np.std(ydata, axis=-1)

    x = [1,2]
    y = [-1,2,3]
    x_scal = (np.asarray(x)-x_offset)/x_factor
    y_scal = (np.asarray(y)-y_offset)/y_factor

    scaling = OffsetScaling(offset_inputs=x_offset,
                            factor_inputs=x_factor,
                            offset_outputs=y_offset,
                            factor_outputs=y_factor)

    test_x_scal = scaling.get_scaled_input_expressions(x)
    test_y_unscal = scaling.get_unscaled_output_expressions(y_scal)
    np.testing.assert_almost_equal(test_x_scal, x_scal)
    np.testing.assert_almost_equal(test_y_unscal, y)
    
    
