import omlt.scaling as scaling
from omlt.scaling import OffsetScaling
from omlt.block import OmltBlock
from omlt.io.sklearn_reader import convert_sklearn_scalers
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import numpy as np
from scipy.stats import iqr
import pyomo.environ as pyo

def test_sklearn_scaler_conversion():
    X = np.array(
        [[42, 10, 29],
         [12, 19, 15]]
    )

    Y = np.array(
        [[1, 2],
         [3, 4]]
    )

    # Create sklearn scalers
    xMinMax = MinMaxScaler()
    xMaxAbs = MaxAbsScaler()
    xStandard = StandardScaler()
    xRobust = RobustScaler()

    yMinMax = MinMaxScaler()
    yMaxAbs = MaxAbsScaler()
    yStandard = StandardScaler()
    yRobust = RobustScaler()

    sklearn_scalers = [(xMinMax, yMinMax), (xMaxAbs, yMaxAbs), (xStandard, yStandard), (xRobust, yRobust)]
    for scalers in sklearn_scalers:
        scalers[0].fit(X)
        scalers[1].fit(Y)

    # Create OMLT scalers using OMLT function
    MinMaxOMLT = convert_sklearn_scalers(xMinMax, yMinMax)
    MaxAbsOMLT = convert_sklearn_scalers(xMaxAbs, yMaxAbs)
    StandardOMLT = convert_sklearn_scalers(xStandard, yStandard)
    RobustOMLT = convert_sklearn_scalers(xRobust, yRobust)

    omlt_scalers = [MinMaxOMLT, MaxAbsOMLT, StandardOMLT, RobustOMLT]

    # Generate test data
    x = {0: 10, 1: 29, 2: 42}
    y = {0: 2, 1: 1}

    # Test Scalers
    for i in range(len(omlt_scalers)):
        x_s_omlt = omlt_scalers[i].get_scaled_input_expressions(x)
        y_s_omlt = omlt_scalers[i].get_scaled_output_expressions(y)

        x_s_sklearn = sklearn_scalers[i][0].transform([list(x.values())])[0]
        y_s_sklearn = sklearn_scalers[i][1].transform([list(y.values())])[0]

        np.testing.assert_almost_equal(list(x_s_omlt.values()), list(x_s_sklearn))
        np.testing.assert_almost_equal(list(y_s_omlt.values()), list(y_s_sklearn))

def test_sklearn_offset_equivalence():
    X = np.array(
        [[42, 10, 29],
         [12, 19, 15]]
    )

    Y = np.array(
        [[1, 2],
         [3, 4]]
    )

    # Get scaling factors for OffsetScaler
    xmean = X.mean(axis=0)
    xstd = X.std(axis=0)
    xmax = X.max(axis=0)
    absxmax = abs(X).max(axis=0)
    xmin = X.min(axis=0)
    xminmax = xmax-xmin
    xmedian = np.median(X, axis=0)
    xiqr = iqr(X, axis=0)

    ymean = Y.mean(axis=0)
    ystd = Y.std(axis=0)
    ymax = Y.max(axis=0)
    absymax = abs(Y).max(axis=0)
    ymin = Y.min(axis=0)
    yminmax = ymax-ymin
    ymedian = np.median(Y, axis=0)
    yiqr = iqr(Y, axis=0)

    # Create sklearn scalers
    xMinMax = MinMaxScaler()
    xMaxAbs = MaxAbsScaler()
    xStandard = StandardScaler()
    xRobust = RobustScaler()

    yMinMax = MinMaxScaler()
    yMaxAbs = MaxAbsScaler()
    yStandard = StandardScaler()
    yRobust = RobustScaler()

    sklearn_scalers = [(xMinMax, yMinMax), (xMaxAbs, yMaxAbs), (xStandard, yStandard), (xRobust, yRobust)]
    for scalers in sklearn_scalers:
        scalers[0].fit(X)
        scalers[1].fit(Y)

    # Create OMLT scalers manually
    MinMaxOMLT = OffsetScaling(offset_inputs=xmin, factor_inputs=xminmax, offset_outputs=ymin, factor_outputs=yminmax)
    MaxAbsOMLT = OffsetScaling(offset_inputs=[0]*3, factor_inputs=absxmax, offset_outputs=[0]*2, factor_outputs=absymax)
    StandardOMLT = OffsetScaling(offset_inputs=xmean, factor_inputs=xstd, offset_outputs=ymean, factor_outputs=ystd)
    RobustOMLT = OffsetScaling(offset_inputs=xmedian, factor_inputs=xiqr, offset_outputs=ymedian, factor_outputs=yiqr)

    omlt_scalers = [MinMaxOMLT, MaxAbsOMLT, StandardOMLT, RobustOMLT]

    # Generate test data
    x = {0: 10, 1: 29, 2: 42}
    y = {0: 2, 1: 1}

    # Test Scalers
    for i in range(len(omlt_scalers)):
        x_s_omlt = omlt_scalers[i].get_scaled_input_expressions(x)
        y_s_omlt = omlt_scalers[i].get_scaled_output_expressions(y)

        x_s_sklearn = sklearn_scalers[i][0].transform([list(x.values())])[0]
        y_s_sklearn = sklearn_scalers[i][1].transform([list(y.values())])[0]

        np.testing.assert_almost_equal(list(x_s_omlt.values()), list(x_s_sklearn))
        np.testing.assert_almost_equal(list(y_s_omlt.values()), list(y_s_sklearn))