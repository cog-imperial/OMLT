import numpy as np
import pyomo.environ as pe
from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression
import pytest

from omlt import OmltBlock
from omlt.lineartree import LinearTreeGDPFormulation, LinearTreeModel
import omlt

X_small = np.array([[-0.68984135],
    [ 0.91672866],
    [-1.05874972],
    [ 0.95275351],
    [ 1.03796615],
    [ 0.45117668],
    [-0.14704376],
    [ 1.66043409],
    [-0.73972191],
    [-0.8176603 ],
    [ 0.96175973],
    [-1.238874  ],
    [-0.97492265],
    [ 1.07121986],
    [-0.95379269],
    [-0.86546252],
    [ 0.8277057 ],
    [ 0.50486757],
    [-1.38435899],
    [ 1.54092856]])
y_small = np.array([[ 0.04296633],
    [-0.78349216],
    [ 0.27114188],
    [-0.58516476],
    [-0.15997756],
    [-0.37529212],
    [-1.49249696],
    [ 1.56412122],
    [ 0.18697725],
    [ 0.4035928 ],
    [-0.53231771],
    [-0.02669967],
    [ 0.36972983],
    [ 0.09201347],
    [ 0.44041505],
    [ 0.46047019],
    [-1.04855941],
    [-0.586915  ],
    [ 0.15472157],
    [ 1.71225268]])

def linear_model_tree(X, y):
    regr = LinearTreeRegressor(LinearRegression(), criterion='mse', max_depth=5)
    regr.fit(X, y)
    return regr

def test_linear_tree_model():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds={0:(min(X_small)[0], max(X_small)[0])}
    # TODO: check the type of the lmt(in the future there will be two ways)
    # assert(str(type(regr_small)) == "<class 'lineartree.lineartree.LinearTreeRegressor'>")
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds = input_bounds)
    # TODO: is _scaling_object are essential? no
    assert(ltmodel_small._scaled_input_bounds is not None)
    assert(ltmodel_small._n_inputs == 1)
    assert(ltmodel_small._n_outputs == 1)
    # test for splits
    assert(len(ltmodel_small._splits.keys()) == 5)
    splits_key_list = ['col', 'th', 'loss', 'samples', 'parent', 'children', 'models', 'left_leaves', 'right_leaves', 'y_index']
    for i in ltmodel_small._splits.keys():
        for key in ltmodel_small._splits[i].keys():
            assert(key in splits_key_list)
    # test for leaves
    assert(len(ltmodel_small._leaves.keys()) == 6)
    leaves_key_list = ['loss', 'samples', 'models', 'slope', 'intercept', 'parent', 'bounds']
    for j in ltmodel_small._leaves.keys():
        for key in ltmodel_small._leaves[j].keys():
            assert(key in leaves_key_list)
            if key == 'slope':
                assert(len(ltmodel_small._leaves[j][key]) == ltmodel_small._n_inputs)
            elif key == 'bounds':
                features = ltmodel_small._leaves[j][key].keys()
                for k in range(len(features)):
                    lb = ltmodel_small._leaves[j][key][k][0]
                    ub = ltmodel_small._leaves[j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert(lb <= ub)
    # TODO: do we need to test slope and bounds in leaves_dic
    # test for thresholds
    # test whether each feature has threshold
    assert(len(ltmodel_small._thresholds.keys()) == ltmodel_small._n_inputs)
    thresholds_count = 0
    for k in range(len(ltmodel_small._thresholds.keys())):
        for _ in range(len(ltmodel_small._thresholds[k].keys())):
            thresholds_count += 1
    # test number of thresholds
    assert(thresholds_count == len(ltmodel_small._splits.keys()))

def test_bigm_formulation():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds={0:(min(X_small)[0], max(X_small)[0])} 
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds = input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='bigm')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize = 0)
    model1.y = pe.Var(initialize = 0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x.fix(0.5)
    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]
    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x),pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1,-1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)

def test_hull_formulation():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds={0:(min(X_small)[0], max(X_small)[0])} 
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds = input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='hull')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize = 0)
    model1.y = pe.Var(initialize = 0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x.fix(0.5)
    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]
    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x),pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1,-1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)

def test_mbigm_formulation():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds={0:(min(X_small)[0], max(X_small)[0])} 
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds = input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='mbigm')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize = 0)
    model1.y = pe.Var(initialize = 0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x.fix(0.5)
    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]
    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x),pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1,-1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)