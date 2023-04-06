import numpy as np
import pyomo.environ as pe
from lineartree import LinearTreeRegressor
from sklearn.linear_model import LinearRegression
import pytest

from omlt import OmltBlock
from omlt.lineartree import LinearTreeGDPFormulation, LinearTreeModel, LinearTreeHybridBigMFormulation
import omlt

def linear_model_tree(X, y):
    regr = LinearTreeRegressor(LinearRegression(), criterion='mse', max_depth=5)
    regr.fit(X, y)
    return regr

### SINGLE VARIABLE INPUT TESTING ####

X_small = np.array([[-0.68984135],
                    [0.91672866],
                    [-1.05874972],
                    [0.95275351],
                    [1.03796615],
                    [0.45117668],
                    [-0.14704376],
                    [1.66043409],
                    [-0.73972191],
                    [-0.8176603],
                    [0.96175973],
                    [-1.238874],
                    [-0.97492265],
                    [1.07121986],
                    [-0.95379269],
                    [-0.86546252],
                    [0.8277057],
                    [0.50486757],
                    [-1.38435899],
                    [1.54092856]])

y_small = np.array([[0.04296633],
                    [-0.78349216],
                    [0.27114188],
                    [-0.58516476],
                    [-0.15997756],
                    [-0.37529212],
                    [-1.49249696],
                    [1.56412122],
                    [0.18697725],
                    [0.4035928],
                    [-0.53231771],
                    [-0.02669967],
                    [0.36972983],
                    [0.09201347],
                    [0.44041505],
                    [0.46047019],
                    [-1.04855941],
                    [-0.586915],
                    [0.15472157],
                    [1.71225268]])


def test_linear_tree_model_single_var():
    # construct a LinearTreeModel
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds=input_bounds)

    assert(ltmodel_small._scaled_input_bounds is not None)
    assert(ltmodel_small._n_inputs == 1)
    assert(ltmodel_small._n_outputs == 1)
    # test for splits
    # assert the number of splits
    assert(len(ltmodel_small._splits.keys()) == 5)
    splits_key_list = ['col', 'th', 'loss', 'samples', 'parent',
                       'children', 'models', 'left_leaves', 'right_leaves', 'y_index']
    # assert whether all the dicts have such keys
    for i in ltmodel_small._splits.keys():
        for key in ltmodel_small._splits[i].keys():
            assert(key in splits_key_list)
    # test for leaves
    # assert the number of leaves
    assert(len(ltmodel_small._leaves.keys()) == 6)
    # assert whether all the dicts have such keys
    leaves_key_list = ['loss', 'samples', 'models',
                       'slope', 'intercept', 'parent', 'bounds']
    for j in ltmodel_small._leaves.keys():
        for key in ltmodel_small._leaves[j].keys():
            assert(key in leaves_key_list)
            # if the key is slope, test the shape of it
            if key == 'slope':
                assert(len(ltmodel_small._leaves[j][key]) == ltmodel_small._n_inputs)
            # elif the key is bounds, test the lb <= ub
            elif key == 'bounds':
                features = ltmodel_small._leaves[j][key].keys()
                for k in range(len(features)):
                    lb = ltmodel_small._leaves[j][key][k][0]
                    ub = ltmodel_small._leaves[j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert(lb <= ub)
    # test for thresholds
    # assert whether each feature has threshold
    assert(len(ltmodel_small._thresholds.keys()) == ltmodel_small._n_inputs)
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(ltmodel_small._thresholds.keys())):
        for _ in range(len(ltmodel_small._thresholds[k].keys())):
            thresholds_count += 1
    assert(thresholds_count == len(ltmodel_small._splits.keys()))


def test_bigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='bigm')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
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
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)


def test_hull_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='hull')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
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
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)


def test_mbigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='mbigm')
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
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
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)


def test_hybrid_bigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeModel(regr_small, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeHybridBigMFormulation(ltmodel_small)
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
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
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm[1] <= 1e-4)


#### MULTIVARIATE INPUT TESTING ####
X = np.array([[4.98534526, 1.8977914 ],
              [4.38751717, 4.48456528],
              [2.65451539, 2.44426211],
              [3.32761277, 4.58757063],
              [0.36806515, 0.82428634],
              [4.16036314, 1.09680059],
              [2.29025371, 0.72246559],
              [1.92725929, 0.34359974],
              [4.02101578, 1.39448628],
              [3.28019501, 1.22160752],
              [2.73026047, 3.9482306 ],
              [0.45621172, 0.56130164],
              [2.64296795, 4.75411397],
              [4.72526084, 3.35223772],
              [2.39270941, 4.41622262],
              [4.42707908, 0.35276571],
              [1.58452501, 3.28957671],
              [0.20009184, 2.90255483],
              [4.36453075, 3.61985047],
              [1.05576503, 2.57532169]])

Y = np.array([[10.23341638],
              [ 4.00860872],
              [ 3.85046103],
              [ 9.48457266],
              [ 6.36974536],
              [ 3.19763555],
              [ 4.78390803],
              [ 1.51994021],
              [ 3.18768132],
              [ 3.7972809 ],
              [ 7.93779383],
              [ 3.46714285],
              [ 7.89435163],
              [10.62832561],
              [ 1.50713442],
              [ 7.44321537],
              [ 9.39437373],
              [ 4.38891182],
              [ 1.32105126],
              [ 3.37287403]])


def test_linear_tree_model_multi_var():
    # construct a LinearTreeModel
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr, scaled_input_bounds=input_bounds)
    # assert attributes in LinearTreeModel
    assert(ltmodel_small._scaled_input_bounds is not None)
    assert(ltmodel_small._n_inputs == 2)
    assert(ltmodel_small._n_outputs == 1)
    # test for splits
    # assert the number of splits
    assert(len(ltmodel_small._splits.keys()) == 5)
    splits_key_list = ['col', 'th', 'loss', 'samples', 'parent',
                       'children', 'models', 'left_leaves', 'right_leaves', 'y_index']
    # assert whether all the dicts have such keys
    for i in ltmodel_small._splits.keys():
        for key in ltmodel_small._splits[i].keys():
            assert(key in splits_key_list)
    # test for leaves
    # assert the number of leaves
    assert(len(ltmodel_small._leaves.keys()) == 6)
    # assert whether all the dicts have such keys
    leaves_key_list = ['loss', 'samples', 'models',
                       'slope', 'intercept', 'parent', 'bounds']
    for j in ltmodel_small._leaves.keys():
        for key in ltmodel_small._leaves[j].keys():
            assert(key in leaves_key_list)
            # if the key is slope, test the shape of it
            if key == 'slope':
                assert(len(ltmodel_small._leaves[j][key]) == ltmodel_small._n_inputs)
            # elif the key is bounds, test the lb <= ub
            elif key == 'bounds':
                features = ltmodel_small._leaves[j][key].keys()
                for k in range(len(features)):
                    lb = ltmodel_small._leaves[j][key][k][0]
                    ub = ltmodel_small._leaves[j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert(lb <= ub)
    # test for thresholds
    # assert whether each feature has threshold
    assert(len(ltmodel_small._thresholds.keys()) == ltmodel_small._n_inputs)
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(ltmodel_small._thresholds.keys())):
        for _ in range(len(ltmodel_small._thresholds[k].keys())):
            thresholds_count += 1
    assert(thresholds_count == len(ltmodel_small._splits.keys()))


def test_bigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='bigm')
    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]
    
    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(np.array([pe.value(model1.x0), pe.value(model1.x1)]
                                   ).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm <= 1e-4)


def test_hull_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='hull')
    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]
    
    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(np.array([pe.value(model1.x0), pe.value(model1.x1)]
                                   ).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm <= 1e-4)


def test_mbigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation='mbigm')
    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]
    
    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(np.array([pe.value(model1.x0), pe.value(model1.x1)]
                                   ).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm <= 1e-4)


def test_hybrid_bigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr, scaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeHybridBigMFormulation(ltmodel_small)
    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)
    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]
    
    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]
    status_1_bigm = pe.SolverFactory('gurobi').solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(np.array([pe.value(model1.x0), pe.value(model1.x1)]
                                   ).reshape(1, -1))
    assert(y_pred[0] - solution_1_bigm <= 1e-4)


def test_summary_dict_as_argument():
    # construct a LinearTreeModel
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:,0]), max(X[:,0])),
                    1: (min(X[:,1]), max(X[:,1]))}
    ltmodel_small = LinearTreeModel(regr.summary(), scaled_input_bounds=input_bounds)
    # assert attributes in LinearTreeModel
    assert(ltmodel_small._scaled_input_bounds is not None)
    assert(ltmodel_small._n_inputs == 2)
    assert(ltmodel_small._n_outputs == 1)
    # test for splits
    # assert the number of splits
    assert(len(ltmodel_small._splits.keys()) == 5)
    splits_key_list = ['col', 'th', 'loss', 'samples', 'parent',
                       'children', 'models', 'left_leaves', 'right_leaves', 'y_index']
    # assert whether all the dicts have such keys
    for i in ltmodel_small._splits.keys():
        for key in ltmodel_small._splits[i].keys():
            assert(key in splits_key_list)
    # test for leaves
    # assert the number of leaves
    assert(len(ltmodel_small._leaves.keys()) == 6)
    # assert whether all the dicts have such keys
    leaves_key_list = ['loss', 'samples', 'models',
                       'slope', 'intercept', 'parent', 'bounds']
    for j in ltmodel_small._leaves.keys():
        for key in ltmodel_small._leaves[j].keys():
            assert(key in leaves_key_list)
            # if the key is slope, test the shape of it
            if key == 'slope':
                assert(len(ltmodel_small._leaves[j][key]) == ltmodel_small._n_inputs)
            # elif the key is bounds, test the lb <= ub
            elif key == 'bounds':
                features = ltmodel_small._leaves[j][key].keys()
                for k in range(len(features)):
                    lb = ltmodel_small._leaves[j][key][k][0]
                    ub = ltmodel_small._leaves[j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert(lb <= ub)
    # test for thresholds
    # assert whether each feature has threshold
    assert(len(ltmodel_small._thresholds.keys()) == ltmodel_small._n_inputs)
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(ltmodel_small._thresholds.keys())):
        for _ in range(len(ltmodel_small._thresholds[k].keys())):
            thresholds_count += 1
    assert(thresholds_count == len(ltmodel_small._splits.keys()))