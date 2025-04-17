import numpy as np
import pyomo.environ as pe
import pytest
from pyomo.common.collections import ComponentSet
from pyomo.core.expr import identify_variables

from omlt.dependencies import lineartree_available

if lineartree_available:
    from lineartree import LinearTreeRegressor
    from sklearn.linear_model import LinearRegression

    from omlt.linear_tree import (
        LinearTreeDefinition,
        LinearTreeGDPFormulation,
        LinearTreeHybridBigMFormulation,
    )

import omlt
from omlt import OmltBlock

NUM_INPUTS = 2
NUM_SPLITS = 5
NUM_LEAVES = 6

scip_available = pe.SolverFactory("scip").available()
cbc_available = pe.SolverFactory("cbc").available()
gurobi_available = pe.SolverFactory("gurobi").available()


def linear_model_tree(X, y):
    regr = LinearTreeRegressor(LinearRegression(), criterion="mse", max_depth=5)
    regr.fit(X, y)
    return regr


### SINGLE VARIABLE INPUT TESTING ####

X_small = np.array(
    [
        [-0.68984135],
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
        [1.54092856],
    ]
)

y_small = np.array(
    [
        [0.04296633],
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
        [1.71225268],
    ]
)


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_linear_tree_model_single_var():  # noqa: C901
    # construct a LinearTreeDefinition
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)

    scaled_input_bounds = ltmodel_small.scaled_input_bounds
    n_inputs = ltmodel_small.n_inputs
    n_outputs = ltmodel_small.n_outputs
    splits = ltmodel_small.splits
    leaves = ltmodel_small.leaves
    thresholds = ltmodel_small.thresholds
    is_scaled = ltmodel_small.is_scaled
    unscaled_input_bounds = ltmodel_small.unscaled_input_bounds

    assert scaled_input_bounds is not None
    assert unscaled_input_bounds is not None
    assert not is_scaled
    assert n_inputs == 1
    assert n_outputs == 1
    # test for splits
    # assert the number of splits
    assert len(splits[0].keys()) == NUM_SPLITS
    splits_key_list = [
        "col",
        "th",
        "loss",
        "samples",
        "parent",
        "children",
        "models",
        "left_leaves",
        "right_leaves",
        "y_index",
    ]
    # assert whether all the dicts have such keys
    for i in splits[0]:
        for key in splits[0][i]:
            assert key in splits_key_list
    # test for leaves
    # assert the number of leaves
    assert len(leaves[0].keys()) == NUM_LEAVES
    # assert whether all the dicts have such keys
    leaves_key_list = [
        "loss",
        "samples",
        "models",
        "slope",
        "intercept",
        "parent",
        "bounds",
    ]
    for j in leaves[0]:
        for key in leaves[0][j]:
            assert key in leaves_key_list
            # if the key is slope, ensure slope dimension match n_inputs
            if key == "slope":
                assert len(leaves[0][j][key]) == n_inputs
            # elif the key is bounds, test ensure lb <= ub
            elif key == "bounds":
                features = leaves[0][j][key].keys()
                for k in range(len(features)):
                    lb = leaves[0][j][key][k][0]
                    ub = leaves[0][j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert lb <= ub
    # test for thresholds
    # assert whether each feature has threshold
    assert len(thresholds[0].keys()) == n_inputs
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(thresholds[0].keys())):
        for _ in range(len(thresholds[0][k].keys())):
            thresholds_count += 1
    assert thresholds_count == len(splits[0].keys())


@pytest.mark.skipif(
    not lineartree_available or not cbc_available,
    reason="Need Linear-Tree Package and cbc",
)
def test_bigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="bigm")

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("cbc").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert y_pred[0] == pytest.approx(solution_1_bigm[1])


def get_epsilon_test_model(formulation_lt):
    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=model1.y, sense=pe.maximize)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation_lt)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(1.058749)

    return model1


@pytest.mark.skipif(
    not lineartree_available or not cbc_available,
    reason="Need Linear-Tree Package and cbc",
)
def test_nonzero_epsilon():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)
    formulation_bad = LinearTreeGDPFormulation(
        ltmodel_small, transformation="bigm", epsilon=0
    )
    formulation1_lt = LinearTreeGDPFormulation(
        ltmodel_small, transformation="bigm", epsilon=1e-4
    )

    model_good = get_epsilon_test_model(formulation1_lt)
    model_bad = get_epsilon_test_model(formulation_bad)

    status_1_bigm = pe.SolverFactory("cbc").solve(model_bad)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model_bad.x), pe.value(model_bad.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    # Without an epsilon, the model cheats and does not match the tree prediction
    assert y_pred[0] != pytest.approx(solution_1_bigm[1])

    status = pe.SolverFactory("cbc").solve(model_good)
    pe.assert_optimal_termination(status)
    solution = (pe.value(model_good.x), pe.value(model_good.y))
    y_pred = regr_small.predict(np.array(solution[0]).reshape(1, -1))
    # With epsilon, the model matches the tree prediction
    assert y_pred[0] == pytest.approx(solution[1], abs=1e-4)


@pytest.mark.skipif(
    not lineartree_available or not cbc_available,
    reason="Need Linear-Tree Package and cbc",
)
def test_hull_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="hull")

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("cbc").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert y_pred[0] == pytest.approx(solution_1_bigm[1])


@pytest.mark.skipif(
    not lineartree_available or not gurobi_available,
    reason="Need Linear-Tree Package and gurobi",
)
def test_mbigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="mbigm")

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("gurobi").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert y_pred[0] == pytest.approx(solution_1_bigm[1])


@pytest.mark.skipif(
    not lineartree_available or not scip_available,
    reason="Need Linear-Tree Package and scip",
)
def test_hybrid_bigm_formulation_single_var():
    regr_small = linear_model_tree(X=X_small, y=y_small)
    input_bounds = {0: (min(X_small)[0], max(X_small)[0])}
    ltmodel_small = LinearTreeDefinition(regr_small, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeHybridBigMFormulation(ltmodel_small)

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("scip").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr_small.predict(np.array(solution_1_bigm[0]).reshape(1, -1))
    assert y_pred[0] == pytest.approx(solution_1_bigm[1])


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_scaling_only_scaler():
    mean_x_small = np.mean(X_small)
    std_x_small = np.std(X_small)
    mean_y_small = np.mean(y_small)
    std_y_small = np.std(y_small)
    scaled_x = (X_small - mean_x_small) / std_x_small
    scaled_y = (y_small - mean_y_small) / std_y_small
    scaled_input_bounds = {0: (np.min(scaled_x), np.max(scaled_x))}
    unscaled_input_bounds = {0: (np.min(X_small), np.max(X_small))}

    scaler = omlt.scaling.OffsetScaling(
        offset_inputs=[mean_x_small],
        factor_inputs=[std_x_small],
        offset_outputs=[mean_y_small],
        factor_outputs=[std_y_small],
    )

    regr = linear_model_tree(scaled_x, scaled_y)

    regr.fit(np.reshape(scaled_x, (-1, 1)), scaled_y)

    lt_def2 = LinearTreeDefinition(
        regr, unscaled_input_bounds=unscaled_input_bounds, scaling_object=scaler
    )
    assert lt_def2.scaled_input_bounds[0][0] == pytest.approx(scaled_input_bounds[0][0])
    assert lt_def2.scaled_input_bounds[0][1] == pytest.approx(scaled_input_bounds[0][1])
    with pytest.raises(
        Exception, match="Input Bounds needed to represent linear trees as MIPs"
    ):
        LinearTreeDefinition(regr)

    formulation = LinearTreeHybridBigMFormulation(lt_def2)

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("scip").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr.predict(
        np.array((solution_1_bigm[0] - mean_x_small) / std_x_small).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx((solution_1_bigm[1] - mean_y_small) / std_y_small)


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_scaling_bounds_and_scaler():
    mean_x_small = np.mean(X_small)
    std_x_small = np.std(X_small)
    mean_y_small = np.mean(y_small)
    std_y_small = np.std(y_small)
    scaled_x = (X_small - mean_x_small) / std_x_small
    scaled_y = (y_small - mean_y_small) / std_y_small
    scaled_input_bounds = {0: (np.min(scaled_x), np.max(scaled_x))}
    unscaled_input_bounds = {0: (np.min(X_small), np.max(X_small))}

    scaler = omlt.scaling.OffsetScaling(
        offset_inputs=[mean_x_small],
        factor_inputs=[std_x_small],
        offset_outputs=[mean_y_small],
        factor_outputs=[std_y_small],
    )

    regr = linear_model_tree(scaled_x, scaled_y)

    regr.fit(np.reshape(scaled_x, (-1, 1)), scaled_y)

    lt_def2 = LinearTreeDefinition(
        regr, scaled_input_bounds=scaled_input_bounds, scaling_object=scaler
    )
    assert lt_def2.scaled_input_bounds[0][0] == pytest.approx(scaled_input_bounds[0][0])
    assert lt_def2.scaled_input_bounds[0][1] == pytest.approx(scaled_input_bounds[0][1])
    assert lt_def2.unscaled_input_bounds[0][0] == pytest.approx(
        unscaled_input_bounds[0][0]
    )
    assert lt_def2.unscaled_input_bounds[0][1] == pytest.approx(
        unscaled_input_bounds[0][1]
    )
    with pytest.raises(
        Exception, match="Input Bounds needed to represent linear trees as MIPs"
    ):
        LinearTreeDefinition(regr)

    formulation = LinearTreeHybridBigMFormulation(lt_def2)

    model1 = pe.ConcreteModel()
    model1.x = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation)

    @model1.Constraint()
    def connect_inputs(mdl):
        return mdl.x == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x.fix(0.5)

    status_1_bigm = pe.SolverFactory("scip").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = (pe.value(model1.x), pe.value(model1.y))
    y_pred = regr.predict(
        np.array((solution_1_bigm[0] - mean_x_small) / std_x_small).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx((solution_1_bigm[1] - mean_y_small) / std_y_small)


#### MULTIVARIATE INPUT TESTING ####

X = np.array(
    [
        [4.98534526, 1.8977914],
        [4.38751717, 4.48456528],
        [2.65451539, 2.44426211],
        [3.32761277, 4.58757063],
        [0.36806515, 0.82428634],
        [4.16036314, 1.09680059],
        [2.29025371, 0.72246559],
        [1.92725929, 0.34359974],
        [4.02101578, 1.39448628],
        [3.28019501, 1.22160752],
        [2.73026047, 3.9482306],
        [0.45621172, 0.56130164],
        [2.64296795, 4.75411397],
        [4.72526084, 3.35223772],
        [2.39270941, 4.41622262],
        [4.42707908, 0.35276571],
        [1.58452501, 3.28957671],
        [0.20009184, 2.90255483],
        [4.36453075, 3.61985047],
        [1.05576503, 2.57532169],
    ]
)

Y = np.array(
    [
        [10.23341638],
        [4.00860872],
        [3.85046103],
        [9.48457266],
        [6.36974536],
        [3.19763555],
        [4.78390803],
        [1.51994021],
        [3.18768132],
        [3.7972809],
        [7.93779383],
        [3.46714285],
        [7.89435163],
        [10.62832561],
        [1.50713442],
        [7.44321537],
        [9.39437373],
        [4.38891182],
        [1.32105126],
        [3.37287403],
    ]
)


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_linear_tree_model_multi_var():  # noqa: C901
    # construct a LinearTreeDefinition
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)

    scaled_input_bounds = ltmodel_small.scaled_input_bounds
    n_inputs = ltmodel_small.n_inputs
    n_outputs = ltmodel_small.n_outputs
    splits = ltmodel_small.splits
    leaves = ltmodel_small.leaves
    thresholds = ltmodel_small.thresholds

    # assert attributes in LinearTreeDefinition
    assert scaled_input_bounds is not None
    assert n_inputs == NUM_INPUTS
    assert n_outputs == 1

    # test for splits
    # assert the number of splits
    assert len(splits[0].keys()) == NUM_SPLITS
    splits_key_list = [
        "col",
        "th",
        "loss",
        "samples",
        "parent",
        "children",
        "models",
        "left_leaves",
        "right_leaves",
        "y_index",
    ]
    # assert whether all the dicts have such keys
    for i in splits[0]:
        for key in splits[0][i]:
            assert key in splits_key_list
    # test for leaves
    # assert the number of leaves
    assert len(leaves[0].keys()) == NUM_LEAVES
    # assert whether all the dicts have such keys
    leaves_key_list = [
        "loss",
        "samples",
        "models",
        "slope",
        "intercept",
        "parent",
        "bounds",
    ]
    for j in leaves[0]:
        for key in leaves[0][j]:
            assert key in leaves_key_list
            # if the key is slope, test the shape of it
            if key == "slope":
                assert len(leaves[0][j][key]) == n_inputs
            # elif the key is bounds, test the lb <= ub
            elif key == "bounds":
                features = leaves[0][j][key].keys()
                for k in range(len(features)):
                    lb = leaves[0][j][key][k][0]
                    ub = leaves[0][j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert lb <= ub
    # test for thresholds
    # assert whether each feature has threshold
    assert len(thresholds[0].keys()) == n_inputs
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(thresholds[0].keys())):
        for _ in range(len(thresholds[0][k].keys())):
            thresholds_count += 1
    assert thresholds_count == len(splits[0].keys())


@pytest.mark.skipif(
    not lineartree_available or not cbc_available,
    reason="Need Linear-Tree Package and cbc",
)
def test_bigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="bigm")

    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    status_1_bigm = pe.SolverFactory("cbc").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(
        np.array([pe.value(model1.x0), pe.value(model1.x1)]).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx(solution_1_bigm)


@pytest.mark.skipif(
    not lineartree_available or not cbc_available,
    reason="Need Linear-Tree Package and cbc",
)
def test_hull_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="hull")

    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    status_1_bigm = pe.SolverFactory("cbc").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(
        np.array([pe.value(model1.x0), pe.value(model1.x1)]).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx(solution_1_bigm)


@pytest.mark.skipif(
    not lineartree_available or not gurobi_available,
    reason="Need Linear-Tree Package and gurobi",
)
def test_mbigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeGDPFormulation(ltmodel_small, transformation="mbigm")

    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    status_1_bigm = pe.SolverFactory("gurobi").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(
        np.array([pe.value(model1.x0), pe.value(model1.x1)]).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx(solution_1_bigm)


@pytest.mark.skipif(
    not lineartree_available or not scip_available,
    reason="Need Linear-Tree Package and scip",
)
def test_hybrid_bigm_formulation_multi_var():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)
    formulation1_lt = LinearTreeHybridBigMFormulation(ltmodel_small)

    model1 = pe.ConcreteModel()
    model1.x0 = pe.Var(initialize=0)
    model1.x1 = pe.Var(initialize=0)
    model1.y = pe.Var(initialize=0)
    model1.obj = pe.Objective(expr=1)
    model1.lt = OmltBlock()
    model1.lt.build_formulation(formulation1_lt)

    num_constraints = 0
    var_set = ComponentSet()
    for cons in model1.lt.component_data_objects(pe.Constraint, active=True):
        num_constraints += 1
        for v in identify_variables(cons.expr):
            var_set.add(v)

    num_leaves = len(ltmodel_small.leaves[0])
    # binary for each leaf + two inputs and an output + 5 scaled input/output vars
    assert len(var_set) == num_leaves + 3 + 4
    # 2 bounds constraints for each input, the xor, the output constraint, and
    # four scaling constraints from OMLT
    assert num_constraints == 2 * 2 + 1 + 1 + 4

    @model1.Constraint()
    def connect_input1(mdl):
        return mdl.x0 == mdl.lt.inputs[0]

    @model1.Constraint()
    def connect_input2(mdl):
        return mdl.x1 == mdl.lt.inputs[1]

    @model1.Constraint()
    def connect_outputs(mdl):
        return mdl.y == mdl.lt.outputs[0]

    model1.x0.fix(0.5)
    model1.x1.fix(0.8)

    status_1_bigm = pe.SolverFactory("scip").solve(model1, tee=True)
    pe.assert_optimal_termination(status_1_bigm)
    solution_1_bigm = pe.value(model1.y)
    y_pred = regr.predict(
        np.array([pe.value(model1.x0), pe.value(model1.x1)]).reshape(1, -1)
    )
    assert y_pred[0] == pytest.approx(solution_1_bigm)


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_summary_dict_as_argument():  # noqa: C901
    # construct a LinearTreeDefinition
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    ltmodel_small = LinearTreeDefinition(
        regr.summary(), unscaled_input_bounds=input_bounds
    )

    scaled_input_bounds = ltmodel_small.scaled_input_bounds
    n_inputs = ltmodel_small.n_inputs
    n_outputs = ltmodel_small.n_outputs
    splits = ltmodel_small.splits
    leaves = ltmodel_small.leaves
    thresholds = ltmodel_small.thresholds

    # assert attributes in LinearTreeDefinition
    assert scaled_input_bounds is not None
    assert n_inputs == NUM_INPUTS
    assert n_outputs == 1
    # test for splits
    # assert the number of splits
    assert len(splits[0].keys()) == NUM_SPLITS
    splits_key_list = [
        "col",
        "th",
        "loss",
        "samples",
        "parent",
        "children",
        "models",
        "left_leaves",
        "right_leaves",
        "y_index",
    ]
    # assert whether all the dicts have such keys
    for i in splits[0]:
        for key in splits[0][i]:
            assert key in splits_key_list
    # test for leaves
    # assert the number of leaves
    assert len(leaves[0].keys()) == NUM_LEAVES
    # assert whether all the dicts have such keys
    leaves_key_list = [
        "loss",
        "samples",
        "models",
        "slope",
        "intercept",
        "parent",
        "bounds",
    ]
    for j in leaves[0]:
        for key in leaves[0][j]:
            assert key in leaves_key_list
            # if the key is slope, test the shape of it
            if key == "slope":
                assert len(leaves[0][j][key]) == n_inputs
            # elif the key is bounds, test the lb <= ub
            elif key == "bounds":
                features = leaves[0][j][key].keys()
                for k in range(len(features)):
                    lb = leaves[0][j][key][k][0]
                    ub = leaves[0][j][key][k][1]
                    # there is chance that don't have lb and ub at this step
                    if lb is not None and ub is not None:
                        assert lb <= ub
    # test for thresholds
    # assert whether each feature has threshold
    assert len(thresholds[0].keys()) == n_inputs
    # assert the number of thresholds
    thresholds_count = 0
    for k in range(len(thresholds[0].keys())):
        for _ in range(len(thresholds[0][k].keys())):
            thresholds_count += 1
    assert thresholds_count == len(splits[0].keys())


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_raise_exception_if_wrong_model_instance():
    regr = linear_model_tree(X=X, y=Y)
    wrong_summary_dict = regr.summary()
    del wrong_summary_dict[1]
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    with pytest.raises(
        Exception,
        match=(
            "Input dict must be the summary of the linear-tree model"
            " e.g. dict = model.summary()"
        ),
    ):
        LinearTreeDefinition(
            regr.summary(only_leaves=True), scaled_input_bounds=input_bounds
        )
    with pytest.raises(
        Exception, match="Model entry must be dict or linear-tree instance"
    ):
        LinearTreeDefinition((0, 0), scaled_input_bounds=input_bounds)
    with pytest.raises(
        Exception,
        match=(
            "Input dict must be the summary of the linear-tree model"
            " e.g. dict = model.summary()"
        ),
    ):
        LinearTreeDefinition(wrong_summary_dict, scaled_input_bounds=input_bounds)


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_children_node_finders():
    # Train a linear model decision tree
    X = np.linspace(-4, 4).reshape((-1, 1))
    Y = np.sin(X)
    regr = linear_model_tree(X=X, y=Y)

    # Create a LinearTreeDefinition Object
    inBounds = {0: (-4, 4)}
    model_def = LinearTreeDefinition(regr, unscaled_input_bounds=inBounds)

    # Extract leaf and split information
    spts = model_def.splits
    lvs = model_def.leaves

    # Ensure that at the root node, the number of left leaves and the number of
    # right leaves sum to the total number of leaves in the tree
    num_left_leaves_at_root = len(spts[0][0]["left_leaves"])
    num_right_leaves_at_root = len(spts[0][0]["right_leaves"])
    total_leaves = len(lvs[0])

    assert num_left_leaves_at_root + num_right_leaves_at_root == total_leaves


@pytest.mark.skipif(not lineartree_available, reason="Need Linear-Tree Package")
def test_raise_exception_for_wrong_transformation():
    regr = linear_model_tree(X=X, y=Y)
    input_bounds = {0: (min(X[:, 0]), max(X[:, 0])), 1: (min(X[:, 1]), max(X[:, 1]))}
    model_def = LinearTreeDefinition(regr, unscaled_input_bounds=input_bounds)
    with pytest.raises(
        Exception,
        match="Supported transformations are: bigm, mbigm, hull, and custom",
    ):
        LinearTreeGDPFormulation(model_def, transformation="hello")
