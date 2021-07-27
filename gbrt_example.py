import numpy as np
import random
from onnxconverter_common.data_types import FloatTensorType, Int64TensorType
from onnxmltools.convert import convert_lightgbm

### requirements:
# lightgbm, gurobipy

# fix random seed for reproducible results
random.seed(10)

### we define a simple black-box function to generate data
class SimpleCat:
    def __init__(self):
        # self.cat_idx = [3]
        self.cat_idx = []
        self.name = 'simple_cat'

    def get_bounds(self):
        temp_bounds = [(-2.0, 2.0) for _ in range(3)]

        # lightgbm requires categorical variables as integers
        # in this case 0 is not meant to be 'smaller' than one
        # but the integers are rather keys to individual categories
        temp_bounds.append([0, 1])
        return temp_bounds

    def __call__(self, X):
        cat = X[-1]
        X = np.asarray_chkfinite(X[:-1])
        X0 = X[:-1]
        X1 = X[1:]

        add1 = X0[0]
        add2 = X1[0]

        if cat == 0:
            return 6 * (add1 + add2)
        elif cat == 1:
            return (add1 + add2) ** 2
        return add1 + add2


### generate example dataset
bb_func = SimpleCat()
bb_bnds = bb_func.get_bounds()

# generate samples for initial dataset

def generate_samples(num_samples, bb_bnds):
    data = {
        'X': [],
        'y': []
    }

    for _ in range(num_samples):
        sample = []

        # iterate through all dimension bounds
        for idx, var_bnds in enumerate(bb_bnds):

            if idx not in bb_func.cat_idx:
                # pick uniformly random number between continuous bounds
                # for continuous (conti) variables
                val = random.uniform(var_bnds[0], var_bnds[1])
            else:
                # pick random integer and map it to string for categorical (cat) variables
                cat_int = random.randrange(0, len(var_bnds))
                val = cat_int

            # populate the sample
            sample.append(val)

        data['X'].append(sample)
        data['y'].append(bb_func(sample))
    return data

data = generate_samples(100, bb_bnds)

### train tree model
import lightgbm as lgb

FIXED_PARAMS = {'objective': 'regression',
                'metric': 'rmse',
                'boosting': 'gbdt',
                'num_boost_round': 20,
                'max_depth': 3,
                'min_data_in_leaf': 2,
                'min_data_per_group': 2,
                'random_state': 100,
                'verbose': -1}

if bb_func.cat_idx:
    train_data = lgb.Dataset(data['X'], label=data['y'],
                             categorical_feature=bb_func.cat_idx,
                             free_raw_data=False,
                             params={'verbose': -1})

    model = lgb.train(FIXED_PARAMS, train_data,
                      categorical_feature=bb_func.cat_idx,
                      verbose_eval=False)
else:
    train_data = lgb.Dataset(data['X'], label=data['y'],
                             params={'verbose': -1})

    model = lgb.train(FIXED_PARAMS, train_data,
                      verbose_eval=False)

### translate tree model to flexible python representation

# dump_model creates a dictionary that has all the
# information needed to represent the model
tree_dict = model.dump_model()
initial_types = [('float_input', FloatTensorType([None, model.num_feature()]))]
onnx = convert_lightgbm(model, initial_types=initial_types)
with open('tree_dict.onnx', 'wb') as f:
    f.write(onnx.SerializeToString())

# remove comments to print the model to a file so you can have a look
import json
with open('tree_dict.json', 'w') as fp:
    json.dump(tree_dict, fp, indent=4)

# we re-order the dict so it works with the gbm model representation
from gbrt_utils import order_tree_model_dict
ordered_tree_dict = \
    order_tree_model_dict(
        tree_dict,
        cat_column=bb_func.cat_idx
    )

# init python representation, we define a dict of gbm_models to be able to include more tree_models
# into the same optimization model, e.g. just add more keys with gbm_models to the dict
from gbrt_utils import GbmModel
gbm_model_dict = {'first': GbmModel(ordered_tree_dict)}

### build gurobi model core
import gurobipy as gp
from gurobipy import GRB, quicksum
opt_model = gp.Model()

opt_model.Params.LogToConsole = 0

# we define two sets of variables for conti and cat vars
opt_model._n_feat = len(bb_bnds)
opt_model._cont_var_dict = {}
opt_model._cat_var_dict = {}

for idx, var_bnds in enumerate(bb_bnds):
    if idx not in bb_func.cat_idx:
        # define continuous vars
        opt_model._cont_var_dict[idx] = \
            opt_model.addVar(lb=var_bnds[0],
                         ub=var_bnds[1],
                         name=f"var_{idx}",
                         vtype='C')
    else:
        # define categorical vars
        opt_model._cat_var_dict[idx] = {}
        # we need a binary variable for every category in every categorical features
        for cat in var_bnds:
            opt_model._cat_var_dict[idx][cat] = \
                opt_model.addVar(name=f"var_{idx}_{cat}",
                             vtype=GRB.BINARY)

### we add the tree_model representation to the model

## add parameters to model
opt_model._gbm_models = gbm_model_dict

opt_model._gbm_set = set(gbm_model_dict.keys())
opt_model._num_trees = lambda label: \
    gbm_model_dict[label].n_trees

opt_model._leaves = lambda label, tree: \
    tuple(gbm_model_dict[label].get_leaf_encodings(tree))

opt_model._leaf_weight = lambda label, tree, leaf: \
    gbm_model_dict[label].get_leaf_weight(tree, leaf)

vbs = [v.get_var_break_points() for v in gbm_model_dict.values()]

all_breakpoints = {}
for i in range(opt_model._n_feat):
    if i in bb_func.cat_idx:
        continue
    else:
        s = set()
        for vb in vbs:
            try:
                s = s.union(set(vb[i]))
            except KeyError:
                pass
        if s:
            all_breakpoints[i] = sorted(s)

opt_model._breakpoint_index = list(all_breakpoints.keys())

opt_model._breakpoints = lambda i: all_breakpoints[i]

opt_model._leaf_vars = lambda label, tree, leaf: \
    tuple(i
          for i in gbm_model_dict[label].get_participating_variables(
        tree, leaf))
opt_model.update()

## add variables to model
from gbrt_utils import leaf_index, misic_interval_index
opt_model._z_l = opt_model.addVars(
    leaf_index(opt_model),
    lb=0,
    ub=GRB.INFINITY,
    name="z_l", vtype='C'
)

yv = opt_model._y = opt_model.addVars(
    misic_interval_index(opt_model),
    name="y",
    vtype=GRB.BINARY
)
opt_model.update()

## add constraints to model, for reference: https://arxiv.org/pdf/1705.10883.pdf
from gbrt_utils import tree_index, misic_split_index

# Equ. (2b)
def single_leaf_rule(model_, label, tree):
    z_l, leaves = model_._z_l, model_._leaves
    return (
            quicksum(z_l[label, tree, leaf] for leaf in leaves(label, tree))
            == 1)


cons = opt_model.addConstrs(
    (single_leaf_rule(opt_model, label, tree)
     for (label, tree) in tree_index(opt_model)),
    name="single_leaf"
)

# Equ. (2c)
def left_split_r(model_, label, tree, split_enc):
    gbt = model_._gbm_models[label]
    split_var, split_val = gbt.get_branch_partition_pair(
        tree,
        split_enc
    )
    y_var = split_var

    if not isinstance(split_val, list):
        # for conti vars
        y_val = model_._breakpoints(y_var).index(split_val)
        return \
            quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_left_leaves(tree, split_enc)
            ) <= \
            model_._y[y_var, y_val]
    else:
        # for cat vars
        return \
            quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_left_leaves(tree, split_enc)
            ) <= \
            quicksum(
                model_._cat_var_dict[split_var][cat]
                for cat in split_val
            )

opt_model.addConstrs(
    (left_split_r(opt_model, label, tree, encoding)
     for (label, tree, encoding) in misic_split_index(opt_model)),
    name="left_split"
)


# Equ. (2d)
def right_split_r(model_, label, tree, split_enc):
    gbt = model_._gbm_models[label]
    split_var, split_val = gbt.get_branch_partition_pair(
        tree,
        split_enc
    )
    y_var = split_var
    if not isinstance(split_val, list):
        # for conti vars
        y_val = model_._breakpoints(y_var).index(split_val)
        return \
            quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_right_leaves(tree, split_enc)
            ) <= \
            1 - model_._y[y_var, y_val]
    else:
        # for cat vars
        return \
            quicksum(
                model_._z_l[label, tree, leaf]
                for leaf in gbt.get_right_leaves(tree, split_enc)
            ) <= 1 - \
            quicksum(
                model_._cat_var_dict[split_var][cat]
                for cat in split_val
            )

opt_model.addConstrs(
    (right_split_r(opt_model, label, tree, encoding)
     for (label, tree, encoding) in misic_split_index(opt_model)),
    name="right_split"
)

# Equ. (2f)
def y_order_r(model_, i, j):
    if j == len(model_._breakpoints(i)):
        return Constraint.Skip
    return model_._y[i, j] <= model_._y[i, j + 1]

opt_model.addConstrs(
    (y_order_r(opt_model, var, j)
     for (var, j) in misic_interval_index(opt_model)
     if j != len(opt_model._breakpoints(var)) - 1),
    name="y_order"
)

# Equ. (2e)
def cat_sums(model_, i):
    return quicksum(
        model_._cat_var_dict[i][cat]
        for cat in model_._cat_var_dict[i].keys()
    ) == 1

opt_model.addConstrs(
    (cat_sums(opt_model, var)
     for var in bb_func.cat_idx),
    name="cat_sums"
)


# these constraints are from miten's paper: https://arxiv.org/pdf/1803.00952.pdf
# they link the discrete splits to the continuous variables

# Equ. (4a)
def var_lower_r(model_, i, j):
    lb = model_._cont_var_dict[i].lb
    j_bound = model_._breakpoints(i)[j]
    return model_._cont_var_dict[i] >= lb + (j_bound - lb) * (1 - model_._y[i, j])

opt_model.addConstrs(
    (var_lower_r(opt_model, var, j)
     for (var, j) in misic_interval_index(opt_model)),
    name="var_lower"
)

# Equ. (4b)
def var_upper_r(model_, i, j):
    ub = model_._cont_var_dict[i].ub
    j_bound = model_._breakpoints(i)[j]
    return model_._cont_var_dict[i] <= ub + (j_bound - ub) * (model_._y[i, j])

opt_model.addConstrs(
    (var_upper_r(opt_model, var, j)
     for (var, j) in misic_interval_index(opt_model)),
    name="var_upper"
)

# define objective as the mean value of the tree model
weighted_sum = quicksum(
    opt_model._leaf_weight(label, tree, leaf) * \
    opt_model._z_l[label, tree, leaf]
    for label, tree, leaf in leaf_index(opt_model)
)

opt_model.setObjective(weighted_sum, GRB.MINIMIZE)

opt_model.optimize()
print(f"\n* * * results:")
print(f"   best gurobi model obj: {opt_model.getObjective().getValue()}")

# check optimal results by randomly evaluating 5000 points of the tree model
test_data = generate_samples(5000, bb_bnds)
print(f"   best random point obj: {min(model.predict(test_data['X']))}")



#opt_model.display()