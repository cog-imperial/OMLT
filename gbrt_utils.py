import collections as coll
import numpy as np

# EVERYTHING RELATED TO SORTING THE LGBM DICT

def order_tree_model_dict(tree_model_dict, cat_column=None):
    """
    Sorts a tree model dict provided by LightGBM from .json file to make it
    suitable for use in ENTMOOT.

    Key name `tree_info` is specific to LightGBM and needs to be changed for
    other tree training libraries that support model dump to .json format.

    Parameters
    ----------
    tree_model_dict : dict,
        Includes all information on trees trained in LightGBM

    Returns
    -------
    ordered_tree_list : list,
        Ordered tree list compatible with `GbmModel`
    """
    tree_list = tree_model_dict['tree_info']
    ordered_tree_list = order_tree_list(tree_list, cat_column=cat_column)
    return ordered_tree_list


def order_tree_list(tree_list, cat_column=None):
    """
    Sorts a tree list provided by LightGBM from .json file to make it
    suitable for use in ENTMOOT.

    Parameters
    ----------
    tree_list : list,
        Unsorted list of trees from LightGBM

    Returns
    -------
    ordered_tree_list : list,
        Ordered tree list compatible with `GbmModel`
    """
    ordered_tree_list = []
    for tree in tree_list:
        ordered_node_list = order_node_list(tree, cat_column=cat_column)
        if ordered_node_list:
            ordered_tree_list.append(ordered_node_list)
    return ordered_tree_list


def order_node_list(tree, cat_column=None):
    """
    Sorts a list of node dict from a LightGBM instance. Key `tree_structure` is
    specific for LightGBM.

    Parameters
    ----------
    tree : list,
        Unsorted list of node dicts

    Returns
    -------
    ordered_node_list : list,
        Ordered list of node dicts compatible with `GbmModel`
    """
    node = []
    node.append(tree['tree_structure'])
    ordered_node_list = []
    add_next_nodes(ordered_node_list, node, cat_column=cat_column)
    return ordered_node_list


def add_next_nodes(ordered_node_list, node, cat_column=None):
    """
    Processes LightGBM node and adds sorted node to `ordered_node_list`.

    Parameters
    ----------
    ordered_node_list : list,
        List of nodes to which sorted node is added

    node : list,
        List of unsorted LightGBM nodes for one tree

    Returns
    -------
    -
    """
    if node:
        new_node = {}

        try:
            new_node['split_var'] = node[-1]['split_feature']
            if not new_node['split_var'] in cat_column:
                # read numerical variables
                temp_node_val = node[-1]['threshold']
                new_node['split_code_pred'] = temp_node_val  # round(temp_node_val,5)
            else:
                # read categorical variables
                cat_set = node[-1]['threshold'].split("||")
                temp_node_val = [int(cat) for cat in cat_set]
                new_node['split_code_pred'] = temp_node_val

        except KeyError:
            new_node['split_var'] = -1
            temp_node_val = node[-1]['leaf_value']
            new_node['split_code_pred'] = temp_node_val  # round(temp_node_val,5)

        ordered_node_list.append(new_node)

        temp_node = node.pop(-1)
        try:
            node.append(temp_node['right_child'])
        except KeyError:
            pass

        try:
            node.append(temp_node['left_child'])
        except KeyError:
            pass

        add_next_nodes(ordered_node_list, node, cat_column=cat_column)

### EVERTHING GBM_MODEL RELATED

class GbmModel:
    """Define a gbm model.

    A `GbmModel` is a object-oriented class to enocode tree-based models from
    different libraries. Supported tree training libraries are: LightGBM.

    Use this class as an interface to read in model data from tree model
    training libraries and translate it to the ENTMOOT native structure.

    Parameters
    ----------
    tree_list : list
        Ordered tree list that obeys ENTMOOT input specifications

    Attributes
    ----------
    -

    """
    def __init__(self, tree_list):
        self.load_ordered_tree_dict(tree_list)

    def _build_tree(self, tree):
        """Define `GbmNode` based on single tree.

        Parameters
        ----------
        trees : list of dicts,
            Specifies list of node dicts
        n_trees : number of trees in `tree_list`

        Returns
        -------
        gbm_node : GbmNode
        """
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        return GbmNode(
            split_var=split_var,
            split_code_pred=split_code_pred,
            tree=tree
        )

    def load_ordered_tree_dict(self, tree_list):
        """Define attributes used in `GbmModel`.

        Parameters
        ----------
        tree_list : list of trees,
            Specifies list of trees which contain list of node dicts

        Returns
        -------
        -
        """
        self.n_trees = len(tree_list)

        # assign leaf ids
        leaf_id = 0
        for tree_id, tree in enumerate(tree_list):
            for node in tree:
                if node['split_var'] == -1:
                    node['leaf_id'] = leaf_id
                    leaf_id += 1

        self.n_leaves = leaf_id

        # build trees
        self.trees = [self._build_tree(tree) for tree in tree_list]

    def get_leaf_encodings(self, tree):
        yield from self.trees[tree].get_leaf_encodings()

    def get_branch_encodings(self, tree):
        yield from self.trees[tree].get_branch_encodings()

    def get_leaf_weight(self, tree, encoding):
        return self.trees[tree].get_leaf_weight(encoding)

    def get_leaf_weights(self, tree):
        return self.trees[tree].get_leaf_weights()

    def get_branch_partition_pair(self, tree, encoding):
        return self.trees[tree].get_branch_partition_pair(encoding)

    def get_left_leaves(self, tree, encoding):
        yield from (
            encoding + s
            for s in self.trees[tree].get_left_leaves(encoding)
        )

    def get_right_leaves(self, tree, encoding):
        yield from (
            encoding + s
            for s in self.trees[tree].get_right_leaves(encoding)
        )

    def get_branch_partition_pairs(self, tree, leaf_encoding):
        yield from self.trees[tree].get_branch_partition_pairs(leaf_encoding)

    def get_participating_variables(self, tree, leaf):
        return set(self.trees[tree].get_participating_variables(leaf))

    def get_all_participating_variables(self, tree):
        return set(self.trees[tree].get_all_participating_variables())

    def get_var_lower(self, tree, leaf, var, lower):
        return self.trees[tree].get_var_lower(leaf, var, lower)

    def get_var_upper(self, tree, leaf, var, upper):
        return self.trees[tree].get_var_upper(leaf, var, upper)

    def get_var_interval(self, tree, leaf, var, var_interval):
        return self.trees[tree].get_var_interval(leaf, var, var_interval)

    def get_var_break_points(self):
        var_breakpoints = {}
        for tree in self.trees:
            for var, breakpoint in tree.get_all_partition_pairs():
                try:
                    if isinstance(breakpoint,list):
                        var_breakpoints[var].append(breakpoint)
                    else:
                        var_breakpoints[var].add(breakpoint)
                except KeyError:
                    if isinstance(breakpoint,list):
                        var_breakpoints[var] = [breakpoint]
                    else:
                        var_breakpoints[var] = set([breakpoint])

        for k in var_breakpoints.keys():
            if isinstance(var_breakpoints[k], set):
                var_breakpoints[k] = sorted(var_breakpoints[k])

        return var_breakpoints

    def get_leaf_count(self):
        resulting_counter = sum(
            (tree.get_leaf_count() for tree in self.trees),
            coll.Counter()
        )
        del resulting_counter['leaf']
        return resulting_counter

class GbmType:
    def _populate_active_splits(self, active_splits, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                self.left._populate_active_splits(active_splits, X)
            else:
                self.right._populate_active_splits(active_splits, X)
        else:
            if self.split_var != -1:
                if not self.split_var in active_splits.keys():
                    active_splits[self.split_var] = []

                if X[self.split_var] <= self.split_code_pred:
                    self.left._populate_active_splits(active_splits, X)
                    active_splits[self.split_var].append(self.split_code_pred)
                else:
                    self.right._populate_active_splits(active_splits, X)

    def _populate_active_leaf_encodings(self, active_leaf, X):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                active_leaf.append('0')
                self.left._populate_active_leaf_encodings(active_leaf, X)
            else:
                active_leaf.append('1')
                self.right._populate_active_leaf_encodings(active_leaf, X)
        else:
            if self.split_var != -1:
                if X[self.split_var] <= self.split_code_pred:
                    active_leaf.append('0')
                    self.left._populate_active_leaf_encodings(active_leaf, X)
                else:
                    active_leaf.append('1')
                    self.right._populate_active_leaf_encodings(active_leaf, X)

    def _get_active_leaf_id_vec(self, X, leaf_vec):
        if isinstance(self.split_code_pred, list):
            if X[self.split_var] in self.split_code_pred:
                return self.left._get_active_leaf_id_vec(X, leaf_vec)
            else:
                return self.right._get_active_leaf_id_vec(X, leaf_vec)
        else:
            if X[self.split_var] <= self.split_code_pred:
                return self.left._get_active_leaf_id_vec(X, leaf_vec)
            else:
                return self.right._get_active_leaf_id_vec(X, leaf_vec)

class GbmNode(GbmType):
    """Defines a gbm node which can be a split or leaf.

    Initializing `GbmNode` triggers recursive initialization of all nodes that
    appear in a tree

    Parameters
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree

    Attributes
    ----------
    split_var : int
        Index of split variable used
    split_code_pred : list
        Value that defines split
    tree : list
        List of node dicts that define the tree
    """
    def __init__(
        self,
        split_var,
        split_code_pred,
        tree
    ):
        self.split_var = split_var
        self.split_code_pred = split_code_pred

        # check if tree is empty
        if not tree:
            print("EmptyTreeModelError: LightGBM was not able to train a "
            "tree model based on your parameter specifications. This can "
            "usually be fixed by increasing the number of `n_initial_points` or "
            "reducing `min_child_samples` via `base_estimator_kwargs` (default "
            "is 2). Alternatively, change `acq_optimizer='sampling'`.")
            import sys
            sys.exit(1)

        # read left node
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        # split_var value of -1 refers to leaf node
        if split_var == -1:
            self.left = LeafNode(
                split_code_pred = split_code_pred,
                leaf_id = node['leaf_id']
            )
        else:
            self.left = GbmNode(
                split_var = split_var,
                split_code_pred = split_code_pred,
                tree = tree
            )

        # read right node
        node = tree.pop(0)
        split_var = node['split_var']
        split_code_pred = node['split_code_pred']

        if split_var == -1:
            self.right = LeafNode(
                split_code_pred=split_code_pred,
                leaf_id=node['leaf_id']
            )
        else:
            self.right = GbmNode(
                split_var=split_var,
                split_code_pred=split_code_pred,
                tree=tree
            )

    def __repr__(self):
        return ', '.join([
            str(x)
            for x in [
                self.split_var,
                self.split_code_pred]
        ])

    def _get_next_node(self, direction):
        return self.right if int(direction) else self.left

    def get_leaf_encodings(self, current_string=''):
        yield from self.left.get_leaf_encodings(current_string+'0')
        yield from self.right.get_leaf_encodings(current_string+'1')

    def get_branch_encodings(self, current_string=''):
        yield current_string
        yield from self.left.get_branch_encodings(current_string+'0')
        yield from self.right.get_branch_encodings(current_string+'1')

    def get_leaf_weight(self, encoding):
        next_node = self.right if int(encoding[0]) else self.left
        return next_node.get_leaf_weight(encoding[1:])

    def get_leaf_weights(self):
        yield from self.left.get_leaf_weights()
        yield from self.right.get_leaf_weights()

    def get_branch_partition_pair(self, encoding):
        if not encoding:
            return self.split_var, self.split_code_pred
        else:
            next_node = self._get_next_node(encoding[0])
            return next_node.get_branch_partition_pair(encoding[1:])

    def get_left_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_left_leaves(encoding[1:])
        else:
            yield from self.left.get_leaf_encodings('0')

    def get_right_leaves(self, encoding):
        if encoding:
            next_node = self._get_next_node(encoding[0])
            yield from next_node.get_right_leaves(encoding[1:])
        else:
            yield from self.right.get_leaf_encodings('1')

    def get_branch_partition_pairs(self, encoding):
        yield (self.split_var, self.split_code_pred)
        try:
            next_node = self.right if int(encoding[0]) else self.left
            next_gen = next_node.get_branch_partition_pairs(encoding[1:])
        except IndexError:
            next_gen = []
        yield from next_gen

    def get_participating_variables(self, encoding):
        yield self.split_var
        next_node = self.right if int(encoding[0]) else self.left
        yield from next_node.get_participating_variables(encoding[1:])

    def get_all_participating_variables(self):
        yield self.split_var
        yield from self.left.get_all_participating_variables()
        yield from self.right.get_all_participating_variables()

    def get_all_partition_pairs(self):
        yield (self.split_var, self.split_code_pred)
        yield from self.left.get_all_partition_pairs()
        yield from self.right.get_all_partition_pairs()

    def get_var_lower(self, encoding, var, lower):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 1:
                    assert lower <= self.split_code_pred
                    lower = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_lower(encoding[1:], var, lower)
        else:
            return lower

    def get_var_upper(self, encoding, var, upper):
        if encoding:
            if self.split_var == var:
                if int(encoding[0]) == 0:
                    assert upper >= self.split_code_pred
                    upper = self.split_code_pred
            next_node = self.right if int(encoding[0]) else self.left
            return next_node.get_var_upper(encoding[1:], var, upper)
        else:
            return upper

    def get_var_interval(self, encoding, var, var_interval):
        if int(encoding[0]):
            next_node = self.right
            if self.split_var == var:
                var_interval = (self.split_code_pred, var_interval[1])
        else:
            next_node = self.left
            if self.split_var == var:
                var_interval = (var_interval[0], self.split_code_pred)
        return next_node.get_var_interval(encoding[1:], var, var_interval)

    def get_leaf_count(self):
        left_count = self.left.get_leaf_count()
        right_count = self.right.get_leaf_count()
        joint_count = left_count+right_count

        left_key = (self.split_var, self.split_code_pred, 0)
        right_key = (self.split_var, self.split_code_pred, 1)
        joint_count[left_key] += left_count['leaf']
        joint_count[right_key] += right_count['leaf']
        return joint_count

class LeafNode(GbmType):
    """Defines a child class of `GbmType`. Leaf nodes have `split_var = -1` and
    `split_code_pred` as leaf value defined by training."""
    def __init__(self,
                 split_code_pred,
                 leaf_id):
        self.split_var = -1
        self.split_code_pred = split_code_pred
        self.leaf_id = leaf_id

    def __repr__(self):
        return ', '.join([
            str(x)
            for x in [
                'Leaf',
                self.split_code_pred]
        ])

    def switch_to_maximisation(self):
        """Changes the sign of tree model prediction by changing signs of
        leaf values."""
        self.split_code_pred = -1*self.split_code_pred


    def get_leaf_encodings(self, current_string=''):
        yield current_string

    def get_branch_encodings(self, current_string=''):
        yield from []

    def get_leaf_weight(self, encoding):
        assert not encoding
        return self.split_code_pred

    def get_leaf_weights(self):
        yield self.split_code_pred

    def get_branch_partition_pair(self, encoding):
        raise Exception('Should not get here.')

    def get_left_leaves(self, encoding):
        raise Exception('Should not get here.')

    def get_right_leaves(self, encoding):
        raise Exception('Should not get here.')

    def get_branch_partition_pairs(self, encoding):
        assert not encoding
        yield from []

    def get_participating_variables(self, encoding):
        assert not encoding
        yield from []

    def get_all_participating_variables(self):
        yield from []

    def get_all_partition_pairs(self):
        yield from []

    def get_var_lower(self, encoding, var, lower):
        assert not encoding
        return lower

    def get_var_upper(self, encoding, var, upper):
        assert not encoding
        return upper

    def get_var_interval(self, encoding, var, var_interval):
        assert not encoding
        return var_interval

    def get_leaf_count(self):
        return coll.Counter({'leaf': 1})

### EVERYTHING RELATED TO BUILDING THE TREE MODEL

def label_leaf_index(model, label):
    for tree in range(model._num_trees(label)):
        for leaf in model._leaves(label, tree):
            yield (tree, leaf)


def tree_index(model):
    for label in model._gbm_set:
        for tree in range(model._num_trees(label)):
            yield (label, tree)


tree_index.dimen = 2


def leaf_index(model):
    for label, tree in tree_index(model):
        for leaf in model._leaves(label, tree):
            yield (label, tree, leaf)


leaf_index.dimen = 3


def misic_interval_index(model):
    for var in model._breakpoint_index:
        for j in range(len(model._breakpoints(var))):
            yield (var, j)


misic_interval_index.dimen = 2


def misic_split_index(model):
    gbm_models = model._gbm_models
    for label, tree in tree_index(model):
        for encoding in gbm_models[label].get_branch_encodings(tree):
            yield (label, tree, encoding)


misic_split_index.dimen = 3


def alt_interval_index(model):
    for var in model.breakpoint_index:
        for j in range(1, len(model.breakpoints[var]) + 1):
            yield (var, j)


alt_interval_index.dimen = 2