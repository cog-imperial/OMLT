from typing import Any

import lineartree
import numpy as np


class LinearTreeDefinition:
    """Class to represent a linear tree model trained in the linear-tree package.

    Attributes:
        __model (linear-tree model) : Linear Tree Model trained in linear-tree
        __splits (dict) : Dict containing split node information
        __leaves (dict) : Dict containing leaf node information
        __thresholds (dict) : Dict containing splitting threshold information
        __scaling_object (scaling object) : Scaling object to ensure scaled
            data match units of broader optimization problem
        __scaled_input_bounds (dict): Dict containing scaled input bounds
        __unscaled_input_bounds (dict): Dict containing unscaled input bounds

    References:
        * linear-tree : https://github.com/cerlymarco/linear-tree
    """

    def __init__(
        self,
        lt_regressor,
        scaling_object=None,
        scaled_input_bounds=None,
        unscaled_input_bounds=None,
    ):
        """Initialize LinearTreeDefinition.

        Create a LinearTreeDefinition object and define attributes based on the
        trained linear model decision tree.

        Arguments:
            lt_regressor: A LinearTreeRegressor model that is trained by the
                linear-tree package

        Keyword Arguments:
            scaling_object: A scaling object to specify the scaling parameters
                for the linear model tree inputs and outputs. If None, then no
                scaling is performed. (default: {None})
            scaled_input_bounds: A dict that contains the bounds on the scaled
                variables (the direct inputs to the tree). If None, then the
                user must specify the bounds via the input_bounds argument.
                (default: {None})
            unscaled_input_bounds: A dict that contains the bounds on the
                variables (the direct inputs to the tree). If None, then the
                user must specify the scaled bounds via the scaled_input_bounds
                argument. (default: {None})

        Raises:
            Exception: Input bounds required. If unscaled_input_bounds and
                scaled_input_bounds is None, raise Exception.
        """
        self.__model = lt_regressor
        self.__scaling_object = scaling_object

        is_scaled = True
        # Process input bounds to insure scaled input bounds exist for formulations
        if scaled_input_bounds is None:
            if unscaled_input_bounds is not None and scaling_object is not None:
                lbs = scaling_object.get_scaled_input_expressions(
                    {k: t[0] for k, t in unscaled_input_bounds.items()}
                )
                ubs = scaling_object.get_scaled_input_expressions(
                    {k: t[1] for k, t in unscaled_input_bounds.items()}
                )

                scaled_input_bounds = {
                    k: (lbs[k], ubs[k]) for k in unscaled_input_bounds
                }

            # If unscaled input bounds provided and no scaler provided, scaled
            # input bounds = unscaled input bounds
            elif unscaled_input_bounds is not None and scaling_object is None:
                scaled_input_bounds = unscaled_input_bounds
                is_scaled = False
            elif unscaled_input_bounds is None:
                msg = "Input Bounds needed to represent linear trees as MIPs"
                raise ValueError(msg)
        elif scaling_object is not None:
            factors = scaling_object._OffsetScaling__x_factor
            offsets = scaling_object._OffsetScaling__x_offset
            unscaled_input_bounds = {}
            for key in scaled_input_bounds:
                scaled_lb = scaled_input_bounds[key][0]
                scaled_ub = scaled_input_bounds[key][1]
                unscaled_lb = scaled_lb * factors[key] + offsets[key]
                unscaled_ub = scaled_ub * factors[key] + offsets[key]
                unscaled_input_bounds[key] = (unscaled_lb, unscaled_ub)

        self.__unscaled_input_bounds = unscaled_input_bounds
        self.__scaled_input_bounds = scaled_input_bounds
        self.__is_scaled = is_scaled

        self.__splits, self.__leaves, self.__thresholds = _parse_tree_data(
            lt_regressor, scaled_input_bounds
        )

        self.__n_inputs = _find_n_inputs(self.__leaves)
        self.__n_outputs = 1

    @property
    def scaling_object(self):
        """Returns scaling object."""
        return self.__scaling_object

    @property
    def scaled_input_bounds(self):
        """Returns dict containing scaled input bounds."""
        return self.__scaled_input_bounds

    @property
    def unscaled_input_bounds(self):
        """Returns dict containing unscaled input bounds."""
        return self.__unscaled_input_bounds

    @property
    def is_scaled(self):
        """Returns bool indicating whether model is scaled."""
        return self.__is_scaled

    @property
    def splits(self):
        """Returns dict containing split information."""
        return self.__splits

    @property
    def leaves(self):
        """Returns dict containing leaf information."""
        return self.__leaves

    @property
    def thresholds(self):
        """Returns dict containing threshold information."""
        return self.__thresholds

    @property
    def n_inputs(self):
        """Returns number of inputs to the linear tree."""
        return self.__n_inputs

    @property
    def n_outputs(self):
        """Returns number of outputs to the linear tree."""
        return self.__n_outputs


def _find_all_children_splits(split, splits_dict):
    """Find all children splits.

    This helper function finds all multigeneration children splits for an
    argument split.

    Arguments:
        split: The split for which you are trying to find children splits
        splits_dict: A dictionary of all the splits in the tree

    Returns:
        A list containing the Node IDs of all children splits
    """
    all_splits = []

    # Check if the immediate left child of the argument split is also a split.
    # If so append to the list then use recursion to generate the remainder
    left_child = splits_dict[split]["children"][0]
    if left_child in splits_dict:
        all_splits.append(left_child)
        all_splits.extend(_find_all_children_splits(left_child, splits_dict))

    # Same as above but with right child
    right_child = splits_dict[split]["children"][1]
    if right_child in splits_dict:
        all_splits.append(right_child)
        all_splits.extend(_find_all_children_splits(right_child, splits_dict))

    return all_splits


def _find_all_children_leaves(split, splits_dict, leaves_dict):
    """Find all children leaves.

    This helper function finds all multigeneration children leaves for an
    argument split.

    Arguments:
        split: The split for which you are trying to find children leaves
        splits_dict: A dictionary of all the split info in the tree
        leaves_dict: A dictionary of all the leaf info in the tree

    Returns:
        A list containing all the Node IDs of all children leaves
    """
    # Find all the splits that are children of the relevant split
    all_splits = _find_all_children_splits(split, splits_dict)

    # Ensure the current split is included
    if split not in all_splits:
        all_splits.append(split)

    # For each leaf, check if the parents appear in the list of children
    # splits (all_splits). If so, it must be a leaf of the argument split

    return [leaf for leaf in leaves_dict if leaves_dict[leaf]["parent"] in all_splits]


def _find_n_inputs(leaves):
    """Find n inputs.

    Finds the number of inputs using the length of the slope vector in the
    first leaf

    Arguments:
        leaves: Dictionary of leaf information

    Returns:
        Number of inputs
    """
    tree_indices = np.array(list(leaves.keys()))
    leaf_indices = np.array(list(leaves[tree_indices[0]].keys()))
    tree_one = tree_indices[0]
    leaf_one = leaf_indices[0]
    return len(np.arange(0, len(leaves[tree_one][leaf_one]["slope"])))


def _reassign_none_bounds(leaves, input_bounds):
    """Reassign None bounds.

    This helper function reassigns bounds that are None to the bounds
    input by the user

    Arguments:
        leaves: The dictionary of leaf information. Attribute of the
            LinearTreeDefinition object
        input_bounds: The nested dictionary

    Returns:
        The modified leaves dict without any bounds that are listed as None
    """
    leaf_indices = np.array(list(leaves.keys()))
    leaf_one = leaf_indices[0]
    features = np.arange(0, len(leaves[leaf_one]["slope"]))

    for leaf in leaf_indices:
        for feat in features:
            if leaves[leaf]["bounds"][feat][0] is None:
                leaves[leaf]["bounds"][feat][0] = input_bounds[feat][0]
            if leaves[leaf]["bounds"][feat][1] is None:
                leaves[leaf]["bounds"][feat][1] = input_bounds[feat][1]

    return leaves


def _parse_tree_data(model, input_bounds):  # noqa: C901, PLR0915, PLR0912
    """Parse tree data.

    This function creates the data structures with the information required
    for creation of the variables, sets, and constraints in the pyomo
    reformulation of the linear model decision trees. Note that these data
    structures are attributes of the LinearTreeDefinition Class.

    Arguments:
        model: Trained linear-tree model or dic containing linear-tree model
            summary (e.g. dict = model.summary())
        input_bounds: The input bounds

    Returns:
        leaves - Dict containing the following information for each leaf:
            1) 'slope' - The slope of the fitted line at that leaf
            2) 'intercept' - The intercept of the line at that lead
            3) 'parent' - The parent split or node of that leaf
        splits - Dict containing the following information for each split:
            1) 'children' - The child nodes of that split
            2) 'col' - The variable(feature) to split on (beginning at 0)
            3) 'left_leaves' - All the leaves to the left of that split
            4) 'right_leaves' - All the leaves to the right of that split
            5) 'parent' - The parent node of the split. Node zero has no parent
            6) 'th' - The threshold of the split
            7) 'y_index' - Indices corresponding to Mistry et. al. y binary
                    variable
        vars_dict - Dict of tree inputs and their respective thresholds

    Raises:
            Exception: If input dict is not equal to model.summary()
            Exception: If input model is not a dict or linear-tree instance
    """
    # Create the initial leaves and splits dictionaries depending on the
    # instance of the model (can be either a LinearTreeRegressor or dict).
    # Include checks to ensure that the input dict is the model summary which
    # is obtained by calling the summary() method contained within the
    # linear-tree package (e.g. dict = model.summary())
    if isinstance(model, lineartree.lineartree.LinearTreeRegressor) is True:
        leaves = model.summary(only_leaves=True)
        splits = model.summary()
    elif isinstance(model, dict) is True:
        splits = model
        leaves = {}
        num_splits_in_model = 0
        count = 0
        # Checks to ensure that the input nested dictionary contains the
        # correct information
        for entry in model:
            if "children" not in model[entry]:
                leaves[entry] = model[entry]
            else:
                left_child = model[entry]["children"][0]
                right_child = model[entry]["children"][1]
                num_splits_in_model += 1
                if left_child not in model or right_child not in model:
                    count += 1
        if count > 0 or num_splits_in_model == 0:
            msg = (
                "Input dict must be the summary of the linear-tree model"
                " e.g. dict = model.summary()"
            )
            raise ValueError(msg)
    else:
        msg = "Model entry must be dict or linear-tree instance"
        raise TypeError(msg)

    # This loop adds keys for the slopes and intercept and removes the leaf
    # keys in the splits dictionary
    for leaf in leaves:
        del splits[leaf]
        leaves[leaf]["slope"] = list(leaves[leaf]["models"].coef_)
        leaves[leaf]["intercept"] = leaves[leaf]["models"].intercept_

    # This loop creates an parent node id entry for each node in the tree
    for split in splits:
        left_child = splits[split]["children"][0]
        right_child = splits[split]["children"][1]

        if left_child in splits:
            splits[left_child]["parent"] = split
        else:
            leaves[left_child]["parent"] = split

        if right_child in splits:
            splits[right_child]["parent"] = split
        else:
            leaves[right_child]["parent"] = split

    # This loop creates an entry for the all the leaves to the left and right
    # of a split
    for split in splits:
        left_child = splits[split]["children"][0]
        right_child = splits[split]["children"][1]

        if left_child in splits:
            splits[split]["left_leaves"] = _find_all_children_leaves(
                left_child, splits, leaves
            )
        else:
            splits[split]["left_leaves"] = [left_child]

        if right_child in splits:
            splits[split]["right_leaves"] = _find_all_children_leaves(
                right_child, splits, leaves
            )
        else:
            splits[split]["right_leaves"] = [right_child]

    # For each variable that appears in the tree, go through all the splits
    # and record its splitting threshold
    splitting_thresholds: dict[int, Any] = {}
    for split in splits:
        var = splits[split]["col"]
        splitting_thresholds[var] = {}
    for split in splits:
        var = splits[split]["col"]
        splitting_thresholds[var][split] = splits[split]["th"]

    # Make sure every nested dictionary in the splitting_thresholds dictionary
    # is sorted by value
    for var in splitting_thresholds:
        splitting_thresholds[var] = dict(
            sorted(splitting_thresholds[var].items(), key=lambda x: x[1])
        )

    # NOTE: Can eliminate if not implementing the Mistry et. al. formulations
    # Record the ordered indices of the binary variable y. The first index
    # is the splitting variable. The second index is its location in the
    # ordered dictionary of thresholds for that variable.
    for split in splits:
        var = splits[split]["col"]
        splits[split]["y_index"] = []
        splits[split]["y_index"].append(var)
        splits[split]["y_index"].append(list(splitting_thresholds[var]).index(split))

    # For each leaf, create an empty dictionary that will store the lower
    # and upper bounds of each feature.
    for leaf in leaves:
        leaves[leaf]["bounds"] = {}

    leaf_ids = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[leaf_ids[0]]["slope"]))

    # For each feature in each leaf, initialize lower and upper bounds to None
    for feat in features:
        for leaf in leaves:
            leaves[leaf]["bounds"][feat] = [None, None]

    # Finally, go through each split and assign it's threshold value as the
    # upper bound to all the leaves descending to the left of the split and
    # as the lower bound to all the leaves descending to the right.
    for split in splits:
        var = splits[split]["col"]
        for leaf in splits[split]["left_leaves"]:
            leaves[leaf]["bounds"][var][1] = splits[split]["th"]

        for leaf in splits[split]["right_leaves"]:
            leaves[leaf]["bounds"][var][0] = splits[split]["th"]

    leaves = _reassign_none_bounds(leaves, input_bounds)

    # We use the same formulations developed for gradient boosted linear trees
    # so we nest the leaves, splits, and thresholds attributes in a "one-tree"
    # tree.
    leaves = {0: leaves}
    splits = {0: splits}
    splitting_thresholds = {0: splitting_thresholds}

    return splits, leaves, splitting_thresholds
