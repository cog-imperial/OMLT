import numpy as np
import pyomo.environ as pe
from pyomo.gdp import Disjunct

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs


class LinearTreeGDPFormulation(_PyomoFormulation):
    r"""Linear Tree GDP Formulation.

    Class to add a Linear Tree GDP formulation to OmltBlock. We use Pyomo.GDP
    to create the disjuncts and disjunctions and then apply a transformation
    to convert to a mixed-integer programming representation.

    .. math::
        \begin{align*}
            & \underset{\ell \in L}{\bigvee} \left[ \begin{gathered}
            Z_{\ell} \\
            \underline{x}_{\ell} \leq x \leq \overline{x}_{\ell}  \\
            d = a_{\ell}^T x + b_{\ell} \end{gathered} \right] \\
            & \texttt{exactly_one} \{ Z_{\ell} : \ell \in L \} \\
            & x^L \leq x \leq x^U \\
            & x \in \mathbb{R}^n \\
            & Z_{\ell} \in \{ \texttt{True, False} \} \quad \forall \ \ell \in L
        \end{align*}

    Additional nomenclature for this formulation is as follows:

    .. math::
        \begin{align*}
        Z_{\ell} &:= \text{Boolean variable indicating which leaf is selected} \\
        \overline{x}_{\ell} &:= \text{Vector of upper bounds for leaf } \ell \in L \\
        \underline{x}_{\ell} &:= \text{Vector of lower bounds for leaf } \ell \in L \\
        x^U &:= \text{Vector of global upper bounds} \\
        x^L &:= \text{Vector of global lower bounds} \\
        \end{align*}


    Attributes:
        Inherited from _PyomoFormulation Class
        model_definition : LinearTreeDefinition object
        transformation : choose which transformation to apply. The supported
            transformations are bigm, mbigm, hull, and custom.

    References:
        * Ammari et al. (2023) Linear Model Decision Trees as Surrogates in
          Optimization of Engineering Applications. Computers & Chemical Engineering
        * Chen et al. (2022) Pyomo.GDP: An ecosystem for logic based modeling and
          optimization development. Optimization and Engineering, 23:607-642
    """

    def __init__(self, lt_definition, transformation="bigm"):
        """Create a LinearTreeGDPFormulation object.

        Arguments:
            lt_definition: LinearTreeDefintion Object

        Keyword Arguments:
            transformation: choose which Pyomo.GDP formulation to apply.
                Supported transformations are bigm, hull, mbigm, and custom
                (default: {'bigm'})

        Raises:
            Exception: If transformation not in supported transformations
        """
        super().__init__()
        self.model_definition = lt_definition
        self.transformation = transformation

        # Ensure that the GDP transformation given is supported
        supported_transformations = ["bigm", "hull", "mbigm", "custom"]
        if transformation not in supported_transformations:
            msg = "Supported transformations are: bigm, mbigm, hull, and custom"
            raise NotImplementedError(msg)

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition.n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition.n_outputs))

    def _build_formulation(self):
        """Build formulation.

        This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition.scaling_object,
            self.model_definition.scaled_input_bounds,
        )

        _add_gdp_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs,
            transformation=self.transformation,
        )


class LinearTreeHybridBigMFormulation(_PyomoFormulation):
    r"""Class to add a Linear Tree Hybrid Big-M formulation to OmltBlock.

    .. math::
        \begin{align*}
        & d = \sum_{\ell \in L} (a_{\ell}^T x + b_{\ell})z_{\ell} \\
        & x_i \leq \sum_{\ell \in L} \overline{x}_{i,\ell} z_{\ell} &&
            \forall i \in [n] \\
        & x_i \geq \sum_{\ell \in L} \underline{x}_{i,\ell} z_{\ell} &&
            \forall i \in [n] \\
        & \sum_{\ell \in L} z_{\ell} = 1
        \end{align*}

    Where the following additional nomenclature is defined:

    .. math::
        \begin{align*}
        [n] &:= \text{the integer set of variables that the tree splits on
          (e.g. [n] = {1, 2, ... , n})} \\
        \overline{x}_{\ell} &:= \text{Vector of upper bounds for leaf } \ell \in L \\
        \underline{x}_{\ell} &:= \text{Vector of lower bounds for leaf } \ell \in L \\
        \end{align*}

    Attributes:
        Inherited from _PyomoFormulation Class
        model_definition : LinearTreeDefinition object

    """

    def __init__(self, lt_definition):
        """Create a LinearTreeHybridBigMFormulation object.

        Arguments:
            lt_definition: LinearTreeDefinition Object
        """
        super().__init__()
        self.model_definition = lt_definition

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition.n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition.n_outputs))

    def _build_formulation(self):
        """Build formulation.

        This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition.scaling_object,
            self.model_definition.scaled_input_bounds,
        )

        _add_hybrid_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs,
        )


def _build_output_bounds(model_def, input_bounds):
    """Build output bounds.

    This helper function develops bounds of the output variable based on the
    values of the input_bounds and the signs of the slope

    Arguments:
        model_def: Model definition
        input_bounds: Dict of input bounds

    Returns:
        List that contains the conservative lower and upper bounds of the
        output variable
    """
    leaves = model_def.leaves
    n_inputs = model_def.n_inputs
    tree_ids = np.array(list(leaves.keys()))
    features = np.arange(0, n_inputs)

    # Initialize bounds and variables
    bounds = [0, 0]
    upper_bound = 0
    lower_bound = 0
    for tree in tree_ids:
        for leaf in leaves[tree]:
            slopes = leaves[tree][leaf]["slope"]
            intercept = leaves[tree][leaf]["intercept"]
            for k in features:
                if slopes[k] <= 0:
                    upper_bound += slopes[k] * input_bounds[k][0] + intercept
                    lower_bound += slopes[k] * input_bounds[k][1] + intercept
                else:
                    upper_bound += slopes[k] * input_bounds[k][1] + intercept
                    lower_bound += slopes[k] * input_bounds[k][0] + intercept
                bounds[1] = max(bounds[1], upper_bound)
                bounds[0] = min(bounds[0], lower_bound)
            upper_bound = 0
            lower_bound = 0

    return bounds


def _add_gdp_formulation_to_block(
    block, model_definition, input_vars, output_vars, transformation
):
    """This function adds the GDP representation to the OmltBlock using Pyomo.GDP.

    Arguments:
        block: OmltBlock
        model_definition: LinearTreeDefinition Object
        input_vars: input variables to the linear tree model
        output_vars: output variable of the linear tree model
        transformation: Transformation to apply

    """
    leaves = model_definition.leaves
    input_bounds = model_definition.scaled_input_bounds
    n_inputs = model_definition.n_inputs

    # The set of leaves and the set of features
    tree_ids = list(leaves.keys())
    t_l = [(tree, leaf) for tree in tree_ids for leaf in leaves[tree]]
    features = np.arange(0, n_inputs)

    # Use the input_bounds and the linear models in the leaves to calculate
    # the lower and upper bounds on the output variable. Required for Pyomo.GDP
    output_bounds = _build_output_bounds(model_definition, input_bounds)

    # Ouptuts are automatically scaled based on whether inputs are scaled
    block.outputs.setub(output_bounds[1])
    block.outputs.setlb(output_bounds[0])
    block.scaled_outputs.setub(output_bounds[1])
    block.scaled_outputs.setlb(output_bounds[0])

    block.intermediate_output = pe.Var(
        tree_ids, bounds=(output_bounds[0], output_bounds[1])
    )

    # Create a disjunct for each leaf containing the bound constraints
    # and the linear model expression.
    def disjuncts_rule(dsj, tree, leaf):
        def lb_rule(dsj, feat):
            return input_vars[feat] >= leaves[tree][leaf]["bounds"][feat][0]

        dsj.lb_constraint = pe.Constraint(features, rule=lb_rule)

        def ub_rule(dsj, feat):
            return input_vars[feat] <= leaves[tree][leaf]["bounds"][feat][1]

        dsj.ub_constraint = pe.Constraint(features, rule=ub_rule)

        slope = leaves[tree][leaf]["slope"]
        intercept = leaves[tree][leaf]["intercept"]
        dsj.linear_exp = pe.Constraint(
            expr=sum(slope[k] * input_vars[k] for k in features) + intercept
            == block.intermediate_output[tree]
        )

    block.disjunct = Disjunct(t_l, rule=disjuncts_rule)

    @block.Disjunction(tree_ids)
    def disjunction_rule(b, tree):
        leaf_ids = list(leaves[tree].keys())
        return [block.disjunct[tree, leaf] for leaf in leaf_ids]

    block.total_output = pe.Constraint(
        expr=output_vars[0] == sum(block.intermediate_output[tree] for tree in tree_ids)
    )

    transformation_string = "gdp." + transformation

    if transformation != "custom":
        pe.TransformationFactory(transformation_string).apply_to(block)


def _add_hybrid_formulation_to_block(block, model_definition, input_vars, output_vars):
    """This function adds the Hybrid BigM representation to the OmltBlock.

    Arguments:
        block: OmltBlock
        model_definition: LinearTreeDefinition Object
        input_vars: input variables to the linear tree model
        output_vars: output variable of the linear tree model
    """
    leaves = model_definition.leaves
    input_bounds = model_definition.scaled_input_bounds
    n_inputs = model_definition.n_inputs

    # The set of trees
    tree_ids = list(leaves.keys())
    # Create a list of tuples that contains the tree and leaf indices. Note that
    # the leaf indices depend on the tree in the ensemble.
    t_l = [(tree, leaf) for tree in tree_ids for leaf in leaves[tree]]

    features = np.arange(0, n_inputs)

    # Use the input_bounds and the linear models in the leaves to calculate
    # the lower and upper bounds on the output variable. Required for Pyomo.GDP
    output_bounds = _build_output_bounds(model_definition, input_bounds)

    # Ouptuts are automatically scaled based on whether inputs are scaled
    block.outputs.setub(output_bounds[1])
    block.outputs.setlb(output_bounds[0])
    block.scaled_outputs.setub(output_bounds[1])
    block.scaled_outputs.setlb(output_bounds[0])

    # Create the intermeditate variables. z is binary that indicates which leaf
    # in tree t is returned. intermediate_output is the output of tree t and
    # the total output of the model is the sum of the intermediate_output vars
    block.z = pe.Var(t_l, within=pe.Binary)
    block.intermediate_output = pe.Var(tree_ids)

    @block.Constraint(features, tree_ids)
    def lower_bound_constraints(mdl, feat, tree):
        leaf_ids = list(leaves[tree].keys())
        return (
            sum(
                leaves[tree][leaf]["bounds"][feat][0] * mdl.z[tree, leaf]
                for leaf in leaf_ids
            )
            <= input_vars[feat]
        )

    @block.Constraint(features, tree_ids)
    def upper_bound_constraints(mdl, feat, tree):
        leaf_ids = list(leaves[tree].keys())
        return (
            sum(
                leaves[tree][leaf]["bounds"][feat][1] * mdl.z[tree, leaf]
                for leaf in leaf_ids
            )
            >= input_vars[feat]
        )

    @block.Constraint(tree_ids)
    def linear_constraint(mdl, tree):
        leaf_ids = list(leaves[tree].keys())
        return block.intermediate_output[tree] == sum(
            (
                sum(
                    leaves[tree][leaf]["slope"][feat] * input_vars[feat]
                    for feat in features
                )
                + leaves[tree][leaf]["intercept"]
            )
            * block.z[tree, leaf]
            for leaf in leaf_ids
        )

    @block.Constraint(tree_ids)
    def only_one_leaf_per_tree(b, tree):
        leaf_ids = list(leaves[tree].keys())
        return sum(block.z[tree, leaf] for leaf in leaf_ids) == 1

    @block.Constraint()
    def output_sum_of_trees(b):
        return output_vars[0] == sum(
            block.intermediate_output[tree] for tree in tree_ids
        )
