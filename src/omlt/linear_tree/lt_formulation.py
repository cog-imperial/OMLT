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

    def __init__(self, lt_definition, transformation="bigm", epsilon=0):
        """Create a LinearTreeGDPFormulation object.

        Arguments:
            lt_definition: LinearTreeDefintion Object

        Keyword Arguments:
            transformation: choose which Pyomo.GDP formulation to apply.
                Supported transformations are bigm, hull, mbigm, and custom
                (default: {'bigm'})
            epsilon: Tolerance to use in enforcing that choosing the right
                branch of a linear tree node can only happen if the feature
                is strictly greater than the branch value.(default: 0)

        Raises:
            Exception: If transformation not in supported transformations
        """
        super().__init__()
        self.model_definition = lt_definition
        self.transformation = transformation
        self.epsilon = epsilon

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

        input_vars = self.block.scaled_inputs
        if self.model_definition.is_scaled is True:
            output_vars = self.block.scaled_outputs
        else:
            output_vars = self.block.outputs

        _add_gdp_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=input_vars,
            output_vars=output_vars,
            transformation=self.transformation,
            epsilon=self.epsilon,
            include_leaf_equalities=True,
        )

    @property
    def pyomo_only(self):
        return True


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

    def __init__(self, lt_definition, epsilon=0):
        """Create a LinearTreeHybridBigMFormulation object.

        Arguments:
            lt_definition: LinearTreeDefinition Object

        Keyword Arguments:
            epsilon: Tolerance to use in enforcing that choosing the right
                branch of a linear tree node can only happen if the feature
                is strictly greater than the branch value.(default: 0)

        """
        super().__init__()
        self.model_definition = lt_definition
        self.epsilon = epsilon

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition.n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition.n_outputs))

    @property
    def pyomo_only(self):
        return True

    def _build_formulation(self):
        """Build formulation.

        This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        block = self.block
        leaves = self.model_definition.leaves

        _setup_scaled_inputs_outputs(
            block,
            self.model_definition.scaling_object,
            self.model_definition.scaled_input_bounds,
        )

        input_vars = self.block.scaled_inputs
        if self.model_definition.is_scaled is True:
            output_vars = self.block.scaled_outputs
        else:
            output_vars = self.block.outputs

        _add_gdp_formulation_to_block(
            block=block,
            model_definition=self.model_definition,
            input_vars=input_vars,
            output_vars=output_vars,
            transformation="custom",
            epsilon=self.epsilon,
            include_leaf_equalities=False,
        )

        pe.TransformationFactory("gdp.bound_pretransformation").apply_to(block)
        # It doesn't really matter what transformation we call next, so we just
        # use bigm--all it's going to do is create the exactly-one constraints
        # and mark all the disjunctive parts of the model as transformed.
        pe.TransformationFactory("gdp.bigm").apply_to(block)

        # We now create the \sum((a_l^Tx + b_l)*y_l for l in leaves) = d constraints
        # manually.
        features = np.arange(0, self.model_definition.n_inputs)

        @block.Constraint(list(leaves.keys()))
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
                * block.disjunct[tree, leaf].binary_indicator_var
                for leaf in leaf_ids
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


def _add_gdp_formulation_to_block(  # noqa: PLR0913
    block,
    model_definition,
    input_vars,
    output_vars,
    transformation,
    epsilon,
    include_leaf_equalities,
):
    """This function adds the GDP representation to the OmltBlock using Pyomo.GDP.

    Arguments:
        block: OmltBlock
        model_definition: LinearTreeDefinition Object
        input_vars: input variables to the linear tree model
        output_vars: output variable of the linear tree model
        transformation: Transformation to apply
        epsilon: Tolerance to use in enforcing that choosing the right
            branch of a linear tree node can only happen if the feature
            is strictly greater than the branch value.
        include_leaf_equalities: boolean to indicate if the formulation
            should include the equalities setting the leaf values or not.
            (default: True)
    """
    leaves = model_definition.leaves
    scaled_input_bounds = model_definition.scaled_input_bounds
    unscaled_input_bounds = model_definition.unscaled_input_bounds
    n_inputs = model_definition.n_inputs

    # The set of leaves and the set of features
    tree_ids = list(leaves.keys())
    t_l = [(tree, leaf) for tree in tree_ids for leaf in leaves[tree]]
    features = np.arange(0, n_inputs)

    # Use the input_bounds and the linear models in the leaves to calculate
    # the lower and upper bounds on the output variable. Required for Pyomo.GDP
    # Ouptuts are automatically scaled based on whether inputs are scaled
    scaled_output_bounds = _build_output_bounds(model_definition, scaled_input_bounds)
    block.scaled_outputs.setub(scaled_output_bounds[1])
    block.scaled_outputs.setlb(scaled_output_bounds[0])

    if unscaled_input_bounds is not None:
        unscaled_output_bounds = _build_output_bounds(
            model_definition, unscaled_input_bounds
        )
        block.outputs.setub(unscaled_output_bounds[1])
        block.outputs.setlb(unscaled_output_bounds[0])

    if model_definition.is_scaled is True:
        block.intermediate_output = pe.Var(
            tree_ids, bounds=(scaled_output_bounds[0], scaled_output_bounds[1])
        )
    else:
        block.intermediate_output = pe.Var(
            tree_ids, bounds=(unscaled_output_bounds[0], unscaled_output_bounds[1])
        )

    # Create a disjunct for each leaf containing the bound constraints
    # and the linear model expression.
    def disjuncts_rule(dsj, tree, leaf):
        def lb_rule(dsj, feat):
            return input_vars[feat] >= leaves[tree][leaf]["bounds"][feat][0] + epsilon

        dsj.lb_constraint = pe.Constraint(features, rule=lb_rule)

        def ub_rule(dsj, feat):
            return input_vars[feat] <= leaves[tree][leaf]["bounds"][feat][1]

        dsj.ub_constraint = pe.Constraint(features, rule=ub_rule)

        if include_leaf_equalities:
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
