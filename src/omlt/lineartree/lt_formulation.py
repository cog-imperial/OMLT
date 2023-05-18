import collections

import numpy as np
import pyomo.environ as pe

from omlt.formulation import _PyomoFormulation, _setup_scaled_inputs_outputs
from pyomo.gdp import Disjunct, Disjunction


class LinearTreeGDPFormulation(_PyomoFormulation):
    """
    Class to add LinearTree GDP formulation to OmltBlock. We use Pyomo.GDP
    to create the disjuncts and disjunctions and then apply a transformation
    to convert to mixed-integer programming representation.

    Attributes:
        Inherited from _PyomoFormulation Class
        model_definition : LinearTreeModel object
        transformation : choose which transformation to apply. The supported
            transformations are bigm, mbigm, and hull.

    References:
        * Ammari et al. (2023) Linear Model Decision Trees as Surrogates in
            Optimization of Engineering Applications 
        * Chen et al. (2022) Pyomo.GDP: An ecosystem for logic based modeling 
            and optimization development. Optimization and Engineering, 
            23:607–642
    """
    def __init__(self, lt_model, transformation='bigm'):
        """
        Create a LinearTreeGDPFormulation object 

        Arguments:
            lt_model -- trained linear-tree model

        Keyword Arguments:
            transformation -- choose which Pyomo.GDP formulation to apply. 
                Supported transformations are bigm, hull, mbigm 
                (default: {'bigm'})

        Raises:
            Exception: If transformation not in supported transformations
        """
        super().__init__()
        self.model_definition = lt_model
        self.transformation = transformation
        
        # Ensure that the GDP transformation given is supported
        supported_transformations = ['bigm', 'hull', 'mbigm']
        if transformation not in supported_transformations:
            raise Exception("Transformation must be bigm, mbigm, or hull")

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition._n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition._n_outputs))

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition._scaling_object,
            self.model_definition._scaled_input_bounds,
        )

        add_GDP_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs,
            transformation = self.transformation
        )


class LinearTreeHybridBigMFormulation(_PyomoFormulation):
    """
    Class to add LinearTree Hybrid Big-M formulation to OmltBlock. 

    Attributes:
        Inherited from _PyomoFormulation Class
        model_definition : LinearTreeModel object
        transformation : choose which transformation to apply. The supported
            transformations are bigm, mbigm, and hull.

    References:
        * Ammari et al. (2023) Linear Model Decision Trees as Surrogates in
            Optimization of Engineering Applications 
    """
    def __init__(self, lt_model):
        """
        Create a LinearTreeHybridBigMFormulation object 

        Arguments:
            lt_model -- trained linear-tree model

        Keyword Arguments:
            transformation -- choose which Pyomo.GDP formulation to apply. 
                Supported transformations are bigm, hull, mbigm 
                (default: {'bigm'})
        """
        super().__init__()
        self.model_definition = lt_model

    @property
    def input_indexes(self):
        """The indexes of the formulation inputs."""
        return list(range(self.model_definition._n_inputs))

    @property
    def output_indexes(self):
        """The indexes of the formulation output."""
        return list(range(self.model_definition._n_outputs))

    def _build_formulation(self):
        """This method is called by the OmltBlock to build the corresponding
        mathematical formulation on the Pyomo block.
        """
        _setup_scaled_inputs_outputs(
            self.block,
            self.model_definition._scaling_object,
            self.model_definition._scaled_input_bounds,
        )

        add_hybrid_formulation_to_block(
            block=self.block,
            model_definition=self.model_definition,
            input_vars=self.block.scaled_inputs,
            output_vars=self.block.scaled_outputs
        )


def build_output_bounds(model_def, input_bounds):
    """
    This helper function develops bounds of the output variable based on the 
    values of the input_bounds and the signs of the slope

    Arguments:
        model_def -- Model definition
        input_bounds -- Dict of input bounds

    Returns:
        List that contains the conservative lower and upper bounds of the 
        output variable
    """
    leaves = model_def._leaves
    n_inputs = model_def._n_inputs
    T = np.array(list(leaves.keys()))
    features = np.arange(0, n_inputs)

    # Initialize bounds and variables
    bounds = [0, 0]
    upper_bound = 0
    lower_bound = 0
    for t in T:
        for l in leaves[t]:
            slopes = leaves[t][l]['slope']
            intercept = leaves[t][l]['intercept']
            for k in features:
                if slopes[k] <= 0:
                    upper_bound += slopes[k]*input_bounds[k][0]+intercept
                    lower_bound += slopes[k]*input_bounds[k][1]+intercept
                else:
                    upper_bound += slopes[k]*input_bounds[k][1]+intercept
                    lower_bound += slopes[k]*input_bounds[k][0]+intercept
                if upper_bound >= bounds[1]:
                    bounds[1] = upper_bound
                if lower_bound <= bounds[0]:
                    bounds[0]= lower_bound
            upper_bound = 0
            lower_bound = 0
    
    return bounds 


def add_GDP_formulation_to_block(
    block, model_definition, input_vars, output_vars, transformation):
    """
    This function adds the GDP representation to the OmltBlock using Pyomo.GDP

    Arguments:
        block -- OmltBlock
        model_definition -- LinearTreeModel
        input_vars -- input variables to the linear tree model
        output_vars -- output variable of the linear tree model
        transformation -- Transformation to apply

    """
    leaves = model_definition._leaves
    input_bounds = model_definition._scaled_input_bounds
    n_inputs = model_definition._n_inputs

    # The set of leaves and the set of features
    T = list(leaves.keys())
    t_l = []
    for tree in T:
        for l in leaves[tree].keys():
            t_l.append((tree, l))
    features = np.arange(0, n_inputs)

    # Use the input_bounds and the linear models in the leaves to calculate
    # the lower and upper bounds on the output variable. Required for Pyomo.GDP
    output_bounds = build_output_bounds(model_definition, input_bounds)
    
    # Ouptuts are automatically scaled based on whether inputs are scaled
    block.outputs.setub(output_bounds[1])
    block.outputs.setlb(output_bounds[0])
    block.scaled_outputs.setub(output_bounds[1])
    block.scaled_outputs.setlb(output_bounds[0])

    block.intermediate_output = pe.Var(T, bounds=(output_bounds[0], output_bounds[1]))

    # Create a disjunct for each leaf containing the bound constraints
    # and the linear model expression.
    def disjuncts_rule(d, t, l):
        def lb_Rule(d, f):
            return input_vars[f] >= leaves[t][l]['bounds'][f][0]
        d.lb_constraint = pe.Constraint(features, rule=lb_Rule)

        def ub_Rule(d, f):
            return input_vars[f] <= leaves[t][l]['bounds'][f][1]
        d.ub_constraint = pe.Constraint(features, rule=ub_Rule)

        slope = leaves[t][l]['slope']
        intercept = leaves[t][l]['intercept']
        d.linear_exp = pe.Constraint(
            expr=sum(slope[k]*input_vars[k] for k in features) 
            + intercept == block.intermediate_output[t]
            )    
    block.disjunct = Disjunct(t_l, rule=disjuncts_rule)

    @block.Disjunction(T)
    def disjunction_rule(b, t):
        L = list(leaves[t].keys())
        return [block.disjunct[t, l] for l in L]

    block.total_output = pe.Constraint(expr = 
                                       output_vars[0] == 
                                       sum(block.intermediate_output[t] for t in T))
    
    transformation_string = 'gdp.' + transformation
    
    pe.TransformationFactory(transformation_string).apply_to(block)


def add_hybrid_formulation_to_block(
    block, model_definition, input_vars, output_vars):
    """
    This function adds the Hybrid BigM representation to the OmltBlock

    Arguments:
        block -- OmltBlock
        model_definition -- LinearTreeModel
        input_vars -- input variables to the linear tree model
        output_vars -- output variable of the linear tree model
    """
    leaves = model_definition._leaves
    input_bounds = model_definition._scaled_input_bounds
    n_inputs = model_definition._n_inputs
    
    # The set of trees
    T = list(leaves.keys())
    # Create a list of tuples that contains the tree and leaf indices. Note that
    # the leaf indices depend on the tree in the ensemble.
    t_l = []
    for tree in T:
        for l in leaves[tree].keys():
            t_l.append((tree, l))

    features = np.arange(0, n_inputs)

    # Use the input_bounds and the linear models in the leaves to calculate
    # the lower and upper bounds on the output variable. Required for Pyomo.GDP
    output_bounds = build_output_bounds(model_definition, input_bounds)
    
    # Ouptuts are automatically scaled based on whether inputs are scaled
    block.outputs.setub(output_bounds[1])
    block.outputs.setlb(output_bounds[0])
    block.scaled_outputs.setub(output_bounds[1])
    block.scaled_outputs.setlb(output_bounds[0])

    # Create the intermeditate variables. z is binary that indicates which leaf
    # in tree t is returned. intermediate_output is the output of tree t and 
    # the total output of the model is the sum of the intermediate_output vars
    block.z = pe.Var(t_l, within=pe.Binary)
    block.intermediate_output = pe.Var(T)
    
    @block.Constraint(features, T)
    def lower_bound_constraints(m, f, t):
        L = list(leaves[t].keys())
        return sum(leaves[t][l]['bounds'][f][0]*m.z[t, l] for l in L) <= input_vars[f]

    @block.Constraint(features, T)
    def upper_bound_constraints(m, f, t):
        L = list(leaves[t].keys())
        return sum(leaves[t][l]['bounds'][f][1]*m.z[t, l] for l in L) >= input_vars[f]

    @block.Constraint(T)
    def linear_constraint(m, t):
        L = list(leaves[t].keys())
        return block.intermediate_output[t] == sum(
            (sum(leaves[t][l]['slope'][f]*input_vars[f] for f in features) + 
                            leaves[t][l]['intercept'])*block.z[t, l] for l in L
                            )

    @block.Constraint(T)
    def only_one_leaf_per_tree(b, t):
        L = list(leaves[t].keys())
        return sum(block.z[t, l] for l in L) == 1 
    
    @block.Constraint()
    def output_sum_of_trees(b):
        return output_vars[0] == sum(block.intermediate_output[t] for t in T)

