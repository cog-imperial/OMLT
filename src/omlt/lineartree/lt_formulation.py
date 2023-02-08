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
            Optimization for Engineering Applications 
        * Chen et al. (2022) Pyomo.GDP: An ecosystem for logic based modeling 
            and optimization development. Optimization and Engineering, 
            23:607–642
    """
    def __init__(self, lt_model, transformation='bigm'):
        super().__init__()
        self.model_definition = lt_model
        self.transformation = transformation
        
        # Ensure that the GDP transformation given is supported
        supported_transformations = ['bigm', 'hull', 'mbigm']
        if transformation not in supported_transformations:
            raise Exception("Transformation must be bigm, mbigm, or hull")
        
        # Ensure that bounds are given otherwise cannot use Pyomo.GDP
        if self.model_definition._scaled_input_bounds == None:
            raise Exception("Input Bounds needed for Pyomo.GDP transformations")

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

# TODO Figure out a better way to deal with these bounds
def reassign_bounds(leaves, input_bounds):
    """
    This helper function reassigns bounds that are None to the bounds
    input by the user

    Arguments:
        leaves -- The dictionary of leaf information. Attribute of the 
            LinearTreeModel object
        input_bounds -- The nested dictionary

    Returns:
        The modified leaves dict without any bounds that are listed as None
    """
    L = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[L[0]]['slope']))
    if input_bounds is not None:
        for l in L:
            for f in features:
                if leaves[l]['bounds'][f][0] == None:
                    leaves[l]['bounds'][f][0] = input_bounds[f][0]
                if leaves[l]['bounds'][f][1] == None:
                    leaves[l]['bounds'][f][1] = input_bounds[f][1]
    # else:
    #     for l in L:
    #         for f in features:
    #             if leaves[l]['bounds'][f][0] == None:
    #                 leaves[l]['bounds'][f][0] = -1e50
    #             if leaves[l]['bounds'][f][1] == None:
    #                 leaves[l]['bounds'][f][1] = 1e50
    return leaves

def build_output_bounds(leaves, input_bounds):
    return


def add_GDP_formulation_to_block(
    block, model_definition, input_vars, output_vars, transformation):
    
    leaves = model_definition._leaves
    input_bounds = model_definition._scaled_input_bounds
    
    # The set of leaves and the set of features
    L = np.array(list(leaves.keys()))
    features = np.arange(0, len(leaves[L[0]]['slope']))

    leaves = reassign_bounds(leaves, input_bounds)

    # Create a disjunct for each leaf containing the bound constraints
    # and the linear model expression.
    def disjuncts_rule(d, l):

        def lb_Rule(d, f):
            return input_vars[f] >= leaves[l]['bounds'][f][0]
        d.lb_constraint = pe.Constraint(features, rule=lb_Rule)

        def ub_Rule(d, f):
            return input_vars[f] <= leaves[l]['bounds'][f][1]
        d.ub_constraint = pe.Constraint(features, rule=ub_Rule)

        slope = leaves[l]['slope']
        intercept = leaves[l]['intercept']
        d.linear_exp = pe.Constraint(
            expr=sum(slope[k]*input_vars[k] for k in features) 
            + intercept == output_vars[0]
            )
        
    block.disjunct = Disjunct(L, rule=disjuncts_rule)

    block.final_disjunction = Disjunction(
        expr=[block.disjunct[l] for l in L]
        )
    
    transformation_string = 'gdp.' + transformation
    
    pe.TransformationFactory(transformation_string).apply_to(block, bigM=1000)

