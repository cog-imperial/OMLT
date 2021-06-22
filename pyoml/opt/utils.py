import pyomo.environ as pyo
from pyomo.core.base.var import ScalarVar, IndexedVar

pyomo_activations = {
    'tanh': pyo.tanh,
    'sigmoid': lambda x: 1 / (1 + pyo.exp(-x)),
    'softplus': lambda x: pyo.log(pyo.exp(x) + 1)
}

def build_full_space_formulation(block, network_structure, skip_activations=False):
    # for now, we build the full model with extraneous variables and constraints
    # Todo: provide an option to remove extraneous variables and constraints
    # Todo: support scaling
    net = network_structure
    scaling = net.scaling_object

    # map x and y to the inputs and outputs and verify the lengths
    # this is needed since the indexing in the input - output block
    # is not consistent with the nodal network representation
    input_node_ids = net.input_node_ids()
    inputs_list = block.inputs_list
    hidden_output_node_ids = list()
    hidden_output_node_ids.extend(net.hidden_node_ids())
    hidden_output_node_ids.extend(net.output_node_ids())
    output_node_ids = net.output_node_ids()
    outputs_list = block.outputs_list

    x = {input_node_ids[i]:inputs_list[i] for i in range(len(input_node_ids))}
    y = {output_node_ids[i]:outputs_list[i] for i in range(len(output_node_ids))}

    block.input_node_ids = pyo.Set(initialize=input_node_ids, ordered=True)

    # add the intermediate variables
    block.nodes = pyo.Set(initialize=net.all_node_ids(), ordered=True)
    block.hidden_output_nodes = pyo.Set(initialize=hidden_output_node_ids, ordered=True)
    block.z = pyo.Var(block.nodes, initialize=0) # post-activation
    block.zhat = pyo.Var(block.hidden_output_nodes, initialize=0) # pre-activation

    # define the input constraints
    inputs = x
    if scaling is not None:
        inputs = scaling.get_scaled_input_expressions(x)
    # Todo: We could eliminate these constraints and use x[i] directly where applicable
    block.input_constraints = pyo.Constraint(input_node_ids)
    for i in input_node_ids:
        block.input_constraints[i] = block.z[i] == inputs[i]

    # define the linear constraints
    block.linear_constraints = pyo.Constraint(block.hidden_output_nodes)
    w = net.weights
    b = net.biases
    for i in block.hidden_output_nodes:
        block.linear_constraints[i] = block.zhat[i] == sum(w[i][j] * block.z[j] for j in w[i]) + b[i]

    # define the activation constraints
    if not skip_activations:
        activations = net.activations
        block.activation_constraints = pyo.Constraint(block.hidden_output_nodes)
        for i in block.hidden_output_nodes:
            if i not in activations or activations[i] is None or activations[i] == 'linear':
                block.activation_constraints[i] = block.z[i] == block.zhat[i]
            elif type(activations[i]) is str:
                afunc = pyomo_activations[activations[i]]
                block.activation_constraints[i] = block.z[i] == afunc(block.zhat[i])
            else:
                # better have given us a function that is valid for pyomo expressions
                block.activation_constraints[i] = block.z[i] == activations[i](block.zhat[i])

    # define the output constraints
    outputs = {i:block.z[i] for i in output_node_ids}
    if scaling is not None:
        outputs = scaling.get_unscaled_output_expressions(outputs)
    # Todo: we could eliminate these constraints and use y[i] directly where applicable
    block.output_constraints = pyo.Constraint(output_node_ids)
    for i in output_node_ids:
        block.output_constraints[i] = y[i] == outputs[i]


def build_reduced_space_formulation(block, network_structure, skip_activations=False):
    # for now, we build the full model with extraneous variables and constraints
    # Todo: provide an option to remove extraneous variables and constraints
    # Todo: support scaling
    net = network_structure
    scaling = net.scaling_object

    # map x and y to the inputs and outputs and verify the lengths
    # this is needed since the indexing in the input - output block
    # is not consistent with the nodal network representation
    input_node_ids = net.input_node_ids()
    inputs_list = block.inputs_list
    hidden_output_node_ids = net.hidden_node_ids()
    hidden_output_node_ids.extend(net.output_node_ids())
    output_node_ids = net.output_node_ids()
    outputs_list = block.outputs_list
    x = {input_node_ids[i]:inputs_list[i] for i in range(len(input_node_ids))}
    y = {output_node_ids[i]:outputs_list[i] for i in range(len(output_node_ids))}

    # add the intermediate variables
    block.nodes = pyo.Set(initialize=net.all_node_ids(), ordered=True)
    block.hidden_output_nodes = pyo.Set(initialize=hidden_output_node_ids, ordered=True)
    block.z = pyo.Expression(block.nodes) # post-activation
    block.zhat = pyo.Expression(block.hidden_output_nodes) # pre-activation

    # define the input constraints
    inputs = x
    if scaling is not None:
        inputs = scaling.get_scaled_input_expressions(inputs)
    # Todo: We could eliminate these constraints and use x[i] directly where applicable
    for i in input_node_ids:
        block.z[i] = inputs[i]

    # define the linear constraints
    block.linear_constraints = pyo.Constraint(block.hidden_output_nodes)
    w = net.weights
    b = net.biases
    for i in block.hidden_output_nodes:
        block.zhat[i] = sum(w[i][j] * block.z[j] for j in w[i]) + b[i]

    # define the activation constraints
    if not skip_activations:
        activations = net.activations
        for i in block.hidden_output_nodes:
            if i not in activations or activations[i] is None or activations[i] == 'linear':
                block.z[i] = block.zhat[i]
            elif type(activations[i]) is str:
                afunc = pyomo_activations[activations[i]]
                block.z[i] = afunc(block.zhat[i])
            else:
                # better have given us a function that is valid for pyomo expressions
                block.z[i] = activations[i](block.zhat[i])

    # define the output constraints
    outputs = {i:block.z[i] for i in output_node_ids}
    if scaling is not None:
        outputs = scaling.get_unscaled_output_expressions(outputs)
    # Todo: we could eliminate these constraints and use y[i] directly where applicable
    block.output_constraints = pyo.Constraint(output_node_ids)
    for i in output_node_ids:
        block.output_constraints[i] = y[i] == outputs[i]

# def _sparse_keras_sequential_to_dict(keras_model):
#     chain = keras_model
#     n_inputs = len(chain.get_weights()[0])
#
#     w = dict()
#     b = dict()
#     node = n_inputs# + 1
#     from_offset = 0
#     for layer in chain.layers:
#         W,bias = layer.get_weights()
#         n_from,n_nodes = W.shape
#         for i in range(n_nodes):
#             w[node] = dict()
#             for j in range(n_from):
#                 w[node][j+from_offset] = W[j,i]
#             b[node] = bias[i]
#             node += 1
#         from_offset += n_from
#
#     return w,b


def _extract_var_data(vars):
    if isinstance(vars, ScalarVar):
        return [vars]
    elif isinstance(vars, IndexedVar):
        if vars.indexed_set().is_ordered():
            return list(vars.values())
        raise ValueError('Expected IndexedVar: {} to be indexed over an ordered set.'.format(vars))
    elif isinstance(vars, list):
        # Todo: the above if should check if the item supports iteration rather than only list?
        varlist = list()
        for v in vars:
            if v.is_indexed():
                varlist.extend(v.values())
            else:
                varlist.append(v)
        return varlist
    else:
        raise ValueError("Unknown variable type {}".format(vars))


