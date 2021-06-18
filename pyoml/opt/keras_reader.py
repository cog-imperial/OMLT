from pyoml.opt.network_definition import NetworkDefinition

def load_keras_sequential(nn):
    # print('*** NETWORK DEFN ***')
    # for l in nn.layers:
    #     print('... Layer', l)
    #     print('   weights')
    #     print(l.get_weights()[0])
    #     print('   biases')
    #     print(l.get_weights()[1])
    # print('^^^ NETWORK DEFN ^^^')

    n_inputs = len(nn.layers[0].get_weights()[0])
    n_outputs = len(nn.layers[-1].get_weights()[1])
    node_id_offset = n_inputs
    layer_offset = 0
    w = dict()
    b = dict()
    a = dict()
    for l in nn.layers:
        cfg = l.get_config()
        weights, biases = l.get_weights()
        n_layer_inputs, n_layer_nodes = weights.shape
        for i in range(n_layer_nodes):
            layer_w = dict()
            for j in range(n_layer_inputs):
                layer_w[j+layer_offset] = weights[j,i]
            w[node_id_offset] = layer_w
            b[node_id_offset] = biases[i]
            # ToDo: leaky ReLU
            a[node_id_offset] = cfg['activation']
            node_id_offset += 1
        layer_offset += n_layer_inputs
    n_nodes = len(a) + n_inputs
    n_hidden = n_nodes - n_inputs - n_outputs
    return NetworkDefinition(n_inputs=n_inputs,
                              n_hidden=n_hidden,
                              n_outputs=n_outputs,
                              weights=w,
                              biases=b,
                              node_activations=a
                            )

def _keras_sequential_to_dict(keras_model):
    chain = keras_model
    n_inputs = len(chain.get_weights()[0])

    w = dict()
    b = dict()
    node = n_inputs# + 1
    from_offset = 0
    for layer in chain.layers:
        W,bias = layer.get_weights()
        n_from,n_nodes = W.shape
        for i in range(n_nodes):
            w[node] = dict()
            for j in range(n_from):
                w[node][j+from_offset] = W[j,i]
            b[node] = bias[i]
            node += 1
        from_offset += n_from

    return w,b
