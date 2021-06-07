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
