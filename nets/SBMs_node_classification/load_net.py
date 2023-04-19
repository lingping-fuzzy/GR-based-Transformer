"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet,denseGraphTransformerNet

def GraphTransformer(net_params):
    if net_params['use_dense'] == False:
        return GraphTransformerNet(net_params)
    else:
        return denseGraphTransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer
    }
        
    return models[MODEL_NAME](net_params)