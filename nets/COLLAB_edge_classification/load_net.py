"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.COLLAB_edge_classification.gated_gcn_net import GatedGCNNet
from nets.COLLAB_edge_classification.gat_net import GATNet
from nets.COLLAB_edge_classification.mlp_net import MLPNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GAT': GAT,
        'MLP': MLP,
    }
        
    return models[MODEL_NAME](net_params)