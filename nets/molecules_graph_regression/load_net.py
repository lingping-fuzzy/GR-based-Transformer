"""
    Utility file to select GraphNN model as
    selected by the user
"""
#from nets.molecules_graph_regression.graph_transformer_net import GraphTransformerNet, denseGraphTransformerNet
from nets.molecules_graph_regression.gat_net import GATNet
from nets.molecules_graph_regression.graphsage_net import GraphSageNet
from nets.molecules_graph_regression.gin_net import GINNet
from nets.molecules_graph_regression.mo_net import MoNet as MoNet_
from nets.molecules_graph_regression.mlp_net import MLPNet
from nets.molecules_graph_regression.ring_gnn_net import RingGNNNet
#from nets.molecules_graph_regression.three_wl_gnn_net import ThreeWLGNNNet

from nets.molecules_graph_regression.gated_gcn_net import GatedGCNNet
from nets.molecules_graph_regression.gcn_net import GCNNet

def GraphTransformer(net_params):
   # if net_params['use_dense'] == False:
       # return GraphTransformerNet(net_params)
    #else:
       # return denseGraphTransformerNet(net_params)
    return None

def GAT(net_params):
    return GATNet(net_params)
def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def RingGNN(net_params):
    return RingGNNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GraphTransformer': GraphTransformer,
        'GAT': GAT,
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GraphSage': GraphSage,
        'GIN': GIN,
        'MoNet': MoNet,
        'MLP': MLP,
        'RingGNN': RingGNN
    }
        
    return models[MODEL_NAME](net_params)