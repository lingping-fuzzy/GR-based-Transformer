import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
# from layers.gat_layer import GATLayer
from layers.graph_transformer_global_layer_hop import GraphTransformerLayer as GATLayer
from layers.mlp_readout_layer import MLPReadout


# this is a vertion that there is no edge feat, if you want edge feat, one check data-generate, g.e['feat']
# if there is no edge feat, you can add z

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.n_classes = net_params['n_classes']
        self.device = net_params['device']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.dropout = dropout
        
        self.embedding_h = nn.Linear(in_dim, int(hidden_dim * num_heads))

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, int(hidden_dim * num_heads))

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        # self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
        #                                       dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        # self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))

        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim * num_heads, num_heads,
                                              dropout, net_params) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, num_heads, dropout, net_params))

        self.MLP_layer = MLPReadout(out_dim, self.n_classes)
        
    def forward(self, g, h, e, h_lap_pos_enc=None):
        h = self.embedding_h(h)

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)


    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

    #def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        #V = label.size(0)
        #label_count = torch.bincount(label)
       # label_count = label_count[label_count.nonzero()].squeeze()
       # cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
       # cluster_sizes[torch.unique(label).long()] = label_count
       # weight = (V - cluster_sizes).float() / V
       # weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
       # criterion = nn.CrossEntropyLoss(weight=weight)
       # loss = criterion(pred, label)

       # return loss
    

    