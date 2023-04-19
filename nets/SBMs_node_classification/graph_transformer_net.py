import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""
from layers.graph_transformer_global_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        # self.layer_norm = net_params['layer_norm']
        # self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.attention_type = net_params['global_attention_type']
        # self.hidden_rank = net_params['hidden_rank']
        # self.global_batch = net_params['global_batch']
        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        #self.layer_norm, self.batch_norm, self.residual, self.hidden_rank,  self.global_batch

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                              dropout, net_params) for _ in range(n_layers-1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, net_params))

        self.MLP_layer = MLPReadout(out_dim, n_classes)


    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        node_batch, edge_batch = None, None
        if self.attention_type == 'twohop':
            node_batch = g.batch_num_nodes()
            edge_batch = g.batch_num_edges()
            import numpy as np
            adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            t = adj.todense()
            tadj = np.matmul(t, t)
            np.fill_diagonal(tadj, 0)
            sou, tars = tadj.nonzero()
            # num_edge_old = g.num_edges()
            # g.add_edges(sou, tars)
        # input embedding
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            # h = conv(g, h)
            if self.attention_type == 'twohop':
                h = conv(g, h, sou, tars)
            else:
                h = conv(g, h)

        if self.attention_type == 'twohop':
            batch_graph = dgl.unbatch(g, node_batch, edge_batch)
            g = dgl.batch(batch_graph)

        # output
        h_out = self.MLP_layer(h)

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


class denseGraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        # self.layer_norm = net_params['layer_norm']
        # self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.attention_type = net_params['global_attention_type']
        # self.hidden_rank = net_params['hidden_rank']
        # self.global_batch = net_params['global_batch']
        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        # self.layer_norm, self.batch_norm, self.residual, self.hidden_rank,  self.global_batch

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                                           dropout, net_params) for _ in range(n_layers - 1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, net_params))

        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        node_batch, edge_batch = None, None
        if self.attention_type == 'twohop':
            node_batch = g.batch_num_nodes()
            edge_batch = g.batch_num_edges()
            import numpy as np
            adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            t = adj.todense()
            tadj = np.matmul(t, t)
            np.fill_diagonal(tadj, 0)
            sou, tars = tadj.nonzero()
            # num_edge_old = g.num_edges()
            # g.add_edges(sou, tars)

        # input embedding
        h = self.embedding_h(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        h = self.in_feat_dropout(h)

        # GraphTransformer Layers
        out_h = [h]

        for conv in self.layers:
            out_h_s = torch.stack(out_h, dim=0).sum(dim=0)

            if self.attention_type == 'twohop':
                h = conv(g, out_h_s, sou, tars)
            else:
                h = conv(g, out_h_s)

            # h = conv(g, out_h_s)
            out_h.append(h)

            # https: // github.com / Lornatang / DenseNet - PyTorch
            #     https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
        h = torch.stack(out_h, dim=0).sum(dim=0)
        # for conv in self.layers:
        #     h = conv(g, h)

        if self.attention_type == 'twohop':
            batch_graph = dgl.unbatch(g, node_batch, edge_batch)
            g = dgl.batch(batch_graph)
        # output
        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
        
