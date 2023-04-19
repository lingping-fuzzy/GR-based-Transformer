import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

"""
    Graph Transformer with edge features
    
"""
# from layers.graph_transformer_edge_layer import GraphTransformerLayer
# use self-defined global-text based transformer
from layers.graph_transformer_global_edge_layer import GraphTransformerLayer

from layers.mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        # self.layer_norm = net_params['layer_norm']
        # self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        # self.hidden_rank = net_params['hidden_rank']
        self.use_dense = net_params['use_dense']
        self.attention_type = net_params['global_attention_type']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                    net_params) for _ in range(n_layers-1) ])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout,
                                                 net_params))
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        
        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        node_batch, edge_batch = None, None
        if self.attention_type == 'twohop':
            node_batch = g.batch_num_nodes()
            edge_batch = g.batch_num_edges()

            adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            t = adj.todense()
            tadj = np.matmul(t, t)
            np.fill_diagonal(tadj, 0)
            sou, tars = tadj.nonzero()
        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
            if self.attention_type == 'twohop':
                h, e = conv(g, h, e, sou, tars)
            else:
                h, e = conv(g, h, e)

        g.ndata['h'] = h

        if self.attention_type == 'twohop':
            batch_graph = dgl.unbatch(g, node_batch, edge_batch)
            g = dgl.batch(batch_graph)

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss


class denseGraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        num_heads = net_params['n_heads']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        # self.layer_norm = net_params['layer_norm']
        # self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        # self.hidden_rank = net_params['hidden_rank']
        self.use_dense = net_params['use_dense']
        self.attention_type = net_params['global_attention_type']
        max_wl_role_index = 37  # this is maximum graph size in the dataset

        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        # half_dim = int(hidden_dim/2)
        # self.layers = nn.ModuleList([GraphTransformerLayer(int(hidden_dim+half_dim*(i)), half_dim, num_heads, dropout,
        #                                                    net_params) for i in range(n_layers - 1)])
        # self.layers.append(GraphTransformerLayer(int(hidden_dim+half_dim*(n_layers-1)), out_dim, num_heads, dropout,
        #                                          net_params))

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout,
                                                           net_params) for i in range(n_layers - 1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout,
                                                 net_params))
        # if use dense layer then hidden_dim can be half
        self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        # calculate the batch information
        node_batch, edge_batch = None, None
        if self.attention_type == 'twohop':
            node_batch = g.batch_num_nodes()
            edge_batch = g.batch_num_edges()

            adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            t = adj.todense()
            tadj = np.matmul(t, t)
            np.fill_diagonal(tadj, 0)
            sou, tars = tadj.nonzero()
            # num_edge_old = g.num_edges()
            # g.add_edges(sou, tars)
            # g.ndata['sou'] = sou
            # g.ndata['tars'] = tars

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc
        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.size(0), 1).to(self.device)
        e = self.embedding_e(e)

        # convnets
        out_h = [h]
        out_e = [e]
        for conv in self.layers:
            # out_h_s = torch.cat(out_h, 1)
            # out_e_s = torch.cat(out_e, 1)
            out_h_s = torch.stack(out_h, dim=0).sum(dim=0)
            out_e_s = torch.stack(out_e, dim=0).sum(dim=0)

            if self.attention_type == 'twohop':
                h, e = conv(g, out_h_s, out_e_s, sou, tars)
            else:
                h, e = conv(g, out_h_s, out_e_s)

            out_h.append(h)
            out_e.append(e)
            # https: // github.com / Lornatang / DenseNet - PyTorch
            #     https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
        h = torch.stack(out_h, dim=0).sum(dim=0)
        g.ndata['h'] = h

        # if self.attention_type == 'twohop':
        #     data_x = h[count_nr:count_nl]
        if self.attention_type == 'twohop':
            batch_graph = dgl.unbatch(g, node_batch, edge_batch)
            g = dgl.batch(batch_graph)

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes


        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss