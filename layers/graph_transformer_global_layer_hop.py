import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
import dgl.function as fn
import numpy as np
from torch.nn.parameter import Parameter

"""
    Graph Transformer Layer
    
"""

"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, net_params=None):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.in_dim = in_dim

        self.rank = net_params['hidden_rank']
        self.global_batch = net_params['global_batch']
        self.use_bias =  net_params['use_bias']
        self.global_attention_type = net_params['global_attention_type']
        self.use_dense = net_params['use_dense']
        self.mask_rate = net_params['mask_rate']

        self._p = Parameter(torch.Tensor(in_dim, num_heads * self.rank))
        self._q = Parameter(torch.Tensor(in_dim, num_heads * self.rank ))
        self.reset_parameters()

        if self.use_bias:
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            # self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            # self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self._p, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self._q, a=math.sqrt(5))
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def propagate_attention_two(self, g, sou = None, tars = None):

        num_edge_old = g.num_edges()
        g.add_edges(sou, tars)
        num_edge_new = g.num_edges()

        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        remove_id = torch.arange(num_edge_old, num_edge_new).to(g.device)
        g.remove_edges(remove_id)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    # def draw_graph(self, g):
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     sub_g = dgl.unbatch(g)
    #     G = dgl.to_networkx(sub_g[0])
    #     plt.figure(figsize=[15, 7])
    #     nx.draw(G)

    def forward(self, g, h, sou = None, tars = None):

        if self.global_attention_type[:7] == 'onehop-':
            # adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            adj = g.adj( ctx=g.device)
            tadj = adj.to_dense()

            tadj= torch.clamp(tadj, max=1)
            tadj = tadj.fill_diagonal_(1)
            y = (tadj[0] == 1).nonzero().squeeze()
            graph_x =  torch.index_select(h, 0, y)
            if self.global_attention_type[-4:] == 'mean':
                c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'max':
                c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'min':
                c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            else:
                print("there is no this global_attention_type")
            atted = torch.unsqueeze(c, 0)
            for nod in range(1, h.shape[0]):
                y = (tadj[nod] == 1).nonzero().squeeze()
                graph_x = torch.index_select(h, 0, y)
                if self.global_attention_type[-4:] == 'mean':
                    c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'max':
                    c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'min':
                    c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                else:
                    print("there is no this global_attention_type")
                c = torch.unsqueeze(c, 0)
                atted = torch.cat((atted, c), 0)

        elif self.global_attention_type[:7] == 'twohop-':
            # adj = g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
            adj = g.adj( ctx=g.device)
            dadj = adj.to_dense()
            ### tadj = +dadj
            if self.global_attention_type[:11] == 'twohop-only':
                tadj = torch.add( torch.matmul(dadj, dadj), dadj)
            else:
                tadj = torch.matmul(dadj, dadj)

            tadj= torch.clamp(tadj, max=1)
            tadj = tadj.fill_diagonal_(1)
            y = (tadj[0] == 1).nonzero().squeeze()
            graph_x =  torch.index_select(h, 0, y)
            if self.global_attention_type[-4:] == 'mean':
                c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'max':
                c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'min':
                c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            else:
                print("there is no this global_attention_type")
            atted = torch.unsqueeze(c, 0)
            for nod in range(1, h.shape[0]):
                y = (tadj[nod] == 1).nonzero().squeeze()
                graph_x = torch.index_select(h, 0, y)
                if self.global_attention_type[-4:] == 'mean':
                    c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'max':
                    c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'min':
                    c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                else:
                    print("there is no this global_attention_type")
                c = torch.unsqueeze(c, 0)
                atted = torch.cat((atted, c), 0)

        elif self.global_attention_type[:6] == 'graph-':
            node_batch = g.batch_num_nodes()
            # edge_batch = g.batch_num_edges()

            graph_x = h[:node_batch[0]]
            if self.global_attention_type[-4:] == 'mean':
                c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'max':
                c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'min':
                c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
            else:
                print("there is no this global_attention_type")
            atted = c.repeat(node_batch[0], 1)

            for gra_num in range(1, len(node_batch)):
                st_id = node_batch[gra_num-1]
                end_id = node_batch[gra_num-1] + node_batch[gra_num]
                graph_x = h[st_id:end_id]
                if self.global_attention_type[-4:] == 'mean':
                    c = torch.mean(graph_x, dim=0)  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'max':
                    c = torch.max(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type[-3:] == 'min':
                    c = torch.min(graph_x, dim=0).values  # c.shape= 1*num_heads * in_dim
                else:
                    print("there is no this global_attention_type")
                c = c.repeat(node_batch[gra_num], 1)
                atted = torch.cat((atted, c), 0)

        elif self.global_attention_type[:7] == 'global-':
            if self.global_attention_type[-4:] == 'mean':
                c = torch.mean(h, dim=0)  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'max':
                c = torch.max(h, dim=0).values  # c.shape= 1*num_heads * in_dim
            elif self.global_attention_type[-3:] == 'min':
                c = torch.min(h, dim=0).values  # c.shape= 1*num_heads * in_dim
            else:
                dim_all = h.shape[0]

                mask_length = dim_all * self.in_dim
                un_mask = int(mask_length * self.mask_rate)
                onehot = np.ones([1, mask_length])
                using_idx = np.random.choice(np.arange(mask_length), un_mask, replace=False)

                if self.global_attention_type == 'poolmin':
                    np.put(onehot, using_idx, [5])   #we have clamb values into [-5, 5].
                elif self.global_attention_type == 'poolmean':
                    np.put(onehot, using_idx, [0])
                elif self.global_attention_type == 'poolmax':
                    np.put(onehot, using_idx, [-5])

                feature_mask = torch.from_numpy(onehot).type(torch.FloatTensor).to(h.device)
                feature_mask = feature_mask.view(dim_all, self.in_dim)
                h = torch.mul(h, feature_mask)

                if self.global_attention_type == 'poolmin':
                    c = torch.min(h, dim=0).values  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type == 'poolmean':
                    c = torch.mean(h, dim=0)  # c.shape= 1*num_heads * in_dim
                elif self.global_attention_type == 'poolmax':
                    c = torch.max(h, dim=0).values  # c.shape= 1*num_heads * in_dim
            # c =c.clamp(-5, 5)
            c = c.repeat(h.shape[0], 1)
            atted = c
        elif self.global_attention_type == 'transform':
            atted = h


        K_h = atted @ self._q
        Q_h = h @ self._p

        V_h = self.V(h)
        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.rank)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.rank)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        
        if self.global_attention_type == 'twohop':
            self.propagate_attention_two(g, sou= sou, tars=tars)
        else:
            self.propagate_attention(g)

        if torch.min(g.ndata['z']) == 0:
            g.ndata['z'][g.ndata['z'] == 0] = 1e-3
        
        head_out = g.ndata['wV']/g.ndata['z']
        
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, net_params= None):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = net_params['residual']

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.global_attention_type = net_params['global_attention_type']

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, net_params)
        
        self.O = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)
        
    def forward(self, g, h, sou = None, tars = None):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        # attn_out = self.attention(g, h)

        if self.global_attention_type == 'twohop':
            attn_out = self.attention(g, h, sou, tars)
        else:
            attn_out = self.attention(g, h)


        h = attn_out.view(-1, self.out_channels)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)
        
        if self.residual:
            h = h_in1 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection
        
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)