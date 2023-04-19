import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from torch.nn.parameter import Parameter

"""
    Graph Transformer Layer with edge features

"""

"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# # map the score-with r dimension to e-dim dimension
# def extending(field, W, out_field):
#     def func(edges):
#         return {out_field: ((edges.data[field]) @ W )}
#     return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """

    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, net_params=None, activation: Callable = torch.tanh):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.rank = net_params['hidden_rank']
        self.global_batch = net_params['global_batch']
        self.use_bias = net_params['use_bias']
        self.global_attention_type = net_params['global_attention_type']
        self.use_dense = net_params['use_dense']
        self.mask_rate = net_params['mask_rate']

        self._p = Parameter(torch.Tensor(in_dim, num_heads * self.rank))
        self._q = Parameter(torch.Tensor(in_dim, num_heads * self.rank))
        # self._e = Parameter(torch.Tensor(num_heads * self.rank, out_dim * num_heads))
        self.reset_parameters()

        if self.use_bias:
            # self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            # self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, self.rank * num_heads, bias=True)
            self.proj_eo = nn.Linear(self.rank * num_heads, out_dim * num_heads, bias=True)
            # self.W = nn.Linear(self.rank, out_dim * num_heads, bias=False) # false them with no errors
            # self.H = nn.Linear(self.rank, out_dim * num_heads, bias=False)

        else:
            # self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            # self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, self.rank * num_heads, bias=False)
            self.proj_eo = nn.Linear(self.rank * num_heads, out_dim * num_heads, bias=False)
            # self.W = nn.Linear(self.rank, out_dim * num_heads, bias=False)
            # self.H = nn.Linear(self.rank, out_dim * num_heads, bias=False)

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self._p, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self._q, a=math.sqrt(5))

    # propagate_attention with one-hop
    def propagate_attention(self, g):
        # Compute attention score
        # adj=g.adjacency_matrix(transpose=True,scipy_fmt='coo')
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # batch_graphs.nodes() or batch_graphs.edges()
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.rank)))

        # # extending to e-dim
        # g.apply_edges(extending('score', self._e, 'escore'))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))

        # softmax
        g.apply_edges(exp('score'))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def find_index(self, mat1, mat2):
        N, M = mat1.shape
        tid = []
        id = 0
        for i in range(N):
            for j in range(i + 1, M):
                if mat1[i, j] == 1:
                    id = id + 1
                elif mat2[i, j] == 1:
                    tid.append(id)
                    id = id + 1
                if mat1[j, i] == 1:
                    id = id + 1
                elif mat2[j, i] == 1:
                    tid.append(id)
                    id = id + 1
        return tid

    # propagate_attention with two-hop
    def propagate_attention_two(self, g, sou=None, tars=None):
        # Compute attention score
        ## first thing is to change the adjacency matrix, and in the end to change it back.
        # adj=g.adjacency_matrix(scipy_fmt='coo', ctx=g.device)
        # t= adj.todense()
        # tadj = np.matmul(t,t)
        # np.fill_diagonal(tadj, 0)
        # # idofTwo = self.find_index(t, tadj)
        # # edata = g.edata['feat']  # tensor data.
        # sou, tars = tadj.nonzero()
        num_edge_old = g.num_edges()
        # sou = g.extra['sou']
        # tars =g.extra['tars']
        # num_edge_old = g.ndata['num_edge_old']
        g.add_edges(sou, tars)
        num_edge_new = g.num_edges()

        # second_matrix =
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # batch_graphs.nodes() or batch_graphs.edges()
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.rank)))

        # # re-set the edges
        # reid = torch.arange(g.num_edges())
        # g.remove_edges(reid)
        # sour, targs = adj.nonzero()
        # g.add_edges(sour, targs, {'feat': edata})
        remove_id = torch.arange(num_edge_old, num_edge_new).to(g.device)
        g.remove_edges(remove_id)
        # g.remove_edges(torch.as_tensor(idofTwo) )
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # remoce edges and inadvance save the

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        g.apply_edges(exp('score'))
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, h, e, sou=None, tars=None):

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

        # [ h_(1*L) @ W_(L*r) ] @ [ H_(r*L) @ c_(L*1)   ] = scale_(1*1)
        # [ h_(1*L) @ W_(L*r) ] * [ c_(1*L) @ H_(L*r) ] = _(1*r)--> sum(_(1*r), 0) = scale_(1*1)
        # self._activation()

        K_h = atted @ self._q
        Q_h = h @ self._p

        V_h = self.V(h)
        proj_e = self.proj_e(e)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.rank)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.rank)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.rank)

        if self.global_attention_type == 'twohop':
            self.propagate_attention_two(g, sou=sou, tars=tars)
        else:
            self.propagate_attention(g)

        if torch.min(g.ndata['z']) == 0:
            g.ndata['z'][g.ndata['z'] == 0] = 1e-3

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))  # adding eps to all values here
        rank_e = g.edata['e_out']
        e_out = self.proj_eo(rank_e.view(-1, self.num_heads * self.rank))

        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, net_params=None):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = net_params['residual']

        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.global_attention_type = net_params['global_attention_type']

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, net_params)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e, sou=None, tars=None):
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        if self.global_attention_type == 'twohop':
            h_attn_out, e_attn_out = self.attention(g, h, e, sou, tars)
        else:
            h_attn_out, e_attn_out = self.attention(g, h, e)

        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                                                                   self.in_channels,
                                                                                   self.out_channels, self.num_heads,
                                                                                   self.residual)
