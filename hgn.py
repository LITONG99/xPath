import torch
import torch.nn.functional as F
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
import math


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z = 0
        for i in range(len(embeds)):
            z += embeds[i] * beta[i]
        return z


class myHeteroGATConv(nn.Module):
    def __init__(
            self,
            edge_feats,
            num_etypes,
            in_feats,
            out_feats,
            num_heads,
            feat_drop=0.0,
            attn_drop=0.0,
            negative_slope=0.2,
            residual=False,
            activation=None,
            allow_zero_in_degree=False,
            bias=False,
            alpha=0.0,
            share_weight=False
    ):
        super(myHeteroGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._in_src_feats = self._in_dst_feats = in_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._shared_weight = share_weight
        self.edge_emb = nn.Parameter(torch.FloatTensor(size=(num_etypes, edge_feats)))
        if not share_weight:
            self.fc = self.weight = nn.ModuleDict({
                name: nn.Linear(in_feats[name], out_feats * num_heads, bias=False) for name in in_feats
            })
        else:
            in_dim = None
            for name in in_feats:
                if in_dim:
                    assert in_dim == in_feats[name]
                else:
                    in_dim = in_feats[name]
            self.fc = nn.Linear(in_dim, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._shared_weight:
                in_dim = None
                for name in in_feats:
                    if in_dim:
                        assert in_dim == in_feats[name]
                    else:
                        in_dim = in_feats[name]
                if in_dim != num_heads * out_feats:
                    self.res_fc = nn.Linear(in_dim, num_heads * out_feats, bias=False)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = nn.ModuleDict()
                for ntype in in_feats.keys():
                    if self._in_dst_feats[ntype] != num_heads * out_feats:
                        self.res_fc[ntype] = nn.Linear(self._in_dst_feats[ntype], num_heads * out_feats, bias=False)
                    else:
                        self.res_fc[ntype] = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._shared_weight:
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            for name in self.fc:
                nn.init.xavier_normal_(self.fc[name].weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        nn.init.normal_(self.edge_emb, 0, 1)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, nfeat, res_attn=None):
        with graph.local_scope():
            funcs = {}

            for ntype in graph.ntypes:
                h = self.feat_drop(nfeat[ntype])
                if self._shared_weight:
                    feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
                else:
                    feat = self.fc[ntype](h).view(-1, self._num_heads, self._out_feats)
                graph.nodes[ntype].data['ft'] = feat
                if self.res_fc is not None:
                    graph.nodes[ntype].data['h'] = h

            for src, etype, dst in graph.canonical_etypes:
                feat_src = graph.nodes[src].data['ft']
                feat_dst = graph.nodes[dst].data['ft']
                el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
                graph.nodes[src].data['el'] = el
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.nodes[dst].data['er'] = er
                e_feat = self.edge_emb[int(etype)].unsqueeze(0)
                e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)
                ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1).expand(graph.number_of_edges(etype),
                                                                             self._num_heads, 1)
                graph.apply_edges(fn.u_add_v("el", "er", "e"), etype=etype)
                graph.edges[etype].data["a"] = self.leaky_relu(graph.edges[etype].data.pop("e") + ee)

            hg = dgl.to_homogeneous(graph, edata=["a"])
            a = self.attn_drop(edge_softmax(hg, hg.edata.pop("a")))
            e_t = hg.edata['_TYPE']

            for src, etype, dst in graph.canonical_etypes:
                t = graph.get_etype_id(etype)
                graph.edges[etype].data["a"] = a[e_t == t]
                if res_attn is not None:
                    graph.edges[etype].data["a"] = graph.edges[etype].data["a"] * (1 - self.alpha) + res_attn[
                        etype] * self.alpha
                funcs[etype] = (fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))

            graph.multi_update_all(funcs, 'sum')
            rst = graph.ndata.pop('ft')
            graph.edata.pop("el")
            graph.edata.pop("er")
            if self.res_fc is not None:
                for ntype in graph.ntypes:
                    if self._shared_weight:
                        rst[ntype] = self.res_fc(graph.nodes[ntype].data['h']).view(
                            graph.nodes[ntype].data['h'].shape[0], self._num_heads, self._out_feats) + rst[ntype]
                    else:
                        rst[ntype] = self.res_fc[ntype](graph.nodes[ntype].data['h']).view(
                            graph.nodes[ntype].data['h'].shape[0], self._num_heads, self._out_feats) + rst[ntype]

            if self.bias:
                for ntype in graph.ntypes:
                    rst[ntype] = rst[ntype] + self.bias_param

            if self.activation:
                for ntype in graph.ntypes:
                    rst[ntype] = self.activation(rst[ntype])
            res_attn = {e: graph.edges[e].data["a"].detach() for e in graph.etypes}
            graph.edata.pop("a")
            return rst, res_attn


class SimpleHeteroHGN(nn.Module):
    def __init__(
            self,
            edge_dim,
            num_etypes,
            in_dims,
            num_hidden,
            num_classes,
            num_layers,
            heads,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            alpha,
            shared_weight,
    ):
        super(SimpleHeteroHGN, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.g = None
        self.g_cs = []
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu

        self.gat_layers.append(
            myHeteroGATConv(
                edge_dim,
                num_etypes,
                in_dims,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                alpha=alpha,
            )
        )
        for l in range(1, num_layers):
            in_dims = {n: num_hidden * heads[l - 1] for n in in_dims}
            self.gat_layers.append(
                myHeteroGATConv(
                    edge_dim,
                    num_etypes,
                    in_dims,
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    alpha=alpha,
                    share_weight=shared_weight,
                )
            )
        in_dims = num_hidden * heads[-2]
        self.fc = nn.Linear(in_dims, num_classes)
        self.epsilon = 1e-12

    def forward(self, X, target_ntype):
        h = X
        res_attn = None

        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, res_attn=res_attn)
            h = {n: h[n].flatten(1) for n in h}
        h = h[target_ntype]
        logits = self.fc(h)
        logits = logits / (torch.norm(logits, dim=1, keepdim=True) + self.epsilon)
        return logits

    def loss(self, x, target_ntype, target_node, label):
        logits = self.forward(x, target_ntype)
        y = logits[target_node]
        return self.cross_entropy_loss(y, label)



class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata['k'] = k
                sub_graph.dstdata['q'] = q
                sub_graph.srcdata['v_%d' % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
                attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')

                sub_graph.edata['t'] = attn_score.unsqueeze(-1)

            G.multi_update_all({etype: (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
                                for etype, e_id in edge_dict.items()}, cross_reducer='mean')

            new_h = {}
            for ntype in G.ntypes:
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class HGT(nn.Module):
    '''
    The code is based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt .
    '''
    def __init__(self, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(HGT, self).__init__()
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp[t], n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.g = None

    def forward(self, X, target_ntype):
        h = {}
        for ntype in self.node_dict:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](X[ntype]))

        for i in range(self.n_layers):
            h = self.gcs[i](self.g, h)
        return self.out(h[target_ntype])

    def loss(self, x, target_ntype, target_node, label):
        logits = self.forward(x, target_ntype)
        y = logits[target_node]
        return self.cross_entropy_loss(y, label)

