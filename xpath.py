import torch
import numpy as np
import dgl
import torch.nn.functional as F
from tqdm import tqdm
import json
import random

MIN_INT = -1000000

def get_original_name(p, origin_ids):
    p = p[:-1].split('-')
    i = 0
    p_original = ""
    while i < len(p):
        tp = p[i]
        p_original += f"{tp}-{origin_ids[tp][int(p[i + 1])]},"
        i += 2
    return p_original


def get_s(p1, p2, label=0):
    max_id = np.argmax(p2)
    if label == max_id:
        res = -1
    else:
        res = 1
    return res + (p1[label] - p2[label])

class xPath_Explainer:
    def __init__(
            self,
            model,
            g,
            target_ntype,
            num_layers,
            pred_list_path,
            device,
    ):
        self.target_ntype = target_ntype
        self.model = model
        self.device = device
        self.model.eval()
        self.num_layers = num_layers
        self.pred_list_path = pred_list_path
        self.n2etp = {}

        for src_ntype, etype, dst_ntype in g.canonical_etypes:
            self.n2etp[(src_ntype, dst_ntype)] = (src_ntype, etype, dst_ntype)

        self.prediction_list = {}
        self.one_hop_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)


    def sample_step(self, g, ap, sample_n):
        tmp = ap[:-1].split('-')
        aptp = [tmp[i] for i in range(0, len(tmp), 2)]
        apid = [int(tmp[i]) for i in range(1, len(tmp), 2)]
        res = {}
        one_hop_loader = dgl.dataloading.NodeDataLoader(g, {aptp[-1]: torch.tensor([apid[-1]], dtype=torch.int64, device=self.device)},
                                                        self.one_hop_sampler, batch_size=1, shuffle=False, drop_last=False)
        for neighbors, _, _ in one_hop_loader:
            for tp in neighbors:
                res[tp] = neighbors[tp].detach().cpu().tolist()
                if len(res[tp]) > sample_n:
                    res[tp] = random.sample(res[tp], sample_n)
            for tp, aid in zip(aptp, apid):
                if aid in res[tp]:
                    res[tp].remove(aid)
            break

        path = {}
        for tp in res:
            if len(res[tp]) > 0:
                mptp = tuple(aptp + [tp])
                path[mptp] = [apid + [i] for i in res[tp]]
        return path


    def get_proxy_graph(self, g, mptp, p):
        g = g.to('cpu')

        if len(p) == 1:
            return

        x = {tp: g.ndata["nfeat"][tp].clone().detach().cpu() for tp in g.ntypes}
        sg_n = {n: g.nodes[n].data['nfeat'].shape[0] for n in g.ntypes}
        proxy_ids = [-1]

        for i in range(1, len(mptp) - 1):
            node_tp = mptp[i]
            node_id = p[i]
            node_feature = x[node_tp][node_id, :].reshape(shape=(1, -1))

            proxy_ids.append(sg_n[node_tp])
            x[node_tp] = torch.concat([x[node_tp], node_feature], dim=0)
            sg_n[node_tp] += 1

        sg_edges = {}
        for stp, etp, ttp in g.canonical_etypes:
            sg_edges[(stp, etp, ttp)] = [g.edges(etype=etp)[0].tolist(), g.edges(etype=etp)[1].tolist()]

        new_edges = {k: [[], []] for k in g.canonical_etypes}

        for i in range(1, len(mptp)):
            node_tp = mptp[i]

            if i != len(mptp) - 1:
                etp = self.n2etp[(node_tp, node_tp)]
                new_edges[etp][0] += [proxy_ids[i]]
                new_edges[etp][1] += [proxy_ids[i]]

                etp = self.n2etp[(node_tp, mptp[i + 1])]
                if i != len(mptp) - 2:
                    new_edges[etp][0] += [proxy_ids[i]]
                    new_edges[etp][1] += [proxy_ids[i + 1]]
                else:
                    new_edges[etp][0] += [proxy_ids[i]]
                    new_edges[etp][1] += [p[-1]]

            if i != 1:
                etp = self.n2etp[(node_tp, mptp[i - 1])]
                if i != len(mptp) - 1:
                    new_edges[etp][0] += [proxy_ids[i]]
                    new_edges[etp][1] += [proxy_ids[i - 1]]
                else:
                    new_edges[etp][0] += [p[-1]]
                    new_edges[etp][1] += [proxy_ids[i - 1]]

        del_id = -1
        for stp, etp, dtp in g.canonical_etypes:
            for i in range(1, len(mptp) - 1):
                if stp == mptp[i]:
                    for j in range(len(sg_edges[(stp, etp, dtp)][0])):
                        sid = sg_edges[(stp, etp, dtp)][0][j]
                        tid = sg_edges[(stp, etp, dtp)][1][j]
                        if sid == p[i]:
                            if (dtp != mptp[i - 1] or tid != p[i - 1]) \
                                    and (dtp != mptp[i] or tid != p[i]) \
                                    and (dtp != mptp[i + 1] or tid != p[i + 1]):
                                new_edges[(stp, etp, dtp)][0] += [proxy_ids[i]]
                                new_edges[(stp, etp, dtp)][1] += [tid]
            if stp == mptp[-1] and dtp == mptp[-2]:
                for j in range(len(sg_edges[(stp, etp, dtp)][0])):
                    if sg_edges[(stp, etp, dtp)][0][j] == p[-1] and sg_edges[(stp, etp, dtp)][1][j] == p[-2]:
                        del_id = j
                        break
                sg_edges[(stp, etp, dtp)][0].pop(del_id)
                sg_edges[(stp, etp, dtp)][1].pop(del_id)

            sg_edges[(stp, etp, dtp)][0] += new_edges[((stp, etp, dtp))][0]
            sg_edges[(stp, etp, dtp)][1] += new_edges[((stp, etp, dtp))][1]
            sg_edges[(stp, etp, dtp)] = (sg_edges[(stp, etp, dtp)][0], sg_edges[(stp, etp, dtp)][1])

        sg = dgl.heterograph(sg_edges)
        for tp in g.ntypes:
            sg.nodes[tp].data['nfeat'] = x[tp]

        return sg


    def explain_beam(self, g, node_list, beam=3, sample_n=10):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
        subgraph_dataloader = dgl.dataloading.NodeDataLoader(g, {self.target_ntype: node_list.type(torch.int64)}, sampler,
                                                             batch_size=1, shuffle=False, drop_last=False)
        j = 0
        xpath = {}

        for neighbors, _, _ in tqdm(subgraph_dataloader):
            subgraph = dgl.node_subgraph(g, neighbors)
            target = node_list[j]

            subgraph = subgraph.to(self.device)
            origin_ids = {tp: subgraph.ndata['_ID'][tp].tolist() for tp in subgraph.ntypes}
            id_target = origin_ids[self.target_ntype].index(target)

            x = {tp: subgraph.ndata["nfeat"][tp].clone() for tp in subgraph.ntypes}
            self.model.g = subgraph
            logits = self.model.forward(x, self.target_ntype)

            y0 = logits[id_target]
            Prob0 = F.softmax(y0, dim=0).detach().cpu().tolist()
            original_pred_id = np.argmax(Prob0)
            self.prediction_list[int(target.item())] = original_pred_id.item()

            ancestor_p = [f"{self.target_ntype}-{id_target}-"]
            top_k_p = {f"{self.target_ntype}-{id_target}-": -100}
            visited = {}
            path2s = {}

            while len(ancestor_p) > 0:
                for ap in ancestor_p:
                    paths = self.sample_step(subgraph, ap, sample_n=sample_n)
                    for mptp in paths:
                        for pid in range(len(paths[mptp])):
                            p = paths[mptp][pid]
                            path_key = ''
                            for k, tp in enumerate(mptp):
                                path_key += f"{tp}-{p[k]}-"

                            shadow_graph = self.get_proxy_graph(subgraph, mptp, p).to(self.device)
                            x = shadow_graph.ndata.pop("nfeat")
                            self.model.g = shadow_graph
                            logits = self.model.forward(x, self.target_ntype)
                            y = logits[id_target]
                            Prob = F.softmax(y, dim=0).detach().cpu().tolist()

                            path2s[path_key] = get_s(Prob0, Prob, label=original_pred_id)
                            top_k_p[path_key] = path2s[path_key]

                values = list(top_k_p.values())
                keys = list(top_k_p.keys())
                ind = np.argsort(values)[-beam:]
                top_k_p = {keys[b]: top_k_p[keys[b]] for b in ind}

                ancestor_p = []
                for b in top_k_p:
                    tmp = b[:-1].split('-')
                    if (len(tmp) <= 2 * self.num_layers) and (not b in visited):
                        ancestor_p.append(b)
                        visited[b] = 1

            path2s = {get_original_name(i, origin_ids): path2s[i] for i in path2s}
            target = int(target.item())
            xpath[target] = path2s

            j += 1


        with open(self.pred_list_path, 'w') as f:
            json.dump(self.prediction_list, f)

        return xpath
