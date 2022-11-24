import dgl
import torch
import torch.nn.functional as F
from tqdm import tqdm


def fidelity_acc(gc, gm, label):
    tmp = 0
    if gc == label:
        tmp += 1
    if gm == label:
        tmp -= 1
    return tmp


def fidelity_prob(gc, gm, label):
    return gc[label] - gm[label]


def get_path_gm(g, x, target_ntype, nid, n2etp):
    gm_nodes = {tp: [] for tp in g.ntypes}
    gm_edges = {tp: [] for tp in g.canonical_etypes}

    for path in x[nid]:
        for i in range(len(path)):
            stp = path[i][0]
            sid = path[i][1]
            gm_nodes[stp].append(sid)
    gm_nodes = {tp: list(set(gm_nodes[tp])) for tp in gm_nodes}

    for path in x[nid]:
        for i in range(len(path) - 1):
            stp = path[i][0]
            sid = gm_nodes[stp].index(path[i][1])
            ttp = path[i + 1][0]
            tid = gm_nodes[ttp].index(path[i + 1][1])

            if not (stp == ttp and sid == tid):
                gm_edges[n2etp[(stp, ttp)]].append((sid, tid))
                gm_edges[n2etp[(ttp, stp)]].append((tid, sid))
    gm_edges = {k: list(set(gm_edges[k])) for k in gm_edges}
    for k in gm_edges:
        src = [i[0] for i in gm_edges[k]]
        dst = [i[1] for i in gm_edges[k]]
        gm_edges[k] = (src, dst)

    feat = {}

    for tp in gm_nodes:
        n = len(gm_nodes[tp])
        gm_edges[n2etp[(tp, tp)]] = ([i for i in range(n)], [i for i in range(n)])
        feat[tp] = g.nodes[tp].data['nfeat'][gm_nodes[tp], :]

    new_target_id = gm_nodes[target_ntype].index(nid)

    gm = dgl.heterograph(gm_edges)
    for tp in feat:
        gm.nodes[tp].data['nfeat'] = feat[tp]

    return gm, new_target_id


def eval_fidelity(x, g, model, label, target_ntype, n_layer, num_classes, node_list, device):
    n2etp = {}
    for srcntype, etype, dstntype in g.canonical_etypes:
        n2etp[(srcntype, dstntype)] = (srcntype, etype, dstntype)

    g = g.to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layer)
    subgraph_dataloader = \
        dgl.dataloading.NodeDataLoader(g, {target_ntype: node_list.type(torch.int64).to(device)},
                                    sampler, batch_size=1, shuffle=False, drop_last=False)
    i = 0
    fmask_accs = []
    fmask_probs = []
    for neighbors, _, _ in tqdm(subgraph_dataloader):
        target_id = int(node_list[i].item())
        target_label = int(label[target_id])

        # compute graph
        g_c = dgl.node_subgraph(g, neighbors)

        new_target_id = (g_c.ndata['_ID'][target_ntype].tolist()).index(target_id)
        g_c = g_c.to(device)
        X = {}
        for tp in g_c.ntypes:
            X[tp] = g_c.ndata["nfeat"][tp].clone()

        with torch.no_grad():
            model.g = g_c
            logits = model.forward(X, target_ntype).reshape(shape=(-1, num_classes))
            origin_probs = F.softmax(logits[new_target_id], dim=-1).detach().cpu()
            origin_prediction = origin_probs.argmax().item()

        # explanation graph
        g_m, new_target_id = get_path_gm(g.to('cpu'), x, target_ntype, target_id, n2etp=n2etp)
        g_m = g_m.to(device)
        X = {}
        for tp in g_m.ntypes:
            X[tp] = g_m.ndata["nfeat"][tp].clone()
        with torch.no_grad():
            model.g = g_m
            logits = model.forward(X, target_ntype).reshape(shape=(-1, num_classes))
            mask_probs = F.softmax(logits[new_target_id], dim=-1).detach().cpu()
            mask_prediction = mask_probs.argmax().item()

        fmask_acc = fidelity_acc(origin_prediction, mask_prediction, target_label)
        fmask_prob = fidelity_prob(origin_probs.tolist(), mask_probs.tolist(), target_label)
        fmask_accs.append(fmask_acc)
        fmask_probs.append(fmask_prob)

        i += 1

    return sum(fmask_accs) / i, sum(fmask_probs) / i


