import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch.nn import Sequential, ModuleList, Linear, ReLU
from torch_scatter import scatter_mean, scatter_min, scatter_max, scatter_add

import torch.nn.functional as F
from torch_geometric.nn import max_pool_x, TAGConv, SGConv, PANConv
import numpy as np
import os

from configs import GRAPH_SIZE
from configs import *
from models.matching import SuperGlue
from models.regularizer import Regularizer
from models.model_utils import to_freqs, log_freq

from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix
from utils.utils import pytorch_count_params


class EdgeModelDef(torch.nn.Module):
    def __init__(self):
        super(EdgeModelDef, self).__init__()
        self.edge_mlp = Seq(Linear(13, 13), ReLU(), Linear(13, 13))

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat(
            [src, dest, edge_attr.unsqueeze(1)], 1
        )  # torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModelDef(torch.nn.Module):
    def __init__(self):
        super(NodeModelDef, self).__init__()
        self.node_mlp_1 = Seq(Linear(19, 19), ReLU(), Linear(19, 32))
        self.node_mlp_2 = Seq(Linear(38, 64), ReLU(), Linear(64, 64))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)  # torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)


class GlobalModelDef(torch.nn.Module):
    def __init__(self):
        super(GlobalModelDef, self).__init__()
        self.global_mlp = Seq(Linear(64, 64), ReLU(), Linear(64, 64))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = scatter_mean(
            x, batch, dim=0
        )  # torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class EdgeModel(torch.nn.Module):
    def __init__(self, edge_attr_sz_in, edge_attr_sz_out, node_attr_sz):

        super(EdgeModel, self).__init__()

        # feature_in = edge_attr_sz_in + 2 * node_attr_sz

        self.conv1 = torch.nn.Conv1d(edge_attr_sz_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, edge_attr_sz_out)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.

        x = torch.unsqueeze(edge_attr, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x


class NodeModel(torch.nn.Module):
    def __init__(self, node_attr_sz_in, node_attr_sz_out, edge_attr_sz):

        super(NodeModel, self).__init__()

        self.reshape_value = node_attr_sz_out * 8

        self.conv1 = torch.nn.Conv1d(
            node_attr_sz_in + edge_attr_sz, node_attr_sz_out, 1
        )
        self.conv2 = torch.nn.Conv1d(node_attr_sz_out, node_attr_sz_out * 2, 1)
        self.conv3 = torch.nn.Conv1d(node_attr_sz_out * 2, node_attr_sz_out * 8, 1)
        self.fc1 = nn.Linear(node_attr_sz_out * 8, node_attr_sz_out * 4)
        self.fc2 = nn.Linear(node_attr_sz_out * 4, node_attr_sz_out * 2)
        self.fc3 = nn.Linear(node_attr_sz_out * 2, node_attr_sz_out)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(node_attr_sz_out)
        self.bn2 = nn.BatchNorm1d(node_attr_sz_out * 2)
        self.bn3 = nn.BatchNorm1d(node_attr_sz_out * 8)
        self.bn4 = nn.BatchNorm1d(node_attr_sz_out * 4)
        self.bn5 = nn.BatchNorm1d(node_attr_sz_out * 2)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row, col = edge_index
        x = torch.cat([x[col], edge_attr], dim=1)
        x = torch.unsqueeze(x, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, self.reshape_value)

        x = scatter_mean(x, row, dim=0)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x


class Net(torch.nn.Module):
    def __init__(self, edge_attr_in_first=3, node_attr_in_first=3):

        super(Net, self).__init__()

        self.softmax = torch.nn.Softmax(dim=1)
        self.ml_shared = get_metalayer(
            edge_attr_in_first,
            edge_attr_in_first * 2,
            node_attr_in_first,
            FEATURE_SIZE // 4,
        )
        self.ml_d = ModuleList()
        self.ml_l = ModuleList()

        for i in range(2):

            if i == 0:
                node_attr_sz_in = FEATURE_SIZE // 4
                node_attr_sz_out = FEATURE_SIZE // 2
                edge_attr_sz_in = edge_attr_in_first * 2
                edge_attr_sz_out = edge_attr_in_first * 4

            if i == 1:
                node_attr_sz_in = FEATURE_SIZE // 2
                node_attr_sz_out = FEATURE_SIZE
                edge_attr_sz_in = edge_attr_in_first * 4
                edge_attr_sz_out = edge_attr_in_first * 8

            self.ml_d.append(
                get_metalayer(
                    edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out
                )
            )
            self.ml_l.append(
                get_metalayer(
                    edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out
                )
            )

    def forward(self, data):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x, edge_attr, _ = self.ml_shared(
            x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        x_d, edge_attr_d = x, edge_attr
        x_l, edge_attr_l = x, edge_attr

        for meta_layer in self.ml_d:
            x_d, edge_attr_d, _ = meta_layer(
                x=x_d, edge_index=edge_index, edge_attr=edge_attr_d, batch=batch
            )
        for meta_layer in self.ml_l:
            x_l, edge_attr_l, _ = meta_layer(
                x=x_l, edge_index=edge_index, edge_attr=edge_attr_l, batch=batch
            )

        x_d = torch.unsqueeze(x_d, 1)
        x_d, _ = scatter_max(x_d, batch, dim=0)
        x_d = torch.squeeze(x_d, 1)
        x_d = F.normalize(x_d, dim=1)

        x_l = torch.unsqueeze(x_l, 1)
        x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        x_l = torch.squeeze(x_l, 1)
        x_l = x_l.view([len(data.x) // GRAPH_SIZE, GRAPH_SIZE])
        x_l = self.softmax(x_l)

        return {"probabilities": x_l, "descriptors": x_d}


# https://github.com/phil-hawkins/OGREnet-v2/blob/ee21e5224fe18dbfe1cf49e86cdbf070687da899/ogrenetv2/model.py


def get_metalayer(edge_attr_sz_in, edge_attr_sz_out, node_attr_sz_in, node_attr_sz_out):
    return MetaLayer(
        EdgeModel(
            edge_attr_sz_in=edge_attr_sz_in,
            edge_attr_sz_out=edge_attr_sz_out,
            node_attr_sz=node_attr_sz_in,
        ),
        NodeModel(
            node_attr_sz_in=node_attr_sz_in,
            node_attr_sz_out=node_attr_sz_out,
            edge_attr_sz=edge_attr_sz_out,
        ),
        None,
    )


def init_model(path, settings):
    model = NetMatch(
        graph_match=False
    )  # if VERSION == 1 else NetMatch(graph_match=True)
    optimizer_dict = {}
    epoch = 0
    if settings.multi_gpu == 1:
        model = Net()
    if path != None and os.path.exists(path):
        checkpoint = torch.load(path)

        """filter dict"""
        model_dict = model.state_dict()
        # filter key
        pretrained_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict
        }
        # filter shape
        pretrained_dict = {
            k: v if v.shape == model_dict[k].shape else model_dict[k]
            for k, v in pretrained_dict.items()
        }

        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        # model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        model.config_dict = checkpoint["config_dict"]
        print("loaded model state from %s" % path)
        if "optimizer_state_dict" in checkpoint:
            optimizer_dict = checkpoint["optimizer_state_dict"]
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"] + 1

    if torch.cuda.is_available():
        model.to(torch.cuda.current_device())

    return model, optimizer_dict, epoch


class Graphite(torch.nn.Module):
    def __init__(self):

        super(Graphite, self).__init__()

        self.TAG1 = TAGConv(6, FEATURE_SIZE // 4, 1)
        self.TAG2 = TAGConv(FEATURE_SIZE // 4, FEATURE_SIZE // 2, 2)
        self.TAG3 = TAGConv(FEATURE_SIZE // 2, FEATURE_SIZE, 3)
        self.TAG35 = TAGConv(FEATURE_SIZE, FEATURE_SIZE, 4)
        self.bn1 = nn.BatchNorm1d(num_features=FEATURE_SIZE, eps=1e-3, momentum=1e-3)
        self.FCD1 = nn.Linear(FEATURE_SIZE, FEATURE_SIZE)
        self.bn2 = nn.BatchNorm1d(num_features=FEATURE_SIZE, eps=1e-3, momentum=1e-3)
        self.FCD2 = nn.Linear(FEATURE_SIZE, FEATURE_SIZE)

        self.TAG4 = TAGConv(FEATURE_SIZE, 16, 3)
        self.TAG5 = TAGConv(16, 8, 2)
        self.TAG6 = TAGConv(8, 1, 1)

        self.FC1 = nn.Linear(8, 8)
        self.FC2 = nn.Linear(8, 1)

        self.op = MetaLayer(EdgeModelDef(), NodeModelDef(), GlobalModelDef())
        if FIXED_GRAPH:
            self.FC3 = nn.Linear(GRAPH_SIZE, 1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, data):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        m = self.op(x, edge_index, edge_attr, batch=batch)
        x = self.TAG1(x, edge_index, edge_attr)
        x = self.TAG2(x, edge_index, edge_attr)
        x = self.TAG3(x, edge_index, edge_attr)

        # x=self.TAG3F(x,edge_index)
        m_d = m[2].unsqueeze(1)
        x_d = torch.unsqueeze(x, 1)
        x_d, _ = scatter_max(x_d, batch, dim=0)
        # x_d=m_d
        if x_d.shape[0] > 1:
            x_d = self.bn2(torch.transpose(self.FCD1(x_d), 1, 2))
        x_d = self.FCD2(torch.squeeze(x_d, 2))
        x_d = torch.squeeze(x_d, 1)
        x_d = F.normalize(x_d, dim=1)
        # x=torch.relu(self.TAG4F(x,edge_index))

        x = torch.relu(self.TAG4(x, edge_index, edge_attr))
        x = torch.relu(self.TAG5(x, edge_index, edge_attr))

        conf = torch.relu(self.FC2(torch.relu(self.FC1(x))))
        if FIXED_GRAPH:
            conf = conf.view([len(data.x) // GRAPH_SIZE, GRAPH_SIZE])
            conf = torch.relu(self.FC3(conf))
        else:
            conf = scatter_max(conf, batch, dim=0)[0]
        #
        x_l = torch.relu(self.TAG6(x, edge_index, edge_attr))

        # x_l = torch.unsqueeze(x, 1)
        # x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        # x_l = torch.squeeze(x_l, 1)
        if FIXED_GRAPH:
            x_l = x.view([len(data.x_l) // GRAPH_SIZE, GRAPH_SIZE])
            x_l = self.softmax(x_l)

        # x_d = torch.unsqueeze(x_d, 1)
        # x_d, _ = scatter_max(x_d, batch, dim=0)
        # x_d = torch.squeeze(x_d, 1)
        # x_d = F.normalize(x_d, dim=1)

        # x_l = torch.unsqueeze(x_l, 1)
        # x_l = torch.nn.functional._max_pool1d(x_l, FEATURE_SIZE)
        # x_l = torch.squeeze(x_l, 1)
        # x_l = x_l.view([len(data.x)//GRAPH_SIZE, GRAPH_SIZE])
        # x_l = self.softmax(x_l)

        return {"probabilities": x_l, "descriptors": x_d, "confidence": conf}


class NetMatch(torch.nn.Module):
    def __init__(self, graph_match=False):

        super(NetMatch, self).__init__()
        self.net_describe = Graphite()
        self.config_dict = {"graph_match": graph_match}

        self.detach_it = False
        self.net_match = SuperGlue(config={}).to(torch.cuda.current_device())
            # self.net_match_2 = SuperGlue(config={}).to(
            #     torch.cuda.current_device())
        # self.regularizer = Regularizer(num_layer= 3)#TODO:make this hyper

        # print("===trainable parameters===")
        # print("graphite: {}".format(pytorch_count_params(self.net_describe)))
        # print("matching: {}".format(pytorch_count_params(self.net_match)))

    def forward(self, input, meta=None):
        outputs_desc = {key: self.net_describe(input[key]) for key in input.keys()}
        all_matches = torch.zeros(outputs_desc["a"]["descriptors"].shape[0], 2).to(
            torch.cuda.current_device()
        )
        all_matches[:, 0] = all_matches[:, 1] = torch.arange(all_matches.shape[0])
        if self.config_dict["graph_match"] and ("p" in outputs_desc.keys()):
            assert meta
            match_pair = {}
            if self.detach_it == True:
                outputs_desc["a"]["descriptors"], outputs_desc["p"]["descriptors"] = (
                    outputs_desc["a"]["descriptors"].detach(),
                    outputs_desc["p"]["descriptors"].detach(),
                )
            (
                match_pair["descriptors0"],
                match_pair["descriptors1"],
                match_pair["keypoints0"],
                match_pair["keypoints1"],
                match_pair["all_matches"],
            ) = (
                outputs_desc["a"]["descriptors"],
                outputs_desc["p"]["descriptors"],
                meta["seeds_a"].to(torch.cuda.current_device()),
                meta["seeds_p"].to(torch.cuda.current_device()),
                all_matches.unsqueeze(0),
            )
            matching_results = self.net_match(match_pair, meta)

            # '''update descriptor with the esitmated confidnece'''
            # k_loses = matching_results['losses'].keys()
            # total_loss= {k: 0 for k in k_loses}
            # for k in k_loses:
            #     total_loss[k]+=matching_results['losses'][k]
            # match_pair2 = self.regularizer(match_pair, matching_results, meta)

            # # '''do matching again.'''
            # matching_results = self.net_match_2(match_pair2, meta)
            # for k in k_loses:
            #     total_loss[k]+=matching_results['losses'][k]
            # matching_results['losses'] = total_loss

        else:

            dist_d = distance_matrix(
                outputs_desc["a"]["descriptors"].detach().cpu().numpy(),
                outputs_desc["p"]["descriptors"].detach().cpu().numpy(),
                2,
            )
            matching_results = {}
            matching_results["matches0"] = torch.from_numpy(np.argmin(dist_d, 1)).to(
                torch.cuda.current_device()
            )
            matching_results["matches1"] = torch.from_numpy(np.argmin(dist_d, 0)).to(
                torch.cuda.current_device()
            )
            matching_results["score_mat"] = dist_d
        outputs_desc["matching_results"] = matching_results
        ordered_match = torch.arange(outputs_desc["a"]["descriptors"].shape[0]).to(
            torch.cuda.current_device()
        )
        perc = (
            sum(
                matching_results["matches0"]
                == ordered_match * (matching_results["matches0"] > -1)
            )
            / len(ordered_match)
        ).item()
        outputs_desc["correct_percentage"] = perc
        return outputs_desc
