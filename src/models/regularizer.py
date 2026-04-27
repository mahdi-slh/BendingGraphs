#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:53:01 2021

@author: sc
"""

import torch
from torch import nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, Sequential
from configs import FEATURE_SIZE
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric


class GraphNodeGrad(MessagePassing):
    def __init__(self, aggr: str = "add", flow: str = "source_to_target"):
        super().__init__(flow=flow, aggr=aggr)
        # self.flow = flow
        # self.aggr = aggr
        # assert self.aggr in ['add', 'mean', 'max', None]
        # pass

    def reset_parameters(self):
        # torch_geometric.utils.get_laplacian
        # make initial to identity
        # gcn.weight = torch.ones_like(gcn.weight)
        # gcn.bias   = torch.zeros_like(gcn.bias)
        pass

    def forward(
        self, x: Tensor, edge_index: Adj, node_weight: OptTensor = None
    ) -> Tensor:
        out = self.propagate(edge_index, x=x)

        return out

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        out = x_i - x_j

        return out


class WeightedGCN(MessagePassing):
    def __init__(
        self,
        feature_size: int,
        aggr: str = "max",
        flow: str = "source_to_target",
        add_self_loops: bool = True,
    ):
        super().__init__(flow=flow, aggr=aggr)
        self.add_self_loops = add_self_loops
        self.gru = nn.GRU(feature_size, feature_size, 1)

    def reset_parameters(self):
        pass

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        node_weight: Tensor,
        improved=False,
        edge_weight=None,
    ) -> Tensor:
        device = x.device
        fill_value = 2.0 if improved else 1.0
        if self.add_self_loops:
            num_nodes = x.size(self.node_dim)
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes
            )
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        out = self.propagate(edge_index, x=x, w=node_weight)
        h0 = x.contiguous()
        out, h0 = self.gru(out, h0)
        return out

    def message(
        self, x_i: Tensor, x_j: Tensor, w_j: Tensor, edge_index_i: Tensor
    ) -> Tensor:
        out = x_j * w_j
        return out


class Regularizer(nn.Module):
    def __init__(self, dim, num_layer: int):
        super().__init__()

        input_ = "x, edge_index, node_weight"
        output_ = "x, edge_index, node_weight->x"

        self.num_layer = num_layer

        self.use_gru = True

        # self.gru = nn.GRU(FEATURE_SIZE, FEATURE_SIZE, 1)
        self.gcn = WeightedGCN(dim, flow="target_to_source", add_self_loops=False)
        # if self.use_gru:

        # else:
        #     layers=list()
        #     for i in range(num_layer):
        #         layers.append(
        #             (WeightedGCN(FEATURE_SIZE, FEATURE_SIZE, flow='target_to_source'), output_)
        #         )
        #     # if i+1 != num_layer:
        #     #     layers.append(nn.ReLU(inplace=True))

        #     self.gcn = Sequential(input_,layers)

    def forward(self, data):
        desc0 = data["x0"]  # [b, d, n]
        desc1 = data["x1"]  # [b, d, n]
        conf0 = data["conf0"]  # [b, n]
        conf1 = data["conf1"]  # [b, n]
        shape0 = data["shape0"].to(torch.cuda.current_device())
        shape1 = data["shape1"].to(torch.cuda.current_device())

        desc0 = desc0.permute(0, 2, 1)  # [b, n, d]
        desc1 = desc1.permute(0, 2, 1)
        conf0 = conf0.unsqueeze(-1)  # [b,n,d]
        conf1 = conf1.unsqueeze(-1)

        # conf0 = conf0.exp() # from log to [0,1]
        # conf1 = conf1.exp() # from log to [0,1]

        # if self.use_gru:
        #     h0 = desc0
        #     h1 = desc1
        #     for _ in range(self.num_layer):
        #         gcn_desc0 = self.gcn(h0, shape0.edge_index, conf0)
        #         gcn_desc0 = gcn_desc0.unsqueeze(0)
        #         h0 = h0.unsqueeze(0)
        #         gcn_desc0, h0 = self.gru(gcn_desc0, h0)
        #         gcn_desc0 = gcn_desc0.squeeze(0)
        #         h0 = h0.squeeze(0)

        #         gcn_desc1 = self.gcn(h1, shape1.edge_index, conf1)
        #         gcn_desc1 = gcn_desc1.unsqueeze(0)
        #         h1 = h1.unsqueeze(0)
        #         gcn_desc1, h1 = self.gru(gcn_desc1, h1)
        #         gcn_desc1 = gcn_desc1.squeeze(0)
        #         h1 = h1.squeeze(0)
        # else:
        for _ in range(self.num_layer):
            desc0 = self.gcn(desc0, shape0.edge_index, conf0)
            desc1 = self.gcn(desc1, shape1.edge_index, conf1)
        desc0 = desc0.permute(0, 2, 1)  # [b, d, n]
        desc1 = desc1.permute(0, 2, 1)

        return desc0, desc1


if __name__ == "__main__":
    n_pts = 32
    dim_d = 64
    edge_index = torch.randint(0, n_pts - 1, [2, 300])
    weight = torch.rand([n_pts])
    wgcn = WeightedGCN(dim_d, dim_d)
    x = torch.rand(n_pts, dim_d)
    wgcn(x, edge_index, weight)

    tmp = Sequential(
        "x,edge_index,weight",
        [
            (WeightedGCN(dim_d, dim_d), "x,edge_index,weight->x"),
            nn.ReLU(inplace=True),
            (WeightedGCN(dim_d, dim_d), "x,edge_index,weight->x"),
        ],
    )

    net = GraphNodeGrad(dim_d, dim_d)
    net(x, edge_index)
