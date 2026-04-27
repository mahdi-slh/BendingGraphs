# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import time
from copy import deepcopy
from pathlib import Path
from sinkhorn_transformer import SinkhornTransformerLM

import torch
from torch import nn
from models.ot_pytorch import sink, dmat

# import ot
# import neuralnet_pytorch as nnt
# import ot.plot
from torch_geometric.nn import max_pool_x, TAGConv, SGConv, PANConv, GCNConv
from torch_geometric.nn import nearest
from models.regularizer import GraphNodeGrad
from utils.checks import *
from models.regularizer import Regularizer
from models.model_utils import to_freqs, log_freq
from utils.utils import pytorch_count_params


def MLP(channels: list, do_bn=True):
    """Multi-layer perceptron"""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts):
    """Normalize keypoints locations based on image image_shape"""

    min_bound = torch.min(kpts, dim=1, keepdim=True)[0]
    max_bound = torch.max(kpts, dim=1, keepdim=True)[0]

    return (kpts - min_bound) / (max_bound - min_bound)


class KeypointEncoder(nn.Module):
    """Joint encoding of visual appearance and location using MLPs"""

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([21] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        if kpts.ndim == 2:
            kpts = kpts.unsqueeze(0)
        inputs = [kpts.transpose(1, 2)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / dim**0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bdhm->bdhn", prob, value), prob


class MultiHeadedAttention(nn.Module):
    """Multi-head attention to increase model expressivitiy"""

    def __init__(self, num_heads: int, d_model: int, name: str):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        if name == "cross":
            self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        elif name == "self":
            self.merge = TAGConv(d_model, d_model)
        self.name = name
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        if self.name == "cross":
            batch_dim = query.f.size(0)
            query_val, key_val, value_val = [
                l(gr.f).view(batch_dim, self.dim, self.num_heads, -1)
                for l, gr in zip(self.proj, (query, key, value))
            ]
            x, prob = attention(query_val, key_val, value_val)
            self.prob.append(prob)
            return self.merge(
                x.contiguous().view(batch_dim, self.dim * self.num_heads, -1)
            ).view(batch_dim, self.dim * self.num_heads, -1)

        elif self.name == "self":
            batch_dim = query.f.size(0)
            query_val, key_val, value_val = [
                l(gr.f.transpose(1, 2).squeeze(), gr.edge_index, gr.edge_attr).view(
                    batch_dim, self.dim, self.num_heads, -1
                )
                for l, gr in zip(self.proj, (query, key, value))
            ]

            x, prob = attention(query_val, key_val, value_val)
            self.prob.append(prob)
            return self.merge(
                x.contiguous()
                .view(batch_dim, self.dim * self.num_heads, -1)
                .transpose(1, 2)
                .squeeze(),
                query.edge_index,
                query.edge_attr,
            ).view(batch_dim, self.dim * self.num_heads, -1)


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, name: str):
        super().__init__()

        self.attn = MultiHeadedAttention(num_heads, feature_dim, name)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x.f, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4, name) for name in layer_names]
        )
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == "cross":
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0_val, desc1_val = (desc0.f + delta0), (desc1.f + delta1)
        return desc0_val, desc1_val


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int, with_bin: bool = True):
    """Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)
    norm = -(ms + ns).log()

    if with_bin:
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat(
            [torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1
        )
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    else:
        couplings = scores
        log_mu = norm.expand(m)
        log_nu = norm.expand(n)
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end
    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold
    The correspondence ids use -1 to indicate non-matching points.
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763
    """

    default_config = {
        "descriptor_dim": 64,
        "weights": "indoor",
        "keypoint_encoder": [21, 32, 64, 64],
        "GNN_layers": ["self", "cross"] * 5,
        "sinkhorn_iterations": 10,
        "match_threshold": 0.0000,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config["descriptor_dim"], self.config["keypoint_encoder"]
        )

        self.gnn = AttentionalGNN(
            self.config["descriptor_dim"], self.config["GNN_layers"]
        )

        self.final_proj = nn.Conv1d(
            self.config["descriptor_dim"],
            self.config["descriptor_dim"],
            kernel_size=1,
            bias=True,
        )

        self.proj_1d = nn.Conv1d(
            self.config["descriptor_dim"],
            1,
            kernel_size=1,
            bias=True,
        )

        self.with_bin = False
        self.n_got = 2
        self.n_gru = 1
        self.regularizer = Regularizer(
            self.config["descriptor_dim"], num_layer=self.n_gru
        )  # 4

        # self.soft_gcns = [GCNConv(3,3).cuda(),GCNConv(3,3).cuda(),GCNConv(3,3).cuda(),GCNConv(3,3).cuda()]

        # DE_SEQ_LEN = 100
        # EN_SEQ_LEN = 100
        # self.enc = SinkhornTransformerLM(
        #     num_tokens = 20000,
        #     dim = 512,
        #     depth = 6,
        #     heads = 8,
        #     bucket_size = 128,
        #     max_seq_len = DE_SEQ_LEN,
        #     reversible = True,
        #     return_embeddings = True
        # ).cuda()
        # self.losskl=torch.nn.KLDivLoss()
        # self.dec = SinkhornTransformerLM(
        #     num_tokens = 20000,
        #     dim = 512,
        #     depth = 6,
        #     causal = True,
        #     bucket_size = 10,
        #     max_seq_len = EN_SEQ_LEN,
        #     receives_context = True,
        #     context_bucket_size = 10,  # context key / values can be bucketed differently
        #     reversible = True
        # ).cuda()

        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("bin_score", bin_score)
        self.graph_grad = GraphNodeGrad()

        # assert self.config['weights'] in ['indoor', 'outdoor']
        # path = Path(__file__).parent
        # path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        # self.load_state_dict(torch.load(path))
        # print('Loaded SuperGlue model (\"{}\" weights)'.format(
        #     self.config['weights']))
        # self.loss_bce = torch.nn.BCELoss()

        # print("===trainable parameters===")
        # print("kenc: {}".format(pytorch_count_params(self.kenc)))
        # print("[unused]gnn: {}".format(pytorch_count_params(self.gnn)))
        # print("[unused]proj_1d: {}".format(pytorch_count_params(self.proj_1d)))
        # print("final_proj: {}".format(pytorch_count_params(self.final_proj)))
        # print(
        #     "got gated msg aggregation: {}".format(
        #         pytorch_count_params(self.regularizer)
        #     )
        # )
        # print("graph_grad: {}".format(pytorch_count_params(self.graph_grad)))

    def forward(self, data, meta):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        # t = time.time()

        desc0, desc1 = data["descriptors0"].unsqueeze(0).permute(0, 2, 1), data[
            "descriptors1"
        ].unsqueeze(0).permute(0, 2, 1)
        if data["keypoints0"].ndim == 2:
            data["keypoints0"].unsqueeze_(0)
            data["keypoints1"].unsqueeze_(0)

        kpts0, kpts1 = data["keypoints0"].permute(0, 2, 1), data["keypoints1"].permute(
            0, 2, 1
        )

        # desc0 = desc0.transpose(0, 1)
        # desc1 = desc1.transpose(0, 1)
        kpts0 = torch.reshape(kpts0, (1, -1, 3))
        kpts1 = torch.reshape(kpts1, (1, -1, 3))

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                "matches0": kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                "matches1": kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                "matching_scores0": kpts0.new_zeros(shape0)[0],
                "matching_scores1": kpts1.new_zeros(shape1)[0],
                "skip_train": True,
            }

        # file_name = data["file_name"]
        all_matches = data["all_matches"]  # shape=torch.Size([1, 87, 2])

        dist_mat = meta["dist_mat"].to(torch.cuda.current_device()).clone()
        vicinity_mat = dist_mat.detach().clone()
        if vicinity_mat.ndim == 2:
            vicinity_mat = vicinity_mat.unsqueeze(0)
        # vicinity_mat = torch.clamp(vicinity_mat, 0, 5)
        # vicinity_mat = (5 - vicinity_mat) / 5

        vicinity_mat = torch.clamp(vicinity_mat, 0, 0.1)
        vicinity_mat = (0.1 - vicinity_mat) *10

        shape0 = meta["shape_a"].to(torch.cuda.current_device())
        shape1 = meta["shape_p"].to(torch.cuda.current_device())

        # # Keypoint normalization.
        # kpts0 = normalize_keypoints(kpts0)
        # kpts1 = normalize_keypoints(kpts1)

        # Keypoint MLP encoder.
        # logger.info(f'before pos enc {time.time()-t}')
        # t = time.time()
        # xx = log_freq(shape0.x,3,1)
        shape0.f = desc0 + self.kenc(log_freq(shape0.x, 3, 1))
        shape1.f = desc1 + self.kenc(log_freq(shape1.x, 3, 1))
        # if check_valid(shape0.f):
        #     print('shit')
        # if check_valid(shape1.f):
        #     print('shit')
        # Multi-layer Transformer network.

        # logger.info(f'before gnn {time.time()-t}')
        # t = time.time()
        # shape0.f , shape1.f = self.gnn(shape0, shape1)
        # if check_valid(shape0.f):
        #     print('shit')
        # if check_valid(shape1.f):
        #     print('shit')
        # Final MLP projection.
        # logger.info(f'before project {time.time()-t}')
        # t = time.time()
        mdesc0, mdesc1 = self.final_proj(shape0.f), self.final_proj(shape1.f)
        # if check_valid(mdesc0):
        #     print('shit')
        # if check_valid(mdesc1):
        #     print('shit')
        # mdesc0_1d, mdesc1_1d = self.proj_1d(desc0), self.proj_1d(desc1)
        # mdesc0, mdesc1 = desc0, desc1

        # Compute matching descriptor distance.
        # scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        # funcc()
        scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        scores = scores / self.config["descriptor_dim"] ** 0.5
        scores_before_ot = scores.clone().float().squeeze(0)
        # # Run the optimal transport.
        # # logger.info(f'before opt trans {time.time()-t}')
        # # t = time.time()

        # # dist_a, scores_mat = sink(scores.float().squeeze(0), reg=5, cuda=True)
        # # scores_mat = scores_mat.unsqueeze(0)
        # scores = log_optimal_transport(
        #     scores, self.bin_score, iters=self.config["sinkhorn_iterations"],with_bin=self.with_bin
        # )

        # # if check_valid(scores):
        # #     print('shit')

        # # logger.info(f'before final stuff {time.time()-t}')
        # # t = time.time()
        # # Get the matches with score above "match_threshold".
        # if self.with_bin:
        #     max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        # else:
        #     max0, max1 = scores.max(2), scores.max(1)
        # # max0, max1 = scores[:, :, :].max(2), scores[:, :, :].max(1)
        # indices0, indices1 = max0.indices, max1.indices
        # # conf0,conf1= torch.log(-max0.values), torch.log(-max1.values)
        # # conf0,conf1= torch.exp(max0.values), torch.exp(max1.values)
        # conf0,conf1= max0.values, max1.values

        """regularizer"""
        for _ in range(self.n_got):
            scores = log_optimal_transport(
                scores,
                self.bin_score,
                iters=self.config["sinkhorn_iterations"],
                with_bin=self.with_bin,
            )
            if self.with_bin:
                max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            else:
                max0, max1 = scores.max(2), scores.max(1)
            # max0, max1 = scores[:, :, :].max(2), scores[:, :, :].max(1)
            indices0, indices1 = max0.indices, max1.indices
            conf0, conf1 = max0.values, max1.values

            ####
            mdesc0, mdesc1 = self.regularizer(
                {
                    "x0": mdesc0,
                    "x1": mdesc1,
                    "conf0": conf0,
                    "conf1": conf1,
                    "shape0": shape0,
                    "shape1": shape1,
                }
            )
            scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
            scores = scores / self.config["descriptor_dim"] ** 0.5

        # Last OT
        scores = log_optimal_transport(
            scores,
            self.bin_score,
            iters=self.config["sinkhorn_iterations"],
            with_bin=self.with_bin,
        )
        if self.with_bin:
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        else:
            max0, max1 = scores.max(2), scores.max(1)
        indices0, indices1 = max0.indices, max1.indices
        conf0, conf1 = max0.values, max1.values
        """""" """""" """"""
        conf0, conf1 = torch.exp(conf0), torch.exp(conf1)

        # mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        # mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = max0.values.exp()
        mscores1 = mscores0.gather(1, indices1)
        valid0 = mscores0 > self.config["match_threshold"]
        valid1 = valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        # check if indexed correctly
        # loss = []
        # for i in range(len(all_matches[0])):
        #     x = i  # all_matches[0][i][0]
        #     y = all_matches[0][i][0].detach(
        #     ).cpu().numpy().astype("int").item()
        #     # check batch size == 1 ?
        #     loss.append(-torch.log(scores[0][x][y].exp()))
        loss = []

        """"""
        # if True:
            # sparse
        log_score = -torch.log(scores.exp() + 0.000000001)  # TODO: this is eq. to -scores
        for id_a in range(vicinity_mat.shape[1]):
            for id_p in range(vicinity_mat.shape[2]):
                if vicinity_mat[0][id_a][id_p] > 0:
                    loss_index = (
                        log_score[0][id_a][id_p] * vicinity_mat[0][id_a][id_p]
                    )
                    loss.append(loss_index)
        loss = torch.mean(torch.stack(loss))
        # else:
            # # dense
            # log_score = -scores
            # loss = -(scores[vicinity_mat > 0].mean()) + (
            #     (1 - scores[vicinity_mat == 0].exp()).log().mean()
            # )

        if self.with_bin:
            scores_soft = torch.softmax(log_score[0, :-1, :-1], 1)
        else:
            scores_soft = torch.softmax(log_score[0], 1)

        loss_grad = 0
        y_ot = torch.einsum("nm,md->nd", scores_soft, kpts1.squeeze().float())
        y_grad = self.graph_grad(y_ot, shape0.edge_index, shape0.edge_attr)
        x_grad = self.graph_grad(shape0.x, shape0.edge_index, shape0.edge_attr)
        loss_grad = torch.mean(torch.abs(y_grad - x_grad))

        # y_ot = kpts1.squeeze()[indices0[0,:],:].clone().detach().float()

        # y=y_ot.clone()
        # y = torch.cat([y_ot,conf0.transpose(0,1)],1)

        """ post- """
        # for gcn in self.soft_gcns:
        #     y=gcn(y,shape0.edge_index)
        # y_ref = y
        # y_targ= kpts1.squeeze()[torch.argmin(dist_mat.squeeze(),1),:].clone().detach()
        # loss_ref = torch.abs(y_targ-y_ref).mean()
        # new_ind = nearest( y_ref,kpts1.squeeze().float())
        indices0_ot = indices0.clone()
        # indices0_ref = new_ind.view(indices0.shape)
        loss_ref = 0
        indices0_ref = indices0_ot

        # shape0.y_targ= kpts0.squeeze()[torch.argmin(dist_mat.squeeze(),0),:].detach()
        # new_ind = nearest( shape0.y_targ,kpts0.squeeze())
        # indices1 = new_ind.view(indices1.shape)
        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        loss_mean = loss
        loss_mean = torch.reshape(loss_mean, (1, -1))
        # loss_mean = nnt.metrics.emd_loss(scores[:,:100,:100],dist_mat,sinkhorn=True)
        # loss_mean = self.losskl(scores[:,:100,:100],dist_mat)
        # loss_mean =self.losskl(scores_mat.double(),dist_mat)
        # logger.info(f'the end----------------- {time.time()-t}')
        return {
            "matches0": indices0_ref[0],  # use -1 for invalid match
            "matches0_ot": indices0_ot[0],  # use -1 for invalid match
            "matches1": indices1[0],  # use -1 for invalid match
            "matching_scores0": mscores0[0],
            "matching_scores1": mscores1[0],
            "conf0": conf0,
            "conf1": conf1,
            "losses": {"loss_match": loss_mean, "loss_grad": loss_grad},
            "skip_train": False,
            "score_mat": scores.detach().squeeze(0).cpu().numpy(),
            "score_4ot": scores_before_ot.detach().squeeze().cpu().numpy(),
        }

        # scores big value or small value means confidence? log can't take neg value


class SelfAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim, 4) for _ in range(len(layer_names))]
        )
        self.names = layer_names

    def forward(self, desc):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            delta = layer(desc, desc)
            desc = desc + delta
        return desc
