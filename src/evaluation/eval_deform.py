"""Deformable-shape evaluator (FAUST / SMAL / TOSCA / SHREC).

This used to be a sprawling free-function module that drifted from the
hierarchy ``Evaluator -> EvaluatorRigid``.  We now provide
``EvaluatorDeform`` mirroring that shape: ``eval(test_data, net, epoch)``
returns a metrics dict that the trainer can log.  Visual / mesh-geodesic
evaluation modes are kept as opt-ins.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from sklearn.neighbors import NearestNeighbors

from evaluation.evaluator import Evaluator
from evaluation.eval_utils import chamfer_distance


class EvaluatorDeform(Evaluator):
    """Geodesic-error evaluator for deformable shape matching.

    Reports the same correspondence-based metrics as the original rigid
    branch (bijective rate, geodesic average, average after one OT round)
    plus optional Chamfer distance on the matched seeds.
    """

    def __init__(self, tb_writer=None, visualize: bool = False, full_eval: bool = False):
        super().__init__(surface="deform", tb_writer=tb_writer,
                         visualize=visualize, full_eval=full_eval)

    def evaluate_surface(self, dict_predictions, meta, epoch):
        eval_results: dict = {}

        size_a = dict_predictions["a"]["descriptors"].shape[0]
        size_p = dict_predictions["p"]["descriptors"].shape[0]

        ind_a = np.zeros([size_a, 2], dtype=int)
        ind_p = np.zeros([size_p, 2], dtype=int)

        ind_a[:, 0] = np.arange(size_a)
        ind_a[:, 1] = (
            dict_predictions["matching_results"]["matches0"].detach().cpu().numpy()
        )
        ind_a_valid = ind_a[ind_a[:, 1] > -1, :]

        ind_p[:, 0] = np.arange(size_p)
        ind_p[:, 1] = (
            dict_predictions["matching_results"]["matches1"].detach().cpu().numpy()
        )
        ind_p_valid = ind_p[ind_p[:, 1] > -1, :]

        ind_a_ot = np.copy(ind_a)
        if "matches0_ot" in dict_predictions["matching_results"].keys():
            ind_a_ot[:, 1] = (
                dict_predictions["matching_results"]["matches0_ot"]
                .detach()
                .cpu()
                .numpy()
            )

        # Geodesic distance matrix between seeds (computed during the
        # dataset's preprocessing step) is the ground-truth correspondence
        # signal for FAUST/SMAL/TOSCA.
        dist_mat = meta["dist_mat"]
        if isinstance(dist_mat, torch.Tensor):
            dist_np = dist_mat.detach().cpu().numpy()
        else:
            dist_np = np.asarray(dist_mat)

        avg_dist_geo = float(dist_np[ind_a[:, 0], ind_a[:, 1]].mean())
        avg_dist_geo_ot = float(dist_np[ind_a_ot[:, 0], ind_a_ot[:, 1]].mean())

        bijective = ind_p[ind_a[:, 1], 1] == ind_a[:, 0]
        rate = float(np.sum(bijective)) / float(len(bijective)) if len(bijective) else 0.0

        eval_results.update({
            "bij_rate": rate,
            "number_valid": int(len(ind_a_valid)),
            "geod_avg": avg_dist_geo,
            "geod_avg_ot": avg_dist_geo_ot,
        })

        if "score_4ot" in dict_predictions["matching_results"].keys():
            self.add_image_tb(
                dict_predictions["matching_results"]["score_4ot"], "before_ot", epoch
            )
        if "score_mat" in dict_predictions["matching_results"].keys():
            self.add_image_tb(
                np.abs(dict_predictions["matching_results"]["score_mat"]),
                "after_ot",
                epoch,
            )
        if "dist_mat" in meta.keys():
            self.add_image_tb(meta["dist_mat"].squeeze(), "dist_mat", epoch)

        # Optional: Chamfer distance on the matched seeds.  This is cheap
        # (200 points each) and gives a coarse alignment proxy.
        try:
            seeds_a = meta["seeds_a"].detach().cpu().numpy()
            seeds_p = meta["seeds_p"].detach().cpu().numpy()
            cd = chamfer_distance(seeds_p[ind_a[:, 1]], seeds_a[ind_a[:, 0]])
            eval_results["chamfer"] = float(cd)
        except Exception:
            pass

        return eval_results
