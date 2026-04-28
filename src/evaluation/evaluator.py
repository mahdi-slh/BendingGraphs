
import os
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from collections import Counter

from sklearn.neighbors import NearestNeighbors

import configs as _configs

OUTPUTS_DIR = getattr(_configs, "OUTPUTS_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs",
))


class Evaluator:
    def __init__(self ,surface, tb_writer, visualize, full_eval):
        
        self.tb_writer=tb_writer
        self.visualize = visualize
        self.full_eval = full_eval
        self.surface = surface


        self.results_sum = Counter({})
        self.results_ctr = 0
    
    
    def eval(self,test_data,net,epoch):
        
        net.eval()
        for data_frame, meta in test_data:
            self.results_sum = self.results_sum + Counter(
                self.register_pair(net, data_frame, meta,epoch = epoch)
            )
            self.results_ctr += 1

        results_dict = dict(self.results_sum)
        if self.results_ctr > 1:
            for k in results_dict.keys():
                results_dict[k] = results_dict[k] / self.results_ctr

        msg = "".join(
            [" {}: {:.5f},".format(k, v) for k, v in sorted(results_dict.items())]
        )
        print(msg)

        try:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            with open(os.path.join(OUTPUTS_DIR, "metrics.txt"), "w") as f:
                f.write(msg)
        except OSError:
            pass
        return results_dict

    def add_image_tb(self, image, text, step=0):
        if self.tb_writer:

            if type(image).__name__ == "ndarray":
                image = torch.from_numpy(image)
            image_f = image.clone().float()
            max_image = image_f.max()
            min_image = image_f.min()
            if max_image - min_image > 0:

                image_f = (image - min_image) / (max_image - min_image)
            if image_f.ndim == 2:
                image_f.unsqueeze_(0)
            self.tb_writer.add_image(text, image_f, step)

    def evaluate_surface(self, dict_predictions, meta, epoch):
        # Subclasses (EvaluatorRigid / EvaluatorDeform) override this with
        # their full implementation.  We keep the body empty in the base
        # class to avoid runtime errors from the legacy code path.
        raise NotImplementedError(
            "Evaluator.evaluate_surface should be overridden by a subclass; "
            "use EvaluatorRigid or EvaluatorDeform instead."
        )

    def _legacy_evaluate_surface(self, dict_predictions, meta, epoch):
        eval_results = {}

        g1_seeds = meta["seeds_a"]
        g2_seeds = meta["seeds_p"]

        g1_desc = dict_predictions["a"]["descriptors"]
        g2_desc = dict_predictions["p"]["descriptors"]

        seed_a = g1_seeds.cpu().numpy()
        seed_p = g2_seeds.cpu().numpy()

        # conf_a = dict_results['a']['confidence'].cpu().numpy()
        # conf_p = dict_results['p']['confidence'].cpu().numpy()

        if "conf0" in dict_predictions["matching_results"].keys():
            conf_shape_a = dict_predictions["matching_results"]["conf0"].cpu().numpy()
            conf_shape_p = dict_predictions["matching_results"]["conf1"].cpu().numpy()

        scene_a = meta["scene_a"]  # meta.get('scene_a',None)
        scene_p = meta["scene_p"]  # meta.get('scene_p',None)

        # conf_p = conf_p-np.min(conf_p)
        # conf_p = conf_p/(np.max(conf_p)+0.1)

        # conf_a = conf_a-np.min(conf_a)
        # conf_a = conf_a/(np.max(conf_a)+0.1)

        # prop_a = dict_results['max_values'][0]
        # prop_p = dict_results['max_values'][1]

        # prop_p = prop_p-np.min(prop_p)
        # prop_p = conf_p/(np.max(prop_p)+0.1)

        # prop_a = prop_a-np.min(prop_a)
        # prop_a = prop_a/(np.max(prop_a)+0.1)

        size_a, size_p = g1_desc.shape[0], g2_desc.shape[0]

        ind_a = np.zeros([size_a, 2], dtype=int)
        ind_p = np.zeros([size_p, 2], dtype=int)

        ind_a[:, 0] = np.arange(size_a)
        ind_a[:, 1] = dict_predictions["matching_results"]["matches0"].detach().cpu().numpy()
        ind_a_valid = ind_a[ind_a[:, 1] > -1, :]

        ind_p[:, 0] = np.arange(size_p)
        ind_p[:, 1] = dict_predictions["matching_results"]["matches1"].detach().cpu().numpy()
        ind_p_valid = ind_p[ind_p[:, 1] > -1, :]

        ind_a_ot = np.copy(ind_a)

        if "matches0_ot" in dict_predictions["matching_results"].keys():
            ind_a_ot[:, 1] = (
                dict_predictions["matching_results"]["matches0_ot"].detach().cpu().numpy()
            )
        avg_dist_geo_ot = (meta["dist_mat"][ind_a_ot[:, 0], ind_a_ot[:, 1]]).mean().item()

        

        number_valid = len(ind_a_valid)
        avg_dist_geo = (meta["dist_mat"][ind_a[:, 0], ind_a[:, 1]]).mean().item()

        bijective = ind_p[ind_a[:, 1], 1] == ind_a[:, 0]

        ind_a_bijective = ind_a[bijective, :]
        rate = sum(bijective) / len(bijective)

        p_dif = meta["dist_mat"][ind_a[:, 0], ind_a[:, 1]].detach().cpu().numpy()
        p_dif = p_dif / np.max(p_dif)
        p_dif = np.tile(p_dif.reshape(-1, 1), 3)
        p_dif[:, 1] = 1 - p_dif[:, 0]
        p_dif[:, 2] = 0

        if "score_4ot" in dict_predictions["matching_results"].keys():
            self.add_image_tb( dict_predictions["matching_results"]["score_4ot"], "before ot", epoch
            )
        if "score_mat" in dict_predictions["matching_results"].keys():
            self.add_image_tb(
                np.abs(dict_predictions["matching_results"]["score_mat"]),
                "after ot",
                epoch,
            )
        if "dist_mat" in meta.keys():
            self.add_image_tb(meta["dist_mat"].squeeze(), "Dist mat", epoch)

        eval_results = {
            "bij_rate": rate,
            "number_valid": number_valid,
            "geod_avg": avg_dist_geo,
            "geod_avg_ot": avg_dist_geo_ot,
        }
        if self.surface == 'rigid':
            self.evaluate_rigid()
        elif self.surface == 'deform':
            self.evaluate_deform()


    def register_pair(self,model, data, meta,epoch):

        with torch.no_grad():

            inputs = {
                key: Batch.from_data_list(data[key], exclude_keys=["batch", "ptr"])
                for key in data.keys()
            }
            if torch.cuda.is_available():
                inputs = {
                    key: inputs[key].to(torch.cuda.current_device())
                    for key in inputs.keys()
                }
            output = model(inputs, meta)
            max_values = [
                scatter_mean(output[key]["probabilities"], inputs[key].batch, dim=0)
                .detach()
                .cpu()
                .numpy()
                for key in inputs.keys()
            ]
            output["max_values"] = max_values
            return self.evaluate_surface(
                dict_predictions = output, meta=meta, epoch=epoch
            )
        