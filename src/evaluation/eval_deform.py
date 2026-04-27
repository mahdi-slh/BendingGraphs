import torch
import open3d as o3d
import numpy as np
from torch_geometric.data import Batch
# from utils.pyFM import mesh
from utils.visualize_graph import (
    show_from_center,
    show_from_side,
    show_from_top,
    show_from_new,
)

from evaluation.evaluator import register_pair
from sklearn.neighbors import NearestNeighbors
from utils.pyFM.functional import FunctionalMapping

from collections import Counter

import matplotlib


matplotlib.use("TkAgg")



def test(
    settings, net, test_data, tb_writer=None, epoch=0, visualize=True, full_eval=False
):

    net.eval()

    results_sum = Counter({})
    results_ctr = 0
    surface = 'deform'

    for data_frame, meta in test_data:
        results_sum = results_sum + Counter(
            register_pair(net, data_frame, meta, visualize, tb_writer, epoch, full_eval,surface =surface)
        )
        results_ctr += 1

    results_dict = dict(results_sum)
    if results_ctr > 1:
        for k in results_dict.keys():
            results_dict[k] = results_dict[k] / results_ctr

    msg = "".join(
        [" {}: {:.5f},".format(k, v) for k, v in sorted(results_dict.items())]
    )
    print(msg)

    with open("../outputs/metrics.txt", "w") as f:
        f.write(msg)
    return results_dict


class EvaluatorDeform(Evaluator):

    def __init__(self , tb_writer=None, visualize=True, full_eval=False):
        surface='deform'
        super().__init__(surface = surface, tb_writer=tb_writer, visualize=visualize, full_eval=full_eval)
    
    def evaluate_surface(self, dict_predictions, meta, epoch):

        eval_results = {}

        g1_seeds = meta["seeds_a"]
        g2_seeds = meta["seeds_p"]

        g1_desc = dict_predictions["a"]["descriptors"]
        g2_desc = dict_predictions["p"]["descriptors"]

        seed_a = g1_seeds.cpu().numpy()
        seed_p = g2_seeds.cpu().numpy()

        seed_a_o3d = o3d.geometry.PointCloud()
        seed_a_o3d.points = o3d.utility.Vector3dVector(seed_a)

        seed_p_o3d = o3d.geometry.PointCloud()
        seed_p_o3d.points = o3d.utility.Vector3dVector(seed_p)


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
            self.add_image_tb(dict_predictions["matching_results"]["score_4ot"], "before ot", epoch
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


        seed_index_a = meta["seed_index_a"]
        seed_index_p = meta["seed_index_p"]
        full_points_a = meta["full_points_a"]
        full_points_p = meta["full_points_p"]
        # dij_a = meta['dij_a']
        # dij_p = meta['dij_p']
        mesh_a = meta["mesh_a"]
        mesh_p = meta["mesh_p"]
        mesh_a_geod = meta.get("mesh_a_geod", None)

        seed_index_a = np.asarray(seed_index_a)
        seed_index_p = np.asarray(seed_index_p)

        nbrs_knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nbrs_knn.fit(seed_a)
        dist_a, groups_a = nbrs_knn.kneighbors(full_points_a, return_distance=True)
        matches0_full = seed_index_p[ind_a[groups_a, 1]]
        too_far_vis_ind = (dist_a > 0.04).squeeze()

        nbrs_knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nbrs_knn.fit(seed_p)
        groups_p = nbrs_knn.kneighbors(full_points_p, return_distance=False)
        matches1_full = seed_index_a[ind_p[groups_p, 1]]

        ind_a_full = np.zeros([full_points_a.shape[0], 2], np.int)
        ind_a_full[:, 0] = np.arange(full_points_a.shape[0])
        ind_a_full[:, 1] = matches0_full.squeeze()
        # a_file = open("matchesfull.txt", "w")
        # for row in matches0_full:
        #     np.savetxt(a_file, row, fmt='%d')

        # a_file.close()

        if mesh_a_geod is None:
            mesh_a_geod = mesh_a.get_geodesic(verbose=False)

        acc_coarse = eval.accuracy(
            matches0_full.squeeze(),
            np.arange(ind_a_full.shape[0]),
            mesh_a_geod,
            sqrt_area=np.sqrt(mesh_a.area),
        )

        # mesh_a.vertlist *= 10
        # mesh_a_geod_scaled =  mesh_a.get_geodesic(verbose=False)
        # acc_coarse_scaled = eval.accuracy(matches0_full.squeeze(), np.arange(ind_a_full.shape[0]), mesh_a_geod_scaled, sqrt_area=np.sqrt(mesh_a.area))

        coarse_matches = np.asarray(
            [seed_index_p[ind_a[:, 1]], seed_index_p[ind_a[:, 1]]]
        ).transpose()
        fine_corres = self.coarse_to_fine(
            mesh_a=mesh_a, mesh_p=mesh_p, initial_corres=coarse_matches
        )
        acc_fine = eval.accuracy(
            fine_corres,
            np.arange(fine_corres.shape[0]),
            mesh_a_geod,
            sqrt_area=np.sqrt(mesh_a.area),
        )
        print(acc_coarse, acc_fine)

        """Dense Mesh P"""
        cols_p = np.zeros([full_points_p.shape[0], 3], dtype="float32")
        cols_p = full_points_p - np.min(full_points_p, 0)
        cols_p = cols_p / np.max(cols_p, 0)
        mesh_p_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_p),
            o3d.utility.Vector3iVector(mesh_p.facelist),
        )
        mesh_p_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_p)

        """Patch correspondence Mesh A"""
        cols_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        cols_a = cols_p[matches0_full.squeeze(), :]
        # cols_a [too_far_vis_ind,:]=[0.8,0.8,0.8]
        mesh_a_coarse_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_coarse_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_a)
        mesh_a_coarse_o3d = self.split_triangles(
            mesh_a_coarse_o3d, too_far_vis_ind, [0.8, 0.8, 0.8]
        )

        """Patch Confidence"""
        cols_conf_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        cols_conf_a = np.tile(conf_shape_a[:, ind_p[groups_p, 1]].reshape(-1, 1), 3)

        cols_conf_a = cols_conf_a / (np.max(cols_conf_a) + 0.001)  # Normalization
        cols_conf_a[:, 0] = 1 - cols_conf_a[:, 1]
        cols_conf_a[:, 2] = 0

        # cmap = plt.get_cmap('hot')
        # cols_conf_a = cols_conf_a = cmap(1-conf_shape_a[:,ind_p[groups_p, 1]].reshape(-1))[:,:3]*0.8
        cols_conf_a[too_far_vis_ind, :] = [0.8, 0.8, 0.8]
        mesh_a_conf_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_conf_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_conf_a)
        mesh_a_conf_o3d = self.split_triangles(
            mesh_a_conf_o3d, too_far_vis_ind, [0.8, 0.8, 0.8]
        )

        """FM Mesh A"""
        cols_fm_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        cols_fm_a = cols_p[fine_corres.squeeze(), :]
        mesh_a_fm_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_fm_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_fm_a)

        """MESH A GT"""
        cols_gt_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        cols_gt_a = cols_p[np.arange(full_points_a.shape[0]).squeeze(), :]
        mesh_a_gt_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_gt_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_gt_a)

        """PATCH  ERROR MAP MESH A"""
        cols_err_patch_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        col = (((cols_gt_a - cols_a) ** 2).sum(1)) ** 0.5
        cols_err_patch_a[:, 0] = col / 0.2  # /col.max()
        cols_err_patch_a[:, 1] = 1 - col / 0.2  # /col.max()
        cols_err_patch_a[:, 2] = 0
        cols_err_patch_a[too_far_vis_ind, :] = [0.0, 0.0, 0.0]
        mesh_a_patch_err_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_patch_err_o3d.vertex_colors = o3d.utility.Vector3dVector(
            cols_err_patch_a
        )
        mesh_a_patch_err_o3d = self.split_triangles(
            mesh_a_patch_err_o3d, too_far_vis_ind, [0, 0, 0]
        )

        """ERROR MAP DENSE MESH A"""
        cols_err_a = np.zeros([full_points_a.shape[0], 3], dtype="float32")
        cols_err_a = cols_err_a
        col = (((cols_gt_a - cols_fm_a) ** 2).sum(1)) ** 0.5
        cols_err_a[:, 0] = col / 0.2  # /col.max()
        cols_err_a[:, 1] = 1 - col / 0.2  # /col.max()
        cols_err_a[:, 2] = 0
        mesh_a_err_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(full_points_a),
            o3d.utility.Vector3iVector(mesh_a.facelist),
        )
        mesh_a_err_o3d.vertex_colors = o3d.utility.Vector3dVector(cols_err_a)

        list_vis = [
            mesh_p_o3d,
            mesh_a_coarse_o3d,
            mesh_a_conf_o3d,
            mesh_a_patch_err_o3d,
            mesh_a_fm_o3d,
            mesh_a_gt_o3d,
            mesh_a_err_o3d,
        ]

        for ctr, obj in enumerate(list_vis):
            self.save_mesh(obj, "{}_{}_{}".format(scene_a, scene_p, ctr))
            if self.visualize == True:
                show_from_side([obj])
        if self.visualize == True:
            for ctr, obj in enumerate(list_vis):
                obj.translate([ctr, 0, 0])
            o3d.visualization.draw_geometries(list_vis)
            # o3d.visualization.draw_geometries([mesh_p_o3d.translate(
            # [-2, 0, 0]), mesh_a_coarse_o3d.translate([-1, 0, 0]),mesh_a_conf_o3d, mesh_a_fm_o3d.translate([1, 0, 0]), mesh_a_gt_o3d.translate([2, 0, 0]),mesh_a_err_o3d.translate([3, 0, 0])])

        if self.visualize:
            seed_a_o3d = o3d.geometry.PointCloud()
            seed_a_o3d.points = o3d.utility.Vector3dVector(seed_a)

            err_a_o3d = o3d.geometry.PointCloud()
            err_a_o3d.points = o3d.utility.Vector3dVector(seed_a)
            if seed_p.shape == seed_a.shape:
                err_a_o3d.colors = o3d.utility.Vector3dVector(p_dif)

            seed_p_o3d = o3d.geometry.PointCloud()
            seed_p_o3d.points = o3d.utility.Vector3dVector(seed_p)
            cols_p = np.zeros([seed_p.shape[0], 3], dtype="float32")
            cols_p = seed_p - np.min(seed_p, 0)
            cols_p = cols_p / np.max(cols_p, 0)
            seed_p_o3d.colors = o3d.utility.Vector3dVector(cols_p)

            cols_a = np.zeros([seed_a.shape[0], 3], dtype="float32")
            cols_a[ind_a[:, 0], :] = cols_p[ind_a[:, 1], :]
            seed_a_o3d.colors = o3d.utility.Vector3dVector(cols_a)

            seed_a_bij_o3d = o3d.geometry.PointCloud()
            seed_a_bij_o3d.points = o3d.utility.Vector3dVector(seed_a)
            cols_a_bij = np.zeros([seed_a.shape[0], 3], dtype="float32")
            cols_a_bij[ind_a_bijective[:, 0], :] = cols_p[ind_a_bijective[:, 1], :]
            seed_a_bij_o3d.colors = o3d.utility.Vector3dVector(cols_a_bij)

            err_a_bij_o3d = o3d.geometry.PointCloud()
            err_a_bij_o3d.points = o3d.utility.Vector3dVector(seed_a)
            err_a_bij_col = np.zeros(p_dif.shape)
            err_a_bij_col[bijective, :] = p_dif[bijective, :]
            err_a_bij_o3d.colors = o3d.utility.Vector3dVector(err_a_bij_col)

            if "conf0" in dict_predictions["matching_results"].keys():
                conf_a_o3d = o3d.geometry.PointCloud()
                conf_a_o3d.points = o3d.utility.Vector3dVector(seed_a)
                # conf_shape_a = conf_shape_a-np.min(conf_shape_a)
                # conf_shape_a = conf_shape_a/np.max(conf_shape_a)
                conf_a_col = np.tile(conf_shape_a.reshape(-1, 1), 3)
                conf_a_col[:, 1] = 1 - conf_a_col[:, 0]
                conf_a_col[:, 2] = 0
                conf_a_o3d.colors = o3d.utility.Vector3dVector(conf_a_col)

                if self.visualize == True:
                    o3d.visualization.draw_geometries(
                        [
                            seed_p_o3d.translate([-4, 0, 0]),
                            seed_a_o3d.translate([-2, 0, 0]),
                            seed_a_bij_o3d,
                            err_a_o3d.translate([2, 0, 0]),
                            err_a_bij_o3d.translate([4, 0, 0]),
                            conf_a_o3d.translate([6, 0, 0]),
                        ]
                    )
        
        if self.full_eval:
            eval_results.update({"acc_fine": acc_fine, "acc_coarse": acc_coarse})
        # seed_a_o3d.colors = o3d.utility.Vector3dVector(np.tile(conf_a, [1, 3]))
        # seed_p_o3d.colors = o3d.utility.Vector3dVector(np.tile(prop_p, [1, 3]))
        # o3d.visualization.draw_geometries(
        #     [seed_a_o3d.translate([1, 0, 0]), seed_p_o3d])

        return eval_results

    def coarse_to_fine(self, mesh_a, mesh_p, initial_corres):

        process_params = {
            "n_ev": (35, 35),  # Number of eigenvalues on source and Target
            "landmarks": np.asarray(initial_corres),  # loading 5 landmarks
            "subsample_step": 5,  # In order not to use too many descriptors
            "descr_type": "WKS",  # WKS or HKS
        }

        corres = FunctionalMapping(mesh_a, mesh_p)
        corres.preprocess(**process_params, verbose=True)
        fit_params = {
            "descr_mu": 1e0,
            "lap_mu": 1e-3,
            "descr_comm_mu": 1e-1,
            "orient_mu": 0,
        }

        corres.fit(**fit_params, verbose=True)

        FM = corres.FM
        p2p = corres.p2p
        return p2p


    def split_triangles(self, mesh, too_far_vis_ind, invalid_color: list = [0, 0, 0]):
        from copy import deepcopy

        """
        Split the mesh in independent triangles    
        """
        triangles = np.asarray(mesh.triangles).copy()
        vertices = np.asarray(mesh.vertices).copy()
        colors = np.asarray(mesh.vertex_colors).copy()
        triangles_3 = np.zeros_like(triangles)
        vertices_3 = np.zeros((len(triangles) * 3, 3), dtype=vertices.dtype)
        colors_3 = np.zeros((len(triangles) * 3, 3), dtype=colors.dtype)

        for index_triangle, t in enumerate(triangles):
            index_vertex = index_triangle * 3
            vertices_3[index_vertex] = vertices[t[0]]
            vertices_3[index_vertex + 1] = vertices[t[1]]
            vertices_3[index_vertex + 2] = vertices[t[2]]

            if too_far_vis_ind[t[0]] or too_far_vis_ind[t[1]] or too_far_vis_ind[t[2]]:
                colors_3[index_vertex] = invalid_color
                colors_3[index_vertex + 1] = invalid_color
                colors_3[index_vertex + 2] = invalid_color
            else:
                clr = (colors[t[0]] + colors[t[1]] + colors[t[2]]) / 3
                colors_3[index_vertex] = colors[t[0]]
                colors_3[index_vertex + 1] = colors[t[1]]
                colors_3[index_vertex + 2] = colors[t[2]]

            triangles_3[index_triangle] = np.arange(index_vertex, index_vertex + 3)

        mesh_return = deepcopy(mesh)
        mesh_return.triangles = o3d.utility.Vector3iVector(triangles_3)
        mesh_return.vertices = o3d.utility.Vector3dVector(vertices_3)
        mesh_return.vertex_colors = o3d.utility.Vector3dVector(colors_3)
        return mesh_return

    def save_mesh(self, mesh_o3d, obj_name):
        o3d.io.write_triangle_mesh("../outputs/{}.ply".format(obj_name), mesh_o3d)