#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:57:09 2021

@author: sc
"""
import torch
import open3d as o3d
import numpy as np
import copy
import os, glob
from tqdm import tqdm
from torch.utils.data import Dataset
from loaders.augmentations import random_perturb
from graph import (
    create_shape_graph,
    fps_points_random,
    create_local_graph,
    visualize_torch_graphs_full,
    visualize_torch_graphs_local,
)
from utils.pointcloud_utils import fps_points_random, fps_points, move_to_center
from configs import GRAPH_SIZE, GEODESIC_CUT
import open3d as o3d
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from utils.mesh_utils import (
    compute_geod,
    o3d_to_trimesh,
    get_dijkstra,
    normalize_unit_sphere,
    get_single_dijkstra,
)
from utils.visualize_graph import visualize_graph_shape
from utils.utils import vis_graph_with_same_idx

# from smpl_webuser.serialization import load_model
# import pickle as pkl
# from tqdm.contrib.concurrent import process_map
from utils.utils import set_random_seed


class Matching_Dataset(Dataset):
    def __init__(self, train: bool, mode: str, test_folder: str = "training/"):
        set_random_seed(2020)
        self.train = train
        self.mode = mode + "/"
        self.points_per_shape = 200  # if train else 200
        self.DEBUG_PLOT = False
        self.THREADS = 8
        self.test_train = "training/" if train else test_folder
        self.test_folder = test_folder

    def generate(self, mesh_a, mesh_p, name_a=None, name_p=None):
        mesh_a, mesh_p = normalize_unit_sphere([mesh_a, mesh_p])
        # mesh_a, mesh_p = move_to_center(mesh_a), move_to_center(mesh_p)

        pc_a = o3d.geometry.PointCloud()
        pc_p = o3d.geometry.PointCloud()

        pc_a.points = mesh_a.vertices
        pc_a_np = np.asarray(pc_a.points)

        pc_p.points = mesh_p.vertices
        pc_p_np = np.asarray(pc_p.points)

        """ process """
        if self.mode == "noise" and self.train:
            pc_a = random_perturb(pc_a, False, sigma=0.004, clip=0.004)
            pc_p = random_perturb(pc_a, False, sigma=0.004, clip=0.004)

        """ FPS sampling """
        seeds_a_np, indexes_fps_a = fps_points_random(
            pc_a_np, self.points_per_shape, return_indexes=True
        )
        if not self.train:
            seeds_p_np, indexes_fps_p = fps_points_random(
                pc_p_np, self.points_per_shape, return_indexes=True
            )
        else:
            indexes_fps_p = indexes_fps_a
            seeds_p_np = pc_p_np[indexes_fps_a, :]

        """calculate shortest path for the selected seeds"""
        # tmp = get_single_dijkstra(mesh_a,indexes_fps_a[0],GEODESIC_CUT)
        dijkstra_a = get_dijkstra(mesh_a)
        dijkstra_p = get_dijkstra(mesh_p)

        """create patch base on the seeds and the nearby points"""

        def generate_patches(index, dijkstra, pc, distance: float):
            patch_indexes = [np.where(dijkstra[i] < distance)[0] for i in index]
            patches = [pc.select_by_index(idxs) for idxs in patch_indexes]
            dij_local = [dijkstra[idxs, :][:, idxs] for idxs in patch_indexes]
            patches = list(zip(patches, dij_local))

            valid_points = [len(p[0].points) < len(pc.points) for p in patches]
            patches = [patches[i] for i in range(len(patches)) if valid_points[i]]

            graphed_patches = [
                create_local_graph(x[0], -1, True, self.train, adj_mat=x[1])
                for x in patches
            ]
            return patches, graphed_patches

        patches_a, graphed_patches_a = generate_patches(
            indexes_fps_a, dijkstra_a, pc_a, GEODESIC_CUT
        )
        patches_p, graphed_patches_p = generate_patches(
            indexes_fps_p, dijkstra_p, pc_p, GEODESIC_CUT
        )
        seeds_t_a = torch.from_numpy(seeds_a_np)
        seeds_t_p = torch.from_numpy(seeds_p_np)

        if self.train:

            def get_candidates(dijkstra, indexes_fps, distance):
                candidates = [
                    np.where(abs(dijkstra[i] - distance) < 4)[0] for i in indexes_fps
                ]
                candidates = [
                    [np.random.choice(len(dijkstra))]
                    if len(candidates) == 0
                    else candidates
                    for candidates in candidates
                ]
                return [np.random.choice(candidate) for candidate in candidates]

            candidates = get_candidates(dijkstra_p, indexes_fps_p, GEODESIC_CUT * 2)
            patches_n, graphed_patches_n = generate_patches(
                candidates, dijkstra_p, pc_p, GEODESIC_CUT
            )
            seeds_n_np = seeds_p_np
        else:
            graphed_patches_n = None
            patches_n = patches_p
            seeds_n_np = seeds_p_np

        seed_dist = dijkstra_a[indexes_fps_a, :][:, indexes_fps_p]
        dist_mat_t = torch.from_numpy(seed_dist).type(torch.DoubleTensor)

        geod_mat_a = compute_geod(mesh_a, indexes_fps_a)
        geod_mat_p = compute_geod(mesh_p, indexes_fps_p)
        shape_p = create_shape_graph(
            seeds_p_np, None, geod_mat_p[indexes_fps_p, :][:, indexes_fps_p]
        )
        shape_a = create_shape_graph(
            seeds_a_np, None, geod_mat_a[indexes_fps_a, :][:, indexes_fps_a]
        )

        # frame_id_a = '{}_{}'.format(seq_id_a,fid_a)
        # frame_id_p = '{}_{}'.format(seq_id_p,fid_p)

        result = {
            "a": graphed_patches_a,
            "p": graphed_patches_p,
            "n": graphed_patches_n,
            "scene_a": name_a,
            "scene_p": name_p,
            "seeds_a": seeds_t_a,
            "seeds_p": seeds_t_p,
            "dist_mat": dist_mat_t,
            "shape_a": shape_a,
            "shape_p": shape_p,
        }

        if not self.train:
            # path_canon_off_a = self.root+'training/registrations_off/'+frame_id_a
            # path_canon_off_p = self.root+'training/registrations_off/'+frame_id_p
            trimesh_a = o3d_to_trimesh(mesh_a)
            trimesh_p = o3d_to_trimesh(mesh_p)
            mesh_a_geod = None  # trimesh_a.get_geodesic(verbose=False)

            result.update(
                {
                    "seed_index_a": indexes_fps_a,
                    "seed_index_p": indexes_fps_p,
                    "full_points_a": pc_a_np,
                    "full_points_p": pc_p_np,
                    "mesh_a": trimesh_a,
                    "mesh_p": trimesh_p,
                    "mesh_a_geod": mesh_a_geod,
                    # 'dij_a':dijkstra_a,
                    # 'dij_p':dijkstra_p
                }
            )

            graphed_patches_n = graphed_patches_p

        if self.DEBUG_PLOT:
            vis_graph_with_same_idx([mesh_a, mesh_p])
            visualize_graph_shape(shape_p)
            visualize_graph_shape(shape_a)

            if self.train:
                for i in range(len(patches_a)):
                    patches_a[i][0].paint_uniform_color([0, 1, 0])
                    patches_n[i][0].paint_uniform_color([1, 0, 0])
                    patches_p[i][0].paint_uniform_color([0, 0, 1])
                for i in range(len(graphed_patches_a)):
                    visualize_torch_graphs_full(
                        graphed_patches_a[i],
                        graphed_patches_p[i],
                        graphed_patches_n[i],
                        seeds_a_np,
                        seeds_p_np,
                        seeds_n_np,
                    )
        else:
            pass
        return result
