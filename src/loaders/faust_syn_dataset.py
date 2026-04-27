import open3d as o3d
import numpy as np
import copy
import os, glob
from tqdm import tqdm
from torch.utils.data import Dataset
from loaders.augmentations import random_perturb
from graph import *
from configs import *
import open3d as o3d
import torch_geometric
from loaders.matching_dataset_base import Matching_Dataset
from tqdm.contrib.concurrent import process_map
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from utils.mesh_utils import compute_geod, get_dijkstra, normalize_unit_sphere
from utils.pointcloud_utils import fps_points_random, fps_points
from utils.visualize_graph import visualize_graph_shape
from utils.pyFM.mesh import TriMesh
from utils.utils import set_random_seed
import itertools
import json

# DEBUG_PLOT=True
# DEBUG_PLOT=False
# THREADS=4
nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


class FaustSynDataset(Matching_Dataset):
    def __init__(
        self,
        root,
        train,
        mode,
        frame_range=None,
        sample_per_frame=None,
        test_folder="training/",
    ):
        super().__init__(train, mode)
        set_random_seed(2021)
        self.DEBUG_PLOT = True
        self.DEBUG_PLOT = False
        self.root = root
        self.THREADS = 4

        self.sample_per_frame = 1 if train else 1
        process_list = []
        random_pairs = True
        random.seed(7)
        if train:
            print("generate random scan list")
            self.frame_ids_list = range(99)
        else:
            print("load test pairs from file")

        process_list, unprocessed_list = self.get_process_scan_list(random_pairs)
        self.process_list = process_list
        self.unprocessed_list = unprocessed_list

        if sample_per_frame:
            self.sample_per_frame = sample_per_frame
        if frame_range:
            self.frame_ids_list = frame_range

        # generate process list and process

        self.process()
        self.process_list = self.process_list + self.unprocessed_list
        self.data = []
        self.meta = []
        if not train:
            self.process_list = self.process_list[:20]
        # self.raw = {}

        # path = self.processed_paths
        # temp_data = torch.load(path)
        print("loading data...")
        for (f_s_id, f_e_id, s_id) in tqdm(self.process_list):
            filename = self.get_filename(f_s_id, f_e_id, s_id)
            outputpath = os.path.join(self.processed_file_name(), filename)
            if not (os.path.exists(outputpath)):
                raise RuntimeError()
            temp_data = torch.load(outputpath)
            frame_data = {}
            frame_data["a"] = temp_data["a"]
            frame_data["p"] = temp_data["p"]

            len_patches = len(temp_data["a"])
            random_proportion = 1
            np.random.seed(4)
            random_perm = np.random.permutation(len_patches // random_proportion)
            # nordered_perm = np.arange(len_patches)
            if temp_data["n"]:
                frame_data["n"] = []
                # frame_data['n'].extend([temp_data[j]['n'][id] for id in random_perm[:len_patches//random_proportion]])
                frame_data["n"].extend(
                    [temp_data["n"][id] for id in random_perm]
                )  # [len_patches//random_proportion:]])

            self.data.extend([frame_data])
            self.meta.append(
                {k: temp_data[k] for k in temp_data.keys() - ["a", "p", "n"]}
            )

        # for j in range(len(temp_data)):
        #     frame_data = {}

        #     frame_data['a'] = temp_data[j]['a']
        #     frame_data['p'] = temp_data[j]['p']

        #     len_patches = len(temp_data[j]['a'])
        #     random_proportion = 1
        #     np.random.seed(4)
        #     random_perm = np.random.permutation(len_patches//random_proportion)
        #     nordered_perm = np.arange(len_patches)
        #     if temp_data[j]['n']:
        #         frame_data['n']=[]
        #         # frame_data['n'].extend([temp_data[j]['n'][id] for id in random_perm[:len_patches//random_proportion]])
        #         frame_data['n'].extend([temp_data[j]['n'][id] for id in random_perm])#[len_patches//random_proportion:]])

        #     self.data.extend([frame_data])
        #     self.meta.append(
        #         {k: temp_data[j][k] for k in temp_data[j].keys()-['a','p','n']})

    def __getitem__(self, idx):
        return self.data[idx], self.meta[idx]

    def __len__(self):
            return len(self.data)

    def processed_file_name(self):
        p_f = self.points_per_shape
        return (
            self.root
            + "processed/"
            + self.mode
            + self.test_train
            + "FAUSTSyn_p{}_r{}_{}".format(
                p_f, str(GEODESIC_CUT), "trval" if self.train else "test"
            )
        )

    def get_process_scan_list(self, random_pairs: bool):
        is_train = self.train
        process_list = []
        unprocessed_list = []

        created_files = glob.glob(self.processed_file_name() + "/*")
        processed_keys = list()

        if not self.DEBUG_PLOT:
            for path in created_files:
                name = path.split("/")[-1]
                tmp = name.split("(")[1].split(")")[0]
                frame_id_orig = int(tmp.split("-")[0])
                frame_id_targ = int(tmp.split("-")[1])
                sid = int(name.split("_")[1].split(".")[0][1:])
                processed_keys.append((frame_id_orig, frame_id_targ, sid))

        if is_train:
            total_num = 0
            for _ in self.frame_ids_list:
                for _ in range(self.sample_per_frame):
                    total_num += 1

            for k in processed_keys:
                if len(process_list) >= total_num:
                    break
                if (
                    k[0] in self.frame_ids_list
                    and k[1] in self.frame_ids_list
                    and k[2] < self.sample_per_frame
                ):
                    process_list.append(k)

            for frame_id_orig in self.frame_ids_list:
                for i in range(self.sample_per_frame):
                    if len(process_list) + len(unprocessed_list) >= total_num:
                        break
                    if random_pairs:
                        frame_id_targ = random.randint(0, len(self.frame_ids_list))
                    else:
                        frame_id_targ = frame_id_orig + 1

                    key = (frame_id_orig, frame_id_targ, i)
                    if key not in process_list:
                        unprocessed_list.append(key)
                    # else:
                    #     process_list.append(key)
        else:
            pairs_json_file = self.root + "/training/pairs.json"
            self.sample_per_frame = 1
            with open(pairs_json_file) as f:
                data_json = json.load(f)
            for pair in data_json:
                frame_id_orig = int(pair["B"][7:10])
                frame_id_targ = int(pair["A"][7:10])

                key = (frame_id_orig, frame_id_targ, 0)

                if key in processed_keys:
                    process_list.append(key)
                else:
                    unprocessed_list.append(key)

        return process_list, unprocessed_list

    def download(self):
        pass

    @property
    def folder_scan(self):
        return self.root + self.test_train + "scans_off/"

    @property
    def folder_canon(self):
        return (
            self.root
            + self.test_train
            + ("registrations/" if self.test_folder == "training/" else "scans_off/")
        )

    def process(self):
        os.makedirs(self.processed_file_name(), exist_ok=True)
        if self.THREADS == 0 or self.DEBUG_PLOT:
            for d in tqdm(self.unprocessed_list):
                self.process_impl(d)
        else:
            print(f"process data on {self.THREADS} threads")
            process_map(
                self.process_impl,
                self.unprocessed_list,
                max_workers=self.THREADS,
                chunksize=self.THREADS,
            )

    def get_filename(self, fids, fidt, sid):
        return f"fid({fids:03d}-{fidt:03d})_s{sid:02d}.pt"

    def process_impl(self, data):
        frame_id_orig, frame_id_targ, sample_idx = data

        filename = self.get_filename(frame_id_orig, frame_id_targ, sample_idx)
        outputpath = os.path.join(self.processed_file_name(), filename)
        if (os.path.exists(outputpath)) and not self.DEBUG_PLOT:
            # print(f"data {filename} already exists. skip")
            return

        file_canon_a = f"tr_reg_{frame_id_orig:03d}.ply"
        file_canon_a_off = f"tr_reg_{frame_id_orig:03d}.off"

        file_canon_p = f"tr_reg_{frame_id_targ:03d}.ply"
        file_canon_p_off = f"tr_reg_{frame_id_targ:03d}.off"

        path_canon_a = self.folder_canon + file_canon_a
        path_canon_p = self.folder_canon + file_canon_p

        mesh_canon_a = o3d.io.read_triangle_mesh(path_canon_a)
        mesh_canon_p = o3d.io.read_triangle_mesh(path_canon_p)
        # mesh_canon_a, mesh_canon_p = normalize_unit_sphere([mesh_canon_a, mesh_canon_p])

        try:
            result = self.generate(
                mesh_canon_a,
                mesh_canon_p,
                name_a=str(frame_id_orig),
                name_p=str(frame_id_targ),
            )
            torch.save(result, outputpath)
        except:
            pass
        return

        # dijkstra_a = get_dijkstra(mesh_canon_a)
        # dijkstra_p = get_dijkstra(mesh_canon_p)

        # pc_canon_o3d_a = o3d.geometry.PointCloud()
        # pc_canon_o3d_p = o3d.geometry.PointCloud()

        # pc_canon_o3d_a.points = mesh_canon_a.vertices
        # pc_canon_np_a = np.asarray(pc_canon_o3d_a.points)

        # pc_canon_o3d_p.points = mesh_canon_p.vertices
        # pc_canon_np_p = np.asarray(pc_canon_o3d_p.points)

        # if self.mode == 'noise/' and self.train:
        #     pc_canon_o3d_a = random_perturb(
        #         pc_canon_o3d_a, False, sigma=0.004, clip=0.004)
        #     pc_canon_o3d_p = random_perturb(
        #         pc_canon_o3d_p, False, sigma=0.004, clip=0.004)

        # seeds_a_np, indexes_fps_a = fps_points_random(pc_canon_np_a, self.points_per_shape, return_indexes=True)
        # if not self.train:
        #     seeds_p_np, indexes_fps_p = fps_points_random(pc_canon_np_p, self.points_per_shape, return_indexes=True)
        # else:
        #     indexes_fps_p =  indexes_fps_a
        #     seeds_p_np = pc_canon_np_p[indexes_fps_a,:]

        # def generate_patches(index, dijkstra, pc, distance:float):
        #     patch_indexes = [np.where(dijkstra[i] < distance)[0] for i in index]
        #     patches = [pc.select_by_index(idxs) for idxs in patch_indexes]
        #     dij_local = [dijkstra[idxs, :][:, idxs] for idxs in patch_indexes]
        #     patches= list(zip(patches, dij_local))
        #     graphed_patches= [create_local_graph(x[0], -1, True, self.train,adj_mat = x[1]) for x in patches]
        #     return patches, graphed_patches
        # patches_a, graphed_patches_a = generate_patches(indexes_fps_a,dijkstra_a,pc_canon_o3d_a,GEODESIC_CUT)
        # patches_p, graphed_patches_p = generate_patches(indexes_fps_p,dijkstra_p,pc_canon_o3d_p,GEODESIC_CUT)

        # # patch_indexes_a = [np.where(dijkstra_a[i] < GEODESIC_CUT)[0] for i in indexes_fps_a]
        # # patches_a = [pc_canon_o3d_a.select_by_index(idxs) for idxs in patch_indexes_a]
        # # dij_local_a=[dijkstra_a[idxs,:][:,idxs] for idxs in patch_indexes_a]
        # # patches_a = list(zip(patches_a,dij_local_a))

        # # patch_indexes_p = [np.where(dijkstra_p[i] < GEODESIC_CUT)[0] for i in indexes_fps_p]
        # # patches_p = [pc_canon_o3d_p.select_by_index(idxs) for idxs in patch_indexes_p]
        # # dij_local_p=[dijkstra_p[idxs,:][:,idxs] for idxs in patch_indexes_p]
        # # patches_p = list(zip(patches_p,dij_local_p))

        # # graphed_patches_a = [create_local_graph(
        # #     x[0], -1, True, self.train,adj_mat = x[1]) for x in patches_a]

        # # graphed_patches_p = [create_local_graph(
        # #     x[0], -1, True, self.train,adj_mat = x[1]) for x in patches_p]

        # seeds_t_a = torch.from_numpy(seeds_a_np)
        # seeds_t_p = torch.from_numpy(seeds_p_np)

        # if self.train:
        #     def get_candidates(dijkstra,indexes_fps,distance):
        #         candidates = [np.where(abs(dijkstra[i]-distance) < 4)[0] for i in indexes_fps]
        #         candidates = [[np.random.choice(len(dijkstra))] if len(candidates) == 0 else candidates for candidates in candidates]
        #         return  [np.random.choice(candidate) for candidate in candidates]
        #     candidates = get_candidates(dijkstra_p, indexes_fps_p,GEODESIC_CUT*2)
        #     patches_n, graphed_patches_n  = generate_patches(candidates, dijkstra_p, pc_canon_o3d_p, GEODESIC_CUT)
        #     seeds_n_np = seeds_p_np
        #     # neg_candidates = [np.where(
        #     #         abs(dijkstra_a[i]-GEODESIC_CUT*2) < 4)[0] for i in indexes_fps_a]

        #     # neg_candidates = [[np.random.choice(len(dijkstra_a))] if len(candidates)==0 else candidates for candidates in neg_candidates  ]
        #     # index_neg = [np.random.choice(candidates) for candidates in neg_candidates]

        #     # patch_indexes_n = [np.where(dijkstra_a[i] < GEODESIC_CUT)[0] for i in index_neg]
        #     # patches_n = [pc_canon_o3d_a.select_by_index(idxs) for idxs in patch_indexes_n]

        #     # dij_local_n=[dijkstra_a[idxs,:][:,idxs] for idxs in patch_indexes_n]
        #     # patches_n = list(zip(patches_n,dij_local_n))

        #     # graphed_patches_n = [create_local_graph(
        #     # x[0], -1, True, self.train,adj_mat = x[1]) for x in patches_n]
        # else:
        #     patches_n=None
        #     graphed_patches_n = None

        # seed_dist = dijkstra_a[indexes_fps_a,:][:,indexes_fps_p]
        # dist_mat_t= torch.from_numpy(seed_dist).type(torch.DoubleTensor)

        # geod_mat_a= compute_geod(mesh_canon_a,indexes_fps_a)
        # geod_mat_p= compute_geod(mesh_canon_p,indexes_fps_p)
        # shape_p = create_shape_graph(seeds_p_np,None,geod_mat_p[indexes_fps_p,:][:,indexes_fps_p])
        # shape_a = create_shape_graph(seeds_a_np,None,geod_mat_a[indexes_fps_a,:][:,indexes_fps_a])

        # result = {'a': graphed_patches_a,
        #                     'p': graphed_patches_p,
        #                     'n': graphed_patches_n,
        #                     'scene_a': frame_id_orig,
        #                     'scene_p': frame_id_targ,
        #                     'seeds_a': seeds_t_a,
        #                     'seeds_p': seeds_t_p,
        #                     'dist_mat': dist_mat_t,

        #                     'shape_a': shape_a,
        #                     'shape_p': shape_p

        #                     }
        # if not self.train:
        #     path_canon_off_a = self.root+'training/registrations_off/'+file_canon_a_off
        #     path_canon_off_p = self.root+'training/registrations_off/'+file_canon_p_off
        #     trimesh_a = TriMesh(path_canon_off_a)
        #     trimesh_p = TriMesh(path_canon_off_p)
        #     mesh_a_geod =None# trimesh_a.get_geodesic(verbose=False)

        #     result.update({
        #                 'seed_index_a':indexes_fps_a,
        #                 'seed_index_p':indexes_fps_p,
        #                 'full_points_a':pc_canon_np_a,
        #                 'full_points_p':pc_canon_np_p,
        #                 'mesh_a':trimesh_a,
        #                 'mesh_p':trimesh_p,
        #                 'mesh_a_geod':mesh_a_geod,
        #                 # 'dij_a':dijkstra_a,
        #                 # 'dij_p':dijkstra_p
        #                 })
        # if DEBUG_PLOT:
        #     visualize_graph_shape(shape_p)
        #     visualize_graph_shape(shape_a)
        #     for i in range(len(patches_a)):
        #         patches_a[i][0].paint_uniform_color([0,1,0])
        #         patches_n[i][0].paint_uniform_color([1,0,0])
        #         patches_p[i][0].paint_uniform_color([0,0,1])
        #     # for i in range(len(graphed_patches_a)):
        #     #     visualize_torch_graphs_full(graphed_patches_a[i],graphed_patches_p[i],graphed_patches_n[i],seeds_a_np,seeds_p_np,seeds_n_np)

        # torch.save(result, outputpath)


if __name__ == "__main__":
    from configs import training_configs

    settings = training_configs()
    FaustSynDataset(
        root=MPIFAUST_DIR, train=True, sample_per_frame=1, mode=settings.data_mode
    )
