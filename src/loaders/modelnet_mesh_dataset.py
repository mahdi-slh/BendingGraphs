import open3d as o3d
import numpy as np
import copy
import os
import glob
from sklearn import neighbors
from sympy import O
from tqdm import tqdm
from torch.utils.data import Dataset
from loaders.augmentations import random_perturb
from graph import *
from configs import *
import open3d as o3d

from tqdm.contrib.concurrent import process_map
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from utils.pointcloud_utils import fps_points_random, fps_points
from utils.visualize_graph import visualize_graph_pair, visualize_graph_shape
from utils.pyFM.mesh import TriMesh
from utils.utils import set_random_seed
from utils.mesh_utils import compute_geod, get_dijkstra, normalize_unit_cube, normalize_unit_sphere
import json
from scipy.spatial import distance_matrix

DEBUG_PLOT = True
DEBUG_PLOT = False
THREADS = 0
nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


def load_point_cloud_from_off(filename):

    try:
        mesh = o3d.io.read_triangle_mesh(filename)
        pc = mesh.sample_points_uniformly(16000)
    except:
        print(filename)
        lines = open(filename).readlines()
        badline_corrected = lines[0].split("OFF")
        goodlines = lines[1:]
        goodlines.insert(0, badline_corrected[1])
        goodlines.insert(0, "OFF\n")
        with open("temp.off", "w") as f:
            for l in goodlines:
                f.write(l)

        mesh = o3d.io.read_triangle_mesh("temp.off")
        pc = mesh.sample_points_uniformly(16000)

    pc.estimate_normals()
    pc.normalize_normals()
    # pc.colors = o3d.utility.Vector3dVector(np.zeros([len(pc.points), 3]))

    return pc


def get_random_transformation():

    # from dcp

    anglex = np.random.uniform() * np.pi / 4
    angley = np.random.uniform() * np.pi / 4
    anglez = np.random.uniform() * np.pi / 4
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)

    euler_ab = np.asarray([anglez, angley, anglex])

    translation_ab = np.array(
        [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
        ]
    )

    return translation_ab, R_ab, euler_ab


class ModelNetMeshDataset(Dataset):
    def __init__(self, root, train, mode, sample_per_frame=None):
        set_random_seed(2021)
        self.root = root
        # set test_folder to 'test' for training on un-annotated test folder
        self.test_train = "train/" if train else "test/"
        self.mode = mode
        self.train = train

        self.patches_per_shape = 100
        self.sample_per_frame = 1 if train else 1
        process_list = []
        self.same_indices=True
        self.partial=False
        random.seed(7)
        if train:
            print("generate random scan list")
            self.object_ids = [7]  # ,13,14,15,16,17,18,19,20]
        else:
            print("load test pairs from file")
            self.object_ids = [7]

        with open(root + "object_names.txt") as f:
            object_names = f.readlines()
            object_names = [x.replace("\n", "") for x in object_names if len(x) > 3]
        self.object_names = [
            object_names[idx]
            for idx in range(len(object_names))
            if idx in self.object_ids
        ]
        # self.random_pairs = False

        process_list, unprocessed_list = self.get_process_list()
        self.process_list = process_list
        self.unprocessed_list = unprocessed_list

        if sample_per_frame:
            self.sample_per_frame = sample_per_frame

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
        for (obj_name, f_s_id, f_e_id, s_id) in tqdm(self.process_list):
            filename = self.get_filename(obj_name, f_s_id, f_e_id, s_id)
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
                # [len_patches//random_proportion:]])
                frame_data["n"].extend([temp_data["n"][id] for id in random_perm])

            self.data.extend([frame_data])
            self.meta.append(
                {k: temp_data[k] for k in temp_data.keys() - ["a", "p", "n"]}
            )

    def __getitem__(self, idx):
        return self.data[idx], self.meta[idx]

    def __len__(self):
        return len(self.data)

    def processed_file_name(self):
        p_f = self.patches_per_shape
        return (
            self.root
            + "processed/"
            + self.test_train
            + self.mode + ('_partial/' if self.partial else '_full/')
            + "p{}_r{}_{}".format(
                p_f, str(GEODESIC_CUT), "trval" if self.train else "test"
            )
        )

    def get_process_list(self):
        is_train = self.train
        process_list = []
        unprocessed_list = []

        created_files = glob.glob(self.processed_file_name() + "/*")
        processed_keys = list()

        if not DEBUG_PLOT:
            for path in created_files:
                name = path.split("/")[-1]
                tmp = name.split(".")[0]
                obj_name = tmp.split("-")[0]
                frame_id_orig = int(tmp.split("-")[1])
                frame_id_targ = int(tmp.split("-")[2].split("_")[0])
                sid = int(tmp.split("-")[2].split("_")[1][1:])
                processed_keys.append((obj_name, frame_id_orig, frame_id_targ, sid))

        total_num = 0
        for obj_name in self.object_names:
            base_path = self.root + obj_name + "/" + self.test_train
            file_list = os.listdir(base_path)
            id_list = [int(f.split("_")[-1].split(".")[0]) for f in file_list]
            for _ in range(self.sample_per_frame):
                total_num += len(file_list)

        for k in processed_keys:
            if len(process_list) >= total_num:
                break
            if (
                k[0] in self.object_names
                and k[1] in id_list
                and k[3] < self.sample_per_frame
            ):
                process_list.append(k)
        random.seed(2022)
        for obj_name in self.object_names:
            base_path = self.root + obj_name + "/" + self.test_train
            file_list = os.listdir(base_path)
            for file_name in file_list:
                if file_name.split(".")[-1]=='off':
                    frame_id_orig = int(file_name.split("_")[-1].split(".")[0])
                    for i in range(self.sample_per_frame):
                        if len(process_list) + len(unprocessed_list) >= total_num:
                            break
                        if self.mode == 'category':

                            file_rand = random.choice(file_list)
                            frame_id_targ = int(file_rand.split("_")[-1].split(".")[0])
                        else:
                            frame_id_targ = frame_id_orig

                        key = (obj_name, frame_id_orig, frame_id_targ, i)
                        if key not in process_list:
                            unprocessed_list.append(key)
                        # else:
                        #     process_list.append(key)
        # else:
        #     pairs_json_file = self.root+'/training/pairs.json'
        #     self.sample_per_frame = 1
        #     with open(pairs_json_file) as f:
        #         data_json = json.load(f)
        #     for pair in data_json:
        #         frame_id_orig = int(pair['B'][7:10])
        #         frame_id_targ = int(pair['A'][7:10])

        #         key = (frame_id_orig, frame_id_targ, 0)

        #         if key in processed_keys:
        #             process_list.append(key)
        #         else:
        #             unprocessed_list.append(key)

        return process_list, unprocessed_list

    def download(self):
        pass

    @property
    def folder_mesh(self):
        return self.root

    def process(self):
        os.makedirs(self.processed_file_name(), exist_ok=True)
        if THREADS == 0 or DEBUG_PLOT:
            for d in tqdm(self.unprocessed_list):
                self.process_impl(d)
        else:
            print(f"process data on {THREADS} threads")
            process_map(self.process_impl, self.unprocessed_list, max_workers=THREADS)

    def get_filename(self, obj_name, fids, fidt, sid):
        return f"{obj_name}-{fids:04d}-{fidt:04d}_s{sid:02d}.pt"

    def process_impl(self, data):
        obj_name, frame_id_orig, frame_id_targ, sample_idx = data

        filename = self.get_filename(obj_name, frame_id_orig, frame_id_targ, sample_idx)
        outputpath = os.path.join(self.processed_file_name(), filename)
        if (os.path.exists(outputpath)) and not DEBUG_PLOT:
            # print(f"data {filename} already exists. skip")
            return

        # file_mesh_a = f'{obj_name}_{frame_id_orig:03d}.ply'
        file_mesh_a_off = f"{obj_name}_{frame_id_orig:04d}.off"

        # file_mesh_p = f'{obj_name}_{frame_id_targ:03d}.ply'
        file_mesh_p_off = f"{obj_name}_{frame_id_targ:04d}.off"

        path_mesh_a = os.path.join(
            self.folder_mesh, obj_name, self.test_train, file_mesh_a_off
        )
        path_mesh_p = os.path.join(
            self.folder_mesh, obj_name, self.test_train, file_mesh_p_off
        )

        mesh_mesh_a = o3d.io.read_triangle_mesh(path_mesh_a)
        
        # 
        mesh_mesh_a.remove_unreferenced_vertices()
        mesh_mesh_a = normalize_unit_cube(
            mesh_mesh_a
        )#, 

        pc_o3d_a = mesh_mesh_a.sample_points_uniformly(5000)
        pc_np_a = fps_points(np.asarray(pc_o3d_a.points),1024)
        pc_o3d_a.points= o3d.utility.Vector3dVector(pc_np_a)
        # pc_np_a = np.asarray(pc_o3d_a.points)
        if self.partial:
            random_view = np.random.random(3)*4-2

        pc_o3d_p = o3d.geometry.PointCloud()

        if (not self.same_indices) or self.mode == 'category':
            mesh_mesh_p = o3d.io.read_triangle_mesh(path_mesh_p)
            mesh_mesh_p.remove_unreferenced_vertices()
            mesh_mesh_p = normalize_unit_cube(mesh_mesh_p)
            pc_np_p = np.asarray(mesh_mesh_p.sample_points_uniformly(5000).points)
            pc_np_p =fps_points(pc_np_p,1024)
            # if self.partial:
                
            #     nbrs_knn = NearestNeighbors(n_neighbors=512, algorithm="ball_tree")
            #     nbrs_knn.fit(pc_np_p)
            #     closest_to_view = nbrs_knn.kneighbors(random_view.reshape([1,3]), return_distance=False)
            #     pc_np_p = pc_np_p[closest_to_view,:]

            pc_o3d_p.points= o3d.utility.Vector3dVector(pc_np_p)

        else:
            pc_o3d_p.points = o3d.utility.Vector3dVector(
                np.copy(np.asarray(pc_o3d_a.points))
            )

        random_t, random_r, random_euler = get_random_transformation()
        pc_np_p_zero_pose = np.asarray(pc_o3d_p.points)
        pc_o3d_p.rotate(random_r)
        pc_o3d_p.translate(random_t, True)
        pc_np_p = np.asarray(pc_o3d_p.points)

        dist_mat_a = distance_matrix(pc_np_a, pc_np_a)
        dist_mat_p = distance_matrix(pc_np_p, pc_np_p)

        if self.mode == "noise":
            pc_o3d_a = random_perturb(pc_o3d_a, False, sigma=0.01, clip=0.05)
            pc_o3d_p = random_perturb(pc_o3d_p, False, sigma=0.01, clip=0.05)

        seeds_a_np, indices_fps_a = fps_points(
            pc_np_a, self.patches_per_shape, return_indices=True
        )
        if not self.train:
            seeds_p_np, indices_fps_p = fps_points(
                pc_np_p, self.patches_per_shape, return_indices=True
            )
        else:
            indices_fps_p = indices_fps_a

        partial_indices_t = torch.from_numpy(np.arange(len(indices_fps_p))) 
        if self.partial:

            nbrs_knn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
            nbrs_knn.fit(pc_np_p[indices_fps_p])
            closest_to_view = nbrs_knn.kneighbors(random_view.reshape([1,3]), return_distance=False)
            closest_to_view = sorted(closest_to_view.squeeze())

            indices_fps_p = [indices_fps_p[i] for i in range(len(indices_fps_p)) if i in closest_to_view]

            # plane_side = (pc_np_p_zero_pose[indices_fps_p,:]*random_plane).sum(1)
            # indices_fps_p = [id for id in indices_fps_p if plane_side[id]>0]
            partial_indices_t = torch.from_numpy(np.asarray(closest_to_view)) 
        
        seeds_p_np = pc_np_p[indices_fps_p, :]

        def generate_patches(index, dist_mat, pc, distance: float):
            patch_indices = [np.where(dist_mat[i] < distance)[0] for i in index]
            patches = [pc.select_by_index(idxs) for idxs in patch_indices]
            dij_local = [dist_mat[idxs, :][:, idxs] for idxs in patch_indices]
            patches = list(zip(patches, dij_local))
            graphed_patches = [
                create_local_graph(
                    x[0], -1, True, self.train, adj_mat=x[1], mat_type="euc"
                )
                for x in patches
            ]
            return patches, graphed_patches

        patches_a, graphed_patches_a = generate_patches(
            indices_fps_a, dist_mat_a, pc_o3d_a, EUCLIDEAN_CUT
        )
        patches_p, graphed_patches_p = generate_patches(
            indices_fps_p, dist_mat_p, pc_o3d_p, EUCLIDEAN_CUT
        )

        seeds_t_a = torch.from_numpy(seeds_a_np)
        seeds_t_p = torch.from_numpy(seeds_p_np)

        if self.train:

            def get_candidates(dijkstra, indices_fps, distance):
                candidates = [
                    np.where(abs(dijkstra[i] - distance) < 0.1)[0] for i in indices_fps
                ]
                candidates = [
                    [np.random.choice(len(dijkstra))]
                    if len(candidates) == 0
                    else candidates
                    for candidates in candidates
                ]
                return [np.random.choice(candidate) for candidate in candidates]

            candidates = get_candidates(dist_mat_a, indices_fps_a, EUCLIDEAN_CUT * 2)
            patches_n, graphed_patches_n = generate_patches(
                candidates, dist_mat_a, pc_o3d_a, EUCLIDEAN_CUT
            )
            seeds_n_np = seeds_a_np

        else:
            patches_n = None
            graphed_patches_n = None
        

        seed_dist = dist_mat_a[indices_fps_a, :][:, indices_fps_p]
        dist_mat_t = torch.from_numpy(seed_dist).type(torch.DoubleTensor)

        shape_p = create_shape_graph(
            seeds_p_np, None, dist_mat_p[indices_fps_p, :][:, indices_fps_p]
        )
        shape_a = create_shape_graph(
            seeds_a_np, None, dist_mat_a[indices_fps_a, :][:, indices_fps_a]
        )
        
        
        dict_sample = {
            "a": graphed_patches_a,
            "p": graphed_patches_p,
            "n": graphed_patches_n,
            "scene_a": frame_id_orig,
            "scene_p": frame_id_targ,
            "seeds_a": seeds_t_a,
            "seeds_p": seeds_t_p,
            "dist_mat": dist_mat_t,
            "t_p": torch.from_numpy(random_t),
            "r_p": torch.from_numpy(random_r),
            "shape_a": shape_a,
            "shape_p": shape_p,
        }
        if self.partial: dict_sample.update({'partial_matches': partial_indices_t})
        if not self.train:
            # path_canon_off_a = self.root+'training/registrations_off/'+file_mesh_a_off
            # path_canon_off_p = self.root+'training/registrations_off/'+file_mesh_p_off
            # trimesh_a = TriMesh(path_canon_off_a)
            # trimesh_p = TriMesh(path_canon_off_p)
            # mesh_a_geod = None  # trimesh_a.get_geodesic(verbose=False)

            dict_sample.update(
                {
                    "seed_index_a": indices_fps_a,
                    "seed_index_p": indices_fps_p,
                    "full_points_a": pc_np_a,
                    "full_points_p": pc_np_p,
                    # 'mesh_a': trimesh_a,
                    # 'mesh_p': trimesh_p,
                    # 'mesh_a_geod': mesh_a_geod,
                    # 'dij_a':dijkstra_a,
                    # 'dij_p':dijkstra_p
                }
            )
        # DEBUG_PLOT=True
        if DEBUG_PLOT:
            if self.mode=='category':
                visualize_graph_shape(shape_p)
                visualize_graph_shape(shape_a)
            else:
                if self.partial:
                    visualize_graph_pair([shape_a,shape_p],partial_indices_t)
                else:
                    visualize_graph_pair([shape_a,shape_p])
            for i in range(len(patches_p)):
                patches_a[i][0].paint_uniform_color([0, 1, 0])
                patches_n[i][0].paint_uniform_color([1, 0, 0])
                patches_p[i][0].paint_uniform_color([0, 0, 1])
            for i in range(len(graphed_patches_a)):
                visualize_torch_graphs_full(
                    graphed_patches_a[partial_indices_t[i]],
                    graphed_patches_p[i],
                    graphed_patches_n[partial_indices_t[i]],
                    seeds_a_np,
                    seeds_p_np,
                    seeds_n_np,
                )

        torch.save(dict_sample, outputpath)


if __name__ == "__main__":
    from configs import training_configs

    settings = training_configs()
    FaustSynDataset(
        root=MPIFAUST_DIR, train=False, sample_per_frame=0, mode=settings.data_mode
    )
