"""
register here
https://www.di.ens.fr/willow/research/surreal/data/

git clone git@github.com:gulvarol/surreal.git
cd surreal
#run download script:
python download/download_smpl_data.sh {out_dir} {usrname} {password}

download SMPL data
https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip

SMPL explain:
    https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    

paper:
    qualitative:
        3D-RCNN: Instance-level 3D Object Reconstruction via Render-and-Compare
            https://openaccess.thecvf.com/content_cvpr_2018/papers/Kundu_3D-RCNN_Instance-Level_3D_CVPR_2018_paper.pdf
            
    for training
        Correspondence Learning via Linearly-invariant Embedding
        they train over 10K shapes, reampled each shape down to 1K vertices
        https://arxiv.org/pdf/2010.13136.pdf
        train on 
        https://arxiv.org/pdf/2106.13679.pdf

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
from utils.mesh_utils import o3d_to_trimesh
from utils.pointcloud_utils import fps_points_random, fps_points
from configs import GRAPH_SIZE, GEODESIC_CUT
import open3d as o3d
import random
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from utils.mesh_utils import (
    compute_geod,
    get_dijkstra,
    get_mesh_graph,
    normalize_unit_sphere,
)
from utils.visualize_graph import visualize_graph_shape
from utils.utils import vis_graph_with_same_idx
from utils.pyFM.mesh import TriMesh
from smpl_webuser.serialization import load_model
import pickle as pkl
from tqdm.contrib.concurrent import process_map
from utils.utils import set_random_seed, rotation_matrix_from_vectors

# import numpy as np
# from pathlib import Path
#
# DEBUG_PLOT=True
# DEBUG_PLOT=False
"""V1.1.0"""
# base_model_names = ['male','female','nautral']
# base_models = {
#     'male': 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
#     'female': 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
#     'neutral': 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'
#     }
"""V1.0.0"""
base_model_names = ["male", "female"]
base_models = {
    "male": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
    "female": "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
}

THREADS = 8
NUM_SEQUENCES = 2667  # the total number of sequences.

nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


def model_to_mesh(
    pose, tran, betas, model, betas_var: float = 0.01, pose_var: float = 0.1
):
    tmp = np.random.normal(0.0, betas_var, size=model.betas.shape)
    model.betas[:] = (
        betas + tmp
    )  # s a vector of the coefficients of the learned PCA shape space
    model.pose[
        :
    ] = pose  # s the relative rotation of the N = 33 joints in the kinematic tree
    model.trans[:] = tran  # s the global translation applied to the root joint
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.copy(model.r))
    mesh.triangles = o3d.utility.Vector3iVector(np.copy(model.f))
    return mesh


def load_body_data(smpl_data, indices=0):
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith("pose_"):
            cmu_keys.append(seq.replace("pose_", ""))
    cmu_keys = sorted(cmu_keys)
    names = [cmu_keys[idx] for idx in indices]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq.startswith("pose_"):
            tmp = seq.replace("pose_", "")
            if tmp in names:
                # if seq == ('pose_' + name):
                cmu_parms[tmp] = {
                    "poses": smpl_data[seq],
                    "trans": smpl_data[seq.replace("pose_", "trans_")],
                }
    return (cmu_parms, names)


class SURREALDataset(Dataset):
    def __init__(self, path_data, path_model, train, mode, test_folder="training/"):
        set_random_seed(2020)
        self.DEBUG_PLOT = False
        # self.DEBUG_PLOT = True
        self.path_data = path_data
        self.path_model = path_model
        # set test_folder to 'test' for training on un-annotated test folder
        self.test_train = "training/" if train else test_folder
        self.test_folder = test_folder

        """generate data"""
        if train:
            self.num_samples = 100
            self.base_model_name = "random"
            self.points_per_shape = 200
        else:
            self.num_samples = 10
            self.base_model_name = "random"
            self.points_per_shape = 200  # -1

        assert self.base_model_name in base_model_names + ["random"]

        self.mode = mode + "/"
        self.train = train

        """process data"""
        self.sequence_ids_list = np.random.choice(NUM_SEQUENCES - 1, self.num_samples)
        self.process_list, self.unprocessed_list = self.get_process_scan_list()
        self.process()
        self.process_list += self.unprocessed_list

        self.data = []
        self.meta = []
        # self.raw = {}

        # print('loading data...')
        # for (seq_id_a,fid_a,gender_a,seq_id_p,fid_p,gender_p) in tqdm(self.process_list):
        #     filename = self.get_filename(seq_id_a,fid_a,gender_a,seq_id_p,fid_p,gender_p)
        #     outputpath = os.path.join(self.processed_file_name(), filename)
        #     if not (os.path.exists(outputpath)):
        #         raise RuntimeError()
        #     temp_data = torch.load(outputpath)
        #     frame_data = {}
        #     frame_data['a'] = temp_data['a']
        #     frame_data['p'] = temp_data['p']

        #     len_patches = len(temp_data['a'])
        #     random_proportion = 1
        #     np.random.seed(4)
        #     random_perm = np.random.permutation(len_patches//random_proportion)
        #     # nordered_perm = np.arange(len_patches)
        #     if temp_data['n']:
        #         frame_data['n']=[]
        #         # frame_data['n'].extend([temp_data[j]['n'][id] for id in random_perm[:len_patches//random_proportion]])
        #         frame_data['n'].extend([temp_data['n'][id] for id in random_perm])#[len_patches//random_proportion:]])

        #     self.data.extend([frame_data])
        #     self.meta.append(
        #         {k: temp_data[k] for k in temp_data.keys()-['a','p','n','scene_a','scene_p']})

    def __getitem__(self, idx):
        (seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p) = self.process_list[idx]
        filename = self.get_filename(
            seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p
        )
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

        meta = {
            k: temp_data[k]
            for k in temp_data.keys() - ["a", "p", "n", "scene_a", "scene_p"]
        }
        return frame_data, meta
        # return self.data[idx], self.meta[idx]

    def __len__(self):
        return len(self.process_list)


    def processed_folder_name(self):
        return (
            os.path.join(self.path_data, "processed", self.mode, self.test_train) + "/"
        )

    def processed_file_name(self):
        p_f = self.points_per_shape
        return self.processed_folder_name() + "SURREAL_p{}_r{}_{}".format(
            p_f, str(GEODESIC_CUT), "trval" if self.train else "test"
        )

    def download(self):
        pass

    def init_frame(self):
        frame = {}
        frame["pc"] = o3d.geometry.PointCloud()
        return frame

    def get_process_scan_list(self):
        """prepare model and data"""
        smpl_data = np.load(os.path.join(self.path_data, "smpl_data.npz"))
        (cmu_params, names) = load_body_data(smpl_data, indices=self.sequence_ids_list)
        # interval=self.interval # frame interval100 # frame interval
        process_list = []

        """load existing files"""
        if not self.DEBUG_PLOT:
            created_files = glob.glob(self.processed_file_name() + "/*")
            for path in created_files:
                if len(process_list) == self.num_samples:
                    break
                name = path.split("/")[-1]
                tokens = name.split("-")
                seq_id_a = tokens[0][3:]
                fid_a = int(tokens[1][3:])
                gender_a = tokens[2].split("_")[0]

                seq_id_p = tokens[2][len(gender_a) + 1 :][3:]
                fid_p = int(tokens[3][3:])
                gender_p = tokens[4]

                process_list.append(
                    (seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p)
                )

        unprocessed_list = []
        for _ in range(self.num_samples):
            if len(process_list) + len(unprocessed_list) == self.num_samples:
                break
            seq_id_a = random.choice(list(cmu_params.keys()))
            n_frames_a = cmu_params[seq_id_a]["poses"].shape[0]
            fid_a = random.choice(range(n_frames_a))

            seq_id_p = random.choice(list(cmu_params.keys()))
            n_frames_p = cmu_params[seq_id_p]["poses"].shape[0]
            fid_p = random.choice(range(n_frames_p))

            gender_a = (
                random.choice(base_model_names)
                if self.base_model_name == "random"
                else self.base_model_name
            )
            gender_p = (
                random.choice(base_model_names)
                if self.base_model_name == "random"
                else self.base_model_name
            )

            unprocessed_list.append(
                (seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p)
            )

            # for fid in range(interval,n_frames,interval):
            #     if self.base_model_name == 'random':
            #         gender = random.choice(base_model_names)
            #     else:
            #         gender = self.base_model_name
            #     pairs.append((seq_id, gender, fid-interval,fid))
            # # random select K pairs
            # if len(pairs) > self.sample_per_sequence:
            #     selections = np.random.choice(len(pairs), self.sample_per_sequence,replace=False)
            #     pairs = [pairs[idx] for idx in selections]
            # process_list += pairs

        # for seq_id in cmu_params:
        #     poses = cmu_params[seq_id]['poses']
        #     n_frames = poses.shape[0]
        #     pairs = list()

        #     for fid in range(interval,n_frames,interval):
        #         if self.base_model_name == 'random':
        #             gender = random.choice(base_model_names)
        #         else:
        #             gender = self.base_model_name
        #         pairs.append((seq_id, gender, fid-interval,fid))
        #     # random select K pairs
        #     if len(pairs) > self.sample_per_sequence:
        #         selections = np.random.choice(len(pairs), self.sample_per_sequence,replace=False)
        #         pairs = [pairs[idx] for idx in selections]
        #     process_list += pairs
        return process_list, unprocessed_list

    def process(self, betas_var: float = 0.01, pose_var: float = 0.1):
        # load model

        # self.base_model = load_model(path_model)
        # load cmu_params
        smpl_data = np.load(os.path.join(self.path_data, "smpl_data.npz"))
        femaleshapes = smpl_data["femaleshapes"]
        maleshapes = smpl_data["maleshapes"]
        (self.cmu_params, _) = load_body_data(smpl_data, indices=self.sequence_ids_list)
        # (self.cmu_params, _) = load_body_data(smpl_data,indices=self.sequence_ids_list)
        # load betas
        # male_beta = np.load(os.path.join(self.path_data,'male_beta_stds.npy'))
        # female_beta = np.load(os.path.join(self.path_data,'female_beta_stds.npy'))
        # base_betas = np.copy(self.base_model.betas)
        self.var_betas = {"male": maleshapes, "female": femaleshapes}

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(o3d.geometry.TriangleMesh().create_coordinate_frame())

        os.makedirs(self.processed_file_name(), exist_ok=True)
        if THREADS == 0 or self.DEBUG_PLOT:
            for d in tqdm(self.unprocessed_list):
                self.process_impl(d)
        else:
            print(f"process data on {THREADS} threads")
            process_map(
                self.process_impl,
                self.unprocessed_list,
                max_workers=THREADS,
                chunksize=THREADS,
            )
        del self.cmu_params
        del self.var_betas

    def get_filename(self, seq_id_0, fid_0, gender_0, seq_id_1, fid_1, gender_1):
        return (
            f"sid{seq_id_0}-fid{fid_0}-{gender_0}_sid{seq_id_1}-fid{fid_1}-{gender_1}"
        )
        # return f'sid{seq_id_0}-fid{fid_0}_sid{seq_id_1}-{fid_1}_{gender}.pt'

    def process_impl(self, data):
        seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p = data

        filename = self.get_filename(
            seq_id_a, fid_a, gender_a, seq_id_p, fid_p, gender_p
        )
        outputpath = os.path.join(self.processed_file_name(), filename)

        if (os.path.exists(outputpath)) and not self.DEBUG_PLOT:
            # print(f"data {filename} already exists. skip")
            return

        def get_model(seq_id, fid, gender):
            path_model = os.path.join(self.path_model, base_models[gender])
            base_model = load_model(path_model)
            betas = self.var_betas[gender]

            poses = self.cmu_params[seq_id]["poses"]
            poses[:, :3] = 0
            trans = self.cmu_params[seq_id]["trans"]
            beta = betas[random.randrange(betas.shape[0])]
            mesh = model_to_mesh(poses[fid], trans[fid], betas=beta, model=base_model)

            # t2 = o3d.geometry.get_rotation_matrix_from_axis_angle([0,np.pi,0])
            # t1 = o3d.geometry.get_rotation_matrix_from_axis_angle([-np.pi*0.5,0,0])
            # mesh.rotate(t1)
            return mesh

        mesh_a = get_model(seq_id_a, fid_a, gender_a)
        mesh_p = get_model(seq_id_p, fid_p, gender_p)
        mesh_a, mesh_p = normalize_unit_sphere(mesh_a), normalize_unit_sphere(mesh_p)

        # vis.add_geometry(mesh_a)
        # vis.poll_events()
        # vis.update_renderer()
        # vis.remove_geometry(mesh_a)
        # return

        dijkstra_a = get_dijkstra(mesh_a)
        dijkstra_p = get_dijkstra(mesh_p)

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

        def generate_patches(index, dijkstra, pc, distance: float):
            patch_indexes = [np.where(dijkstra[i] < distance)[0] for i in index]
            patches = [pc.select_by_index(idxs) for idxs in patch_indexes]
            dij_local = [dijkstra[idxs, :][:, idxs] for idxs in patch_indexes]
            patches = list(zip(patches, dij_local))
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

        frame_id_a = "{}_{}".format(seq_id_a, fid_a)
        frame_id_p = "{}_{}".format(seq_id_p, fid_p)

        result = {
            "a": graphed_patches_a,
            "p": graphed_patches_p,
            "n": graphed_patches_n,
            "scene_a": frame_id_a,
            "scene_p": frame_id_p,
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

        if self.DEBUG_PLOT:
            vis_graph_with_same_idx([mesh_a, mesh_p])
            visualize_graph_shape(shape_p)
            visualize_graph_shape(shape_a)
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
            # raw[self.file_list[i]+ '.pcd'] = pc
            # data.append(result)
        torch.save(result, outputpath)


if __name__ == "__main__":
    from configs import training_configs, SURREAL_MODEL_DIR, SURREAL_DATA_DIR

    settings = training_configs()
    path_data = "/home/data/dataset/SURREAL/smpl_data/"
    path_model = "/home/data/dataset/SMPL/smpl/models"
    path_data = SURREAL_DATA_DIR
    path_model = SURREAL_MODEL_DIR
    dataset = SURREALDataset(
        path_data,
        path_model,
        train=True,
        mode=settings.data_mode,
    )
    pass
