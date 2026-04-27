import open3d as o3d
import numpy as np
import copy
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from loaders.augmentations import random_perturb
from graph import *
from configs import *
import open3d as o3d

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from utils.mesh_utils import compute_geod, get_dijkstra

from utils.visualize_graph import visualize_graph_shape
from utils.pyFM.mesh import TriMesh


nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


class FaustDataset(Dataset):
    def __init__(
        self,
        root,
        train,
        mode,
        frame_range=None,
        sample_per_frame=None,
        canon_scan="canon",
        test_folder="training/",
    ):

        self.root = root
        # set test_folder to 'test' for training on un-annotated test folder
        self.test_train = "training/" if train else test_folder
        self.test_folder = test_folder
        self.canon_scan = canon_scan

        if train:
            self.frame_ids_list = range(5)
            self.points_per_shape = 200
        else:
            self.frame_ids_list = range(90, 99)
            self.points_per_shape = 200  # -1

        self.sample_per_frame = 1 if train else 1
        if sample_per_frame:
            self.sample_per_frame = sample_per_frame
        if frame_range:
            self.frame_ids_list = frame_range

        self.mode = mode + "/"
        self.train = train

        self.processed_paths = self.processed_file_names()
        self.raw_paths = self.processed_file_raw_folders()

        self.process()

        self.data = []
        self.meta = []
        self.raw = {}

        path = self.processed_paths
        temp_data = torch.load(path)

        for j in range(len(temp_data)):
            frame_data = {}

            frame_data["a"] = temp_data[j]["a"]
            frame_data["p"] = temp_data[j]["p"]

            len_patches = len(temp_data[j]["a"])
            random_proportion = 1
            np.random.seed(4)
            random_perm = np.random.permutation(len_patches // random_proportion)
            nordered_perm = np.arange(len_patches)
            if temp_data[j]["n"]:
                frame_data["n"] = []
                # frame_data['n'].extend([temp_data[j]['n'][id] for id in random_perm[:len_patches//random_proportion]])
                frame_data["n"].extend(
                    [temp_data[j]["n"][id] for id in random_perm]
                )  # [len_patches//random_proportion:]])

            self.data.extend([frame_data])
            self.meta.append(
                {k: temp_data[j][k] for k in temp_data[j].keys() - ["a", "p", "n"]}
            )

    def __getitem__(self, idx):
        return self.data[idx], self.meta[idx]

    def __len__(self):
            return len(self.data)


    def processed_file_names(self):
        fr_s, fr_e, fr_r, p_f = (
            self.frame_ids_list[0],
            self.frame_ids_list[-1],
            self.sample_per_frame,
            self.points_per_shape,
        )
        return (
            self.root
            + "processed/"
            + self.mode
            + self.test_train
            + "FAUST_{}_p{}_r{}_({}-{})x{}_{}.pt".format(
                self.canon_scan,
                p_f,
                str(GEODESIC_CUT),
                fr_s,
                fr_e,
                fr_r,
                "trval" if self.train else "test",
            )
        )

    def processed_file_raw_folders(self):
        return (
            self.root
            + "processed/"
            + self.mode
            + "raw/"
            + self.test_train
            + "{}_{}".format(str(GEODESIC_CUT), "trval" if self.train else "test")
        )

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

    def pc_canon_scan(self, frame):
        if self.canon_scan == "scan":
            return frame["pc_scan"]
        elif self.canon_scan == "canon":
            return frame["pc_canon"]

    def pc_np_canon_scan(self, frame):
        if self.canon_scan == "scan":
            return frame["pc_scan_np"]
        elif self.canon_scan == "canon":
            return frame["pc_canon_np"]

    def init_frame(self):
        frame = {}
        frame["pc_scan"] = o3d.geometry.PointCloud()
        frame["pc_canon"] = o3d.geometry.PointCloud()
        return frame

    def process(self):

        path = self.processed_paths

        if os.path.exists(path):
            print("Dataset already exists. Loading...")
            return
        else:
            print("Preparing dataset...")

        data = []
        raw = {}

        count = 0
        augment = True

        for frame_id_orig in tqdm(self.frame_ids_list):
            for _ in range(self.sample_per_frame):

                frame_id_targ = frame_id_orig + 1
                frame_a = self.init_frame()
                frame_p = self.init_frame()
                file_canon_a = f"tr_reg_{frame_id_orig:03d}.ply"
                file_scan_a = f"tr_scan_{frame_id_orig:03d}.off"
                file_canon_a_off = f"tr_reg_{frame_id_orig:03d}.off"

                if self.mode == "noise/":

                    file_scan_p = f"tr_scan_{frame_id_targ:03d}.off"
                    file_canon_p = f"tr_reg_{frame_id_targ:03d}.ply"
                    file_canon_p_off = f"tr_reg_{frame_id_targ:03d}.off"
                elif self.mode == "inter/":
                    print("not implemented")

                path_canon_a = self.folder_canon + file_canon_a
                mesh_canon_a = o3d.io.read_triangle_mesh(path_canon_a)

                mesh_scan_a = o3d.io.read_triangle_mesh(self.folder_scan + file_scan_a)

                frame_a["pc_scan"].points = mesh_scan_a.vertices
                frame_a["pc_scan_np"] = np.asarray(frame_a["pc_scan"].points)
                dijkstra_a = get_dijkstra(
                    mesh_scan_a if self.canon_scan == "scan" else mesh_canon_a
                )

                frame_a["pc_canon"].points = mesh_canon_a.vertices
                frame_a["pc_canon_np"] = np.asarray(frame_a["pc_canon"].points)
                if self.mode == "noise" and self.train:
                    frame_a["pc_canon"] = random_perturb(
                        frame_a["pc_canon"], False, sigma=0.004, clip=0.004
                    )

                seeds_a_np, indexes_fps_a = fps_points_random(
                    self.pc_np_canon_scan(frame_a),
                    self.points_per_shape,
                    return_indexes=True,
                )

                closest_neig.fit(
                    frame_a["pc_canon_np"], np.arange(frame_a["pc_canon_np"].shape[0])
                )
                closest_canon_index_a = closest_neig.kneighbors(
                    seeds_a_np, return_distance=False
                )

                patches_a = [
                    self.pc_canon_scan(frame_a).select_by_index(
                        np.where(dijkstra_a[i] < GEODESIC_CUT)[0]
                    )
                    for i in indexes_fps_a
                ]

                patch_indexes_a = [
                    np.where(dijkstra_a[i] < GEODESIC_CUT)[0] for i in indexes_fps_a
                ]

                patches_a = [
                    self.pc_canon_scan(frame_a).select_by_index(idxs)
                    for idxs in patch_indexes_a
                ]

                dij_local_a = [dijkstra_a[idxs, :][:, idxs] for idxs in patch_indexes_a]

                patches_a = list(zip(patches_a, dij_local_a))

                # for real scan some lonely points have artifacts with full graph
                valid_points = [
                    len(p[0].points) < len(self.pc_canon_scan(frame_a).points)
                    for p in patches_a
                ]
                indexes_fps_a = [
                    indexes_fps_a[i]
                    for i in range(len(indexes_fps_a))
                    if valid_points[i]
                ]
                if augment:
                    seeds_np_a_shifted = [
                        self.pc_np_canon_scan(frame_a)[random.choice(list_ids), :]
                        for list_ids in patch_indexes_a
                    ]
                    closest_canon_index_a = closest_neig.kneighbors(
                        seeds_np_a_shifted, return_distance=False
                    )

                patches_a = [
                    patches_a[i] for i in range(len(patches_a)) if valid_points[i]
                ]

                seeds_a_np = seeds_a_np[valid_points, :]

                closest_canon_index_a = closest_canon_index_a[valid_points]

                graphed_patches_a = [
                    create_local_graph(x[0], -1, True, self.train, adj_mat=x[1])
                    for x in patches_a
                ]

                mesh_scan_p = o3d.io.read_triangle_mesh(
                    self.root + self.test_train + "scans_off/" + file_scan_p
                )
                path_canon_p = (
                    self.root
                    + self.test_train
                    + (
                        "registrations/"
                        if self.test_folder == "training/"
                        else "scans_off/"
                    )
                    + file_canon_p
                )
                mesh_canon_p = o3d.io.read_triangle_mesh(path_canon_p)
                frame_p["pc_canon"].points = mesh_canon_p.vertices
                frame_p["pc_canon_np"] = np.asarray(frame_p["pc_canon"].points)

                dijkstra_p = get_dijkstra(
                    mesh_scan_p if self.canon_scan == "scan" else mesh_canon_p
                )
                frame_p["pc_scan"].points = mesh_scan_p.vertices
                frame_p["pc_scan_np"] = np.asarray(frame_p["pc_scan"].points)

                seeds_t_a = torch.from_numpy(seeds_a_np)

                if self.train:
                    neg_candidates = [
                        np.where(abs(dijkstra_a[i] - GEODESIC_CUT) * 15 < 4)[0]
                        for i in indexes_fps_a
                    ]

                    neg_candidates = [
                        [np.random.choice(len(dijkstra_a))]
                        if len(candidates) == 0
                        else candidates
                        for candidates in neg_candidates
                    ]
                    index_neg = [
                        np.random.choice(candidates) for candidates in neg_candidates
                    ]

                    # patches_n = [self.pc_canon_scan(frame_a).select_by_index(
                    #     np.where(dijkstra_a[i] < GEODESIC_CUT)[0]) for i in index_neg]

                    patch_indexes_n = [
                        np.where(dijkstra_a[i] < GEODESIC_CUT)[0] for i in index_neg
                    ]
                    patches_n = [
                        self.pc_canon_scan(frame_a).select_by_index(idxs)
                        for idxs in patch_indexes_n
                    ]

                    dij_local_n = [
                        dijkstra_a[idxs, :][:, idxs] for idxs in patch_indexes_n
                    ]
                    patches_n = list(zip(patches_n, dij_local_n))

                    if self.canon_scan == "scan":
                        closest_neig.fit(
                            frame_p["pc_scan_np"],
                            np.arange(frame_p["pc_scan_np"].shape[0]),
                        )
                        closest_to_scan_index = closest_neig.kneighbors(
                            frame_p["pc_canon_np"][closest_canon_index_a[:, 0], :],
                            return_distance=False,
                        )
                    elif self.canon_scan == "canon":
                        closest_to_scan_index = closest_canon_index_a  # np.expand_dims( np.asarray(indexes_fps_a),1)
                    if self.mode == "noise/":
                        pc = random_perturb(
                            self.pc_canon_scan(frame_p), False, sigma=0.004, clip=0.004
                        )

                    # patches_p = [pc.select_by_index(np.where(
                    #     dijkstra_p[i] < GEODESIC_CUT)[0]) for i in closest_to_scan_index[:, 0]]

                    patch_indexes_p = [
                        np.where(dijkstra_p[i] < GEODESIC_CUT)[0]
                        for i in closest_to_scan_index[:, 0]
                    ]
                    patches_p = [pc.select_by_index(idxs) for idxs in patch_indexes_p]

                    dij_local_p = [
                        dijkstra_p[idxs, :][:, idxs] for idxs in patch_indexes_p
                    ]
                    patches_p = list(zip(patches_p, dij_local_p))

                    seeds_p_np = np.asarray(pc.points)[closest_to_scan_index].squeeze(1)

                    # if len(patches_p) <680:
                    #     print('here')
                    seeds_t_p = torch.from_numpy(seeds_p_np)

                    graphed_patches_p = [
                        create_local_graph(x[0], -1, True, self.train, adj_mat=x[1])
                        for x in patches_p
                    ]

                    graphed_patches_n = [
                        create_local_graph(x[0], -1, True, self.train, adj_mat=x[1])
                        for x in patches_n
                    ]

                    indexes_fps_p = indexes_fps_a
                else:

                    seeds_p_np, indexes_fps_p = fps_points(
                        self.pc_np_canon_scan(frame_p),
                        len(indexes_fps_a),
                        return_indices=True,
                    )
                    patch_indexes_p = [
                        np.where(dijkstra_p[i] < GEODESIC_CUT)[0] for i in indexes_fps_p
                    ]
                    patches_p = [
                        self.pc_canon_scan(frame_p).select_by_index(idxs)
                        for idxs in patch_indexes_p
                    ]

                    dij_local_p = [
                        dijkstra_p[idxs, :][:, idxs] for idxs in patch_indexes_p
                    ]
                    patches_p = list(zip(patches_p, dij_local_p))

                    closest_neig.fit(
                        frame_p["pc_canon_np"],
                        np.arange(frame_p["pc_canon_np"].shape[0]),
                    )
                    closest_canon_index_p = closest_neig.kneighbors(
                        seeds_p_np, return_distance=False
                    )

                    closest_neig.fit(
                        frame_a["pc_canon_np"],
                        np.arange(frame_a["pc_canon_np"].shape[0]),
                    )
                    closest_canon_index_a = closest_neig.kneighbors(
                        seeds_a_np, return_distance=False
                    )

                    seeds_t_p = torch.from_numpy(seeds_p_np)

                    graphed_patches_p = [
                        create_local_graph(x[0], -1, True, self.train, adj_mat=x[1])
                        for x in patches_p
                    ]

                    graphed_patches_n = None

                    indexes_fps_p = closest_canon_index_p.squeeze()
                    indexes_fps_a = closest_canon_index_a.squeeze()
                    dijkstra_a = self.get_dijkstra(mesh_canon_a)

                seed_dist = dijkstra_a[indexes_fps_a, :][:, indexes_fps_p]
                dist_mat_t = torch.from_numpy(seed_dist).type(torch.DoubleTensor)
                # dist_mat_t=torch.clamp( torch.from_numpy(seed_dist).type(torch.DoubleTensor),0,20)
                # dist_mat_t = (20 - dist_mat_t)/20
                for i_hash in range(len(graphed_patches_a)):
                    hash_code = random.getrandbits(16)
                    graphed_patches_a[i_hash]["hash"] = hash_code

                    if self.train:
                        graphed_patches_n[i_hash]["hash"] = hash_code
                        graphed_patches_p[i_hash]["hash"] = hash_code
                if self.train:
                    indexes_seeds_p = closest_to_scan_index[:, 0]
                else:
                    indexes_seeds_p = indexes_fps_p.squeeze()
                geod_mat_a = compute_geod(mesh_canon_a, indexes_fps_a)
                geod_mat_p = compute_geod(mesh_canon_p, indexes_seeds_p)
                shape_p = create_shape_graph(
                    seeds_p_np, None, geod_mat_p[indexes_seeds_p, :][:, indexes_seeds_p]
                )
                shape_a = create_shape_graph(
                    seeds_a_np, None, geod_mat_a[indexes_fps_a, :][:, indexes_fps_a]
                )
                visualize_graph_shape(shape_p)
                visualize_graph_shape(shape_a)
                # if self.train:

                result = {
                    "a": graphed_patches_a,
                    "p": graphed_patches_p,
                    "n": graphed_patches_n,
                    "scene_a": file_canon_a_off,
                    "scene_p": file_canon_p_off,
                    "seeds_a": seeds_t_a,
                    "seeds_p": seeds_t_p,
                    "dist_mat": dist_mat_t,
                    "shape_a": shape_a,
                    "shape_p": shape_p,
                }
                if not self.train:
                    path_canon_off_a = (
                        self.root + "training/registrations_off/" + file_canon_a_off
                    )
                    path_canon_off_p = (
                        self.root + "training/registrations_off/" + file_canon_p_off
                    )
                    trimesh_a = TriMesh(path_canon_off_a)
                    trimesh_p = TriMesh(path_canon_off_p)
                    mesh_a_geod = None  # trimesh_a.get_geodesic(verbose=False)

                    result.update(
                        {
                            "seed_index_a": indexes_fps_a,
                            "seed_index_p": indexes_fps_p,
                            "full_points_a": self.pc_np_canon_scan(frame_a),
                            "full_points_p": self.pc_np_canon_scan(frame_p),
                            "mesh_a": trimesh_a,
                            "mesh_p": trimesh_p,
                            "mesh_a_geod": mesh_a_geod,
                            "dij_a": dijkstra_a,
                            "dij_p": dijkstra_p,
                        }
                    )

                # for i in range(len(patches_a)):
                #     patches_a[i][0].paint_uniform_color([0,1,0])
                #     patches_n[i][0].paint_uniform_color([1,0,0])
                #     patches_p[i][0].paint_uniform_color([0,0,1])
                # for i in range(len(graphed_patches_a)):
                #     visualize_torch_graphs_full(graphed_patches_a[i],graphed_patches_p[i],graphed_patches_n[i],seeds_a_np,seeds_p_np)

                # raw[self.file_list_scan[i]+ '.pcd'] = pc_scan
                data.append(result)

                # print(count)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # time_d = timeit.default_timer()
        torch.save(data, path)
        # time_e = timeit.default_timer()
        # print('Time: ', time_b - time_a,time_c - time_b,time_d - time_c,time_e - time_d)
