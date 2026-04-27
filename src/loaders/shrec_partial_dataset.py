"""
https://www.dais.unive.it/~shrec2016/dataset.php
	SHREC 2016: PARTIAL MATCHING OF DEFORMABLE SHAPES

"""
import torch, glob
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from utils.pointcloud_utils import move_to_center
from configs import GRAPH_SIZE, SHREC_PARTIAL_DIR, GEODESIC_CUT
import random
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from smpl_webuser.serialization import load_model
import pickle as pkl
from tqdm.contrib.concurrent import process_map
from loaders.matching_dataset_base import Matching_Dataset
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from utils.mesh_utils import compute_geod, get_dijkstra, normalize_unit_sphere
from utils.visualize_graph import visualize_graph_shape
from scipy.spatial import cKDTree
from utils.utils import simplify_mesh, vis_graph_with_same_idx
import glob
import scipy.io as sio

base_model_names = [
    "cat",
    "centaur",
    "david",
    "dog",
    "gorilla",
    "horse",
    "michael",
    "victoria",
    "wolf",
]

nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


def load_mat2mesh(path):
    # load mat
    mat = sio.loadmat(path)
    tris = mat["surface"]["TRIV"][0][0] - 1
    vts = np.concatenate(
        (
            mat["surface"]["X"][0][0],
            mat["surface"]["Y"][0][0],
            mat["surface"]["Z"][0][0],
        ),
        axis=1,
    )

    x = o3d.geometry.TriangleMesh()
    x.vertices = o3d.utility.Vector3dVector(np.copy(vts))
    x.triangles = o3d.utility.Vector3iVector(np.copy(tris))
    return x


class ShrecPartialDataset(Matching_Dataset):
    def __init__(self, root, train, mode, target_shape: str = "random"):
        super().__init__(train, "cuts")
        self.DEBUG_PLOT = True
        self.DEBUG_PLOT = False
        self.root = root
        self.THREADS = 4
        """generate data"""
        self.num_samples = 200
        self.target_points = 1024 * 8
        self.base_model_name = target_shape
        assert self.base_model_name in base_model_names + ["random"]

        """process data"""
        self.process_list, self.unprocessed_list = self.get_process_scan_list()
        self.process()
        self.process_list += self.unprocessed_list

    def __getitem__(self, idx):
        data = self.process_list[idx]
        filename = self.get_filename(*data)
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

        meta = {k: temp_data[k] for k in temp_data.keys() - ["a", "p", "n"]}
        return frame_data, meta
        # return self.data[idx], self.meta[idx]

    def __len__(self):
        return len(self.process_list)

    def processed_folder_name(self):
        return self.root + "processed/" + self.mode + self.test_train + "/"

    def processed_file_name(self):
        return self.processed_folder_name() + "ShrecPartial_p{}_r{}_{}".format(
            self.points_per_shape, str(GEODESIC_CUT), "trval" if self.train else "test"
        )

    def get_filename(self, cls_a, idx_a, cls_p, idx_p):
        return f"cls{cls_a}-id{idx_a}_cls{cls_p}-id{idx_p}"

    def get_process_scan_list(self):
        process_list = []
        if not self.DEBUG_PLOT:
            created_files = glob.glob(self.processed_file_name() + "/*")
            for path in created_files:
                if len(process_list) == self.num_samples:
                    break
                name = path.split("/")[-1]
                a, p = name.split("_")
                cls_a = a.split("-")[0][3:]
                idx_a = int(a.split("-")[1][2:])
                cls_p = p.split("-")[0][3:]
                idx_p = int(p.split("-")[1][2:])

                if self.base_model_name != "random":
                    if cls_a != self.base_model_name:
                        continue
                    if cls_p != self.base_model_name:
                        continue
                process_list.append((cls_a, idx_a, cls_p, idx_p))

        unprocessed_list = []

        # get all classes
        files = glob.glob(os.path.join(self.root, self.mode, "*.off"))
        filenames = [f.split("/")[-1].split(".")[0] for f in files]
        classnames = [name.split("_")[1] for name in filenames]
        classnames = sorted(np.unique(classnames))

        # sequence dict
        seqmap = defaultdict(list)
        for name in classnames:
            for path in filenames:
                if name in path:
                    seqmap[name].append(path)
            seqmap[name] = sorted(seqmap[name])

        """if it's test, generate pair for the first one to all others"""
        if not self.train:
            if self.base_model_name == "random":
                raise NotImplementedError()
            classname = self.base_model_name
            namelist = seqmap[self.base_model_name]

            for i in range(1, len(namelist)):
                key = (classname, 0, classname, i)
                if key in process_list and not self.DEBUG_PLOT:
                    process_list.append(key)
                else:
                    unprocessed_list.append(key)
        else:
            while True:
                if len(process_list) + len(unprocessed_list) == self.num_samples:
                    break
                if self.base_model_name == "random":
                    cls_a = random.choice(base_model_names)
                    cls_p = cls_a
                else:
                    cls_a = cls_p = self.base_model_name
                idx_a = random.choice(range(len(seqmap[cls_a])))
                idx_p = random.choice(range(len(seqmap[cls_p])))
                key = (cls_a, idx_a, cls_p, idx_p)
                if key in process_list and not self.DEBUG_PLOT:
                    process_list.append(key)
                else:
                    unprocessed_list.append(key)

        return process_list, unprocessed_list

    def process(self):
        os.makedirs(self.processed_file_name(), exist_ok=True)

        # get all classes
        files = glob.glob(os.path.join(self.root, self.mode, "*.off"))
        filenames = [f.split("/")[-1].split(".")[0] for f in files]
        classnames = [name.split("_")[1] for name in filenames]
        classnames = np.unique(classnames)

        # sequence dict
        seqmap = defaultdict(list)
        for name in classnames:
            for path in filenames:
                if name in path:
                    seqmap[name].append(path)
            seqmap[name] = sorted(seqmap[name])
        self.seqmap = seqmap

        if self.THREADS == 0 or self.DEBUG_PLOT:
            for d in tqdm(self.unprocessed_list):
                self.process_impl(d)
        else:
            print(f"process data on {self.THREADS} self.THREADS")
            process_map(
                self.process_impl, self.unprocessed_list, max_workers=self.THREADS
            )
        del self.seqmap

    def process_impl(self, key):
        (cls_a, idx_a, cls_p, idx_p) = key
        filename = self.get_filename(cls_a, idx_a, cls_p, idx_p)
        outputpath = os.path.join(self.processed_file_name(), filename)
        if (os.path.exists(outputpath)) and not self.DEBUG_PLOT:
            return

        mesh_p = o3d.io.read_triangle_mesh(
            os.path.join(self.root, self.mode, self.seqmap[cls_a][idx_a] + ".off")
        )
        mesh_a = o3d.io.read_triangle_mesh(
            os.path.join(self.root, "null", cls_a + ".off")
        )
        if self.target_points > 0:
            mesh_a = mesh_a.simplify_quadric_decimation(self.target_points)
            mesh_a.remove_non_manifold_edges()
            mesh_a.remove_unreferenced_vertices()
            mesh_a.remove_duplicated_vertices()
            mesh_a.remove_degenerate_triangles()

            mesh_p = mesh_p.simplify_quadric_decimation(self.target_points // 2)
            mesh_p.remove_non_manifold_edges()
            mesh_p.remove_unreferenced_vertices()
            mesh_p.remove_duplicated_vertices()
            mesh_p.remove_degenerate_triangles()

            # mesh_a, mesh_p = simplify_mesh([mesh_a,mesh_p], self.target_points, False)

            # indices = np.asarray( mesh_a.get_non_manifold_edges()).tolist()
            # mesh_a.remove_triangles_by_index(indices)
            # mesh_p.remove_triangles_by_index(indices)
        if self.DEBUG_PLOT:
            vis_graph_with_same_idx([mesh_a, mesh_p])
        # return
        # try:
        result = self.generate(
            mesh_a,
            mesh_p,
            name_a=str(cls_a) + "_" + str(idx_a),
            name_p=str(cls_p) + "_" + str(idx_p),
        )
        torch.save(result, outputpath)
        del mesh_a
        del mesh_p
        del result
        # except:
        #     pass


if __name__ == "__main__":
    from configs import training_configs, SHREC_PARTIAL_DIR

    settings = training_configs()

    dataset = ShrecPartialDataset(
        SHREC_PARTIAL_DIR, train=False, mode=settings.data_mode, target_shape="horse"
    )
    pass
