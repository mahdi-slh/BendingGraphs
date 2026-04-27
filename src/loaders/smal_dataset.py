"""
dataset: http://SMAL.cs.technion.ac.il/book/resources_data.html

"""
import torch, glob
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from utils.pointcloud_utils import move_to_center
from configs import GRAPH_SIZE, GEODESIC_CUT
import random
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from smpl_webuser.serialization import load_model
import pickle as pkl
from tqdm.contrib.concurrent import process_map
from loaders.matching_dataset_base import Matching_Dataset
from utils.utils import vis_graph_with_same_idx
from utils.utils import simplify_mesh
from utils.mesh_utils import normalize_unit_sphere

NAMEMAP = {"cat": 0, "dog": 1, "horse": 2, "cow": 3, "hippos": 4}
base_model_names = ["cat", "dog", "horse", "cow", "hippos"]


nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


class SMALDataset(Matching_Dataset):
    def __init__(self, root, train, mode, target_shape: str = "random"):
        super().__init__(train, mode)
        self.DEBUG_PLOT = False
        # self.DEBUG_PLOT = True
        self.root = root
        # self.THREADS=0
        self.target_points = 1024 * 13
        """generate data"""
        if train:
            self.num_samples = 100
            self.base_model_name = target_shape
        else:
            self.num_samples = 3
        assert self.base_model_name in base_model_names + ["random"]

        """process data"""
        self.process_list, self.unprocessed_list = self.get_process_scan_list()
        self.process()
        self.process_list += self.unprocessed_list

    @property
    def smal_model_name(self):
        return "smal_CVPR2017.pkl"

    @property
    def smal_data_name(self):
        return "smal_CVPR2017_data.pkl"

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

        meta = {
            k: temp_data[k]
            for k in temp_data.keys() - ["a", "p", "n", "scene_a", "scene_p"]
        }
        return frame_data, meta

        return self.data[idx], self.meta[idx]

    def __len__(self):
        return len(self.process_list)


    def processed_folder_name(self):
        return self.root + "processed/" + self.mode + self.test_train + "/"

    def processed_file_name(self):
        return self.processed_folder_name() + "SMAL_p{}_r{}_{}".format(
            self.points_per_shape, str(GEODESIC_CUT), "trval" if self.train else "test"
        )

    def get_filename(self, cls_a, idx_a, cls_p, idx_p):
        return f"cls{cls_a}-id{idx_a}_cls{cls_p}-id{idx_p}"

    def download(self):
        pass

    def get_process_scan_list(self):
        process_list = []
        cls_idx_counter_a = dict()
        cls_idx_counter_p = dict()
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

                if cls_a not in cls_idx_counter_a:
                    cls_idx_counter_a[cls_a] = idx_a
                if cls_p not in cls_idx_counter_p:
                    cls_idx_counter_p[cls_p] = idx_p
                if cls_idx_counter_p[cls_p] < idx_p:
                    cls_idx_counter_p[cls_p] = idx_p
                if cls_idx_counter_a[cls_a] < idx_a:
                    cls_idx_counter_a[cls_a] = idx_a

                if self.base_model_name != "random":
                    if cls_a != self.base_model_name:
                        continue
                    if cls_p != self.base_model_name:
                        continue

                process_list.append((cls_a, idx_a, cls_p, idx_p))

        unprocessed_list = []
        for _ in range(self.num_samples):
            if len(process_list) + len(unprocessed_list) == self.num_samples:
                break
            classname_a = (
                random.choice(base_model_names)
                if self.base_model_name == "random"
                else self.base_model_name
            )
            if classname_a in cls_idx_counter_a:
                cls_idx_counter_a[classname_a] += 1
            else:
                cls_idx_counter_a[classname_a] = 0
            idx_a = cls_idx_counter_a[classname_a]

            classname_p = (
                random.choice(base_model_names)
                if self.base_model_name == "random"
                else self.base_model_name
            )
            if classname_p in cls_idx_counter_p:
                cls_idx_counter_p[classname_p] += 1
            else:
                cls_idx_counter_p[classname_p] = 0
            idx_p = cls_idx_counter_p[classname_p]

            unprocessed_list.append((classname_a, idx_a, classname_p, idx_p))
        return process_list, unprocessed_list

    def process(self):
        os.makedirs(self.processed_file_name(), exist_ok=True)

        with open(os.path.join(self.root, self.smal_data_name), "rb") as f:
            self.model_data = pkl.load(f, encoding="latin1")

        if self.THREADS == 0 or self.DEBUG_PLOT:
            pbar = tqdm(self.unprocessed_list)
            for d in pbar:
                pbar.set_description("process {}".format(d))
                self.process_impl(d)
        else:
            print(f"process data on {self.THREADS} self.THREADS")
            process_map(
                self.process_impl,
                self.unprocessed_list,
                max_workers=self.THREADS,
                chunksize=self.THREADS,
            )

        del self.model_data

    def process_impl(self, data):
        cls_a, idx_a, cls_p, idx_p = data
        filename = self.get_filename(cls_a, idx_a, cls_p, idx_p)
        outputpath = os.path.join(self.processed_file_name(), filename)
        if (os.path.exists(outputpath)) and not self.DEBUG_PLOT:
            return

        def get_model(classname):
            betas_var = 0.2
            pose_var = 0.15
            path = os.path.join(self.root, self.smal_model_name)
            model = load_model(path)
            betas = self.model_data["cluster_means"][NAMEMAP[classname]]
            model.betas[:] = betas
            model.pose[
                :
            ] = 0.0  # s the relative rotation of the N = 33 joints in the kinematic tree
            model.trans[:] = 0.0  # s the global translation applied to the root joint
            tmp = np.random.normal(0.0, betas_var, size=model.betas.shape)
            model.betas[:] = betas + tmp
            tmp = np.random.normal(0.0, pose_var, size=model.pose.shape)
            model.pose[:] = tmp

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.copy(model.r))
            mesh.triangles = o3d.utility.Vector3iVector(np.copy(model.f))
            mesh = move_to_center(mesh)  # normalize_unit_sphere
            t1 = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, -np.pi * 0.5])
            mesh.rotate(t1)
            return mesh

        mesh_a = get_model(cls_a)
        mesh_p = get_model(cls_p)

        # o3d.io.write_triangle_mesh("smal_hourse_a.ply",mesh_a)
        # o3d.io.write_triangle_mesh("smal_hourse_p.ply",mesh_p)

        if self.target_points > 0:
            mesh_a, mesh_p = simplify_mesh([mesh_a, mesh_p], self.target_points, False)

        # if self.DEBUG_PLOT:
        #     vis_graph_with_same_idx([mesh_a,mesh_p])
        #     return

        result = self.generate(mesh_a, mesh_p)
        torch.save(result, outputpath)


if __name__ == "__main__":
    from configs import training_configs, SMAL_DIR

    settings = training_configs()
    path = SMAL_DIR  #'/media/sc/SSD1TB/dataset/smal/smal_online_V1.0/'
    dataset = SMALDataset(
        path, train=True, mode=settings.data_mode, target_shape="horse"
    )
    pass
