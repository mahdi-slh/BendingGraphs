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
from utils.mesh_utils import normalize_unit_sphere
from smpl_webuser.serialization import load_model
import pickle as pkl
from tqdm.contrib.concurrent import process_map
from loaders.matching_dataset_base import Matching_Dataset
from utils.utils import vis_graph_with_same_idx
from utils.utils import simplify_mesh
import scipy.io as sio

nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
closest_neig = KNeighborsClassifier(n_neighbors=1)


def read_txt_to_list(file):
    output = []
    with open(file, "r") as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output


class SHREC19Dataset(Matching_Dataset):
    def __init__(self, root, train, mode):
        super().__init__(train, mode)
        self.DEBUG_PLOT = False
        # self.DEBUG_PLOT = True
        self.root = root
        self.THREADS = 0
        self.target_points = 1024 * 4
        """generate data"""
        self.num_samples = 10

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

        meta = {k: temp_data[k] for k in temp_data.keys() - ["a", "p", "n"]}
        return frame_data, meta

        return self.data[idx], self.meta[idx]

    def __len__(self):
        return len(self.process_list)

    def processed_folder_name(self):
        return self.root + "processed/" + self.mode + self.test_train + "/"

    def processed_file_name(self):
        return self.processed_folder_name() + "SHREC19_p{}_r{}_{}".format(
            self.points_per_shape, str(GEODESIC_CUT), "trval" if self.train else "test"
        )

    def get_filename(self, idx_a, idx_p):
        return f"id{idx_a}_id{idx_p}"

    def download(self):
        pass

    def get_process_scan_list(self):
        process_list = []
        unprocessed_list = []
        cls_idx_counter_a = dict()
        cls_idx_counter_p = dict()

        if self.train:
            raise NotImplementedError()
        else:
            fpair_list = self.root + "PAIRS_list_SHREC19_connectivity.txt"
            pair_list = read_txt_to_list(fpair_list)
            p_pair_list = list()
            for pair in pair_list:
                a, b = pair.split(",")
                p_pair_list.append((int(a), int(b)))

            if not self.DEBUG_PLOT:
                created_files = glob.glob(self.processed_file_name() + "/*")
                for path in created_files:
                    if len(process_list) >= self.num_samples:
                        break
                    name = path.split("/")[-1]
                    a, p = name.split("_")
                    idx_a = int(a[2:])
                    idx_p = int(p[2:])
                    process_list.append((idx_a, idx_p))
            for key in p_pair_list:
                if len(process_list) + len(unprocessed_list) >= self.num_samples:
                    break
                if key in process_list:
                    continue
                unprocessed_list.append(key)

        # for _ in range(self.num_samples):
        #     if len(process_list)+len(unprocessed_list) == self.num_samples: break
        #     classname_a = random.choice(base_model_names) if self.base_model_name == 'random' else self.base_model_name
        #     if classname_a in cls_idx_counter_a:
        #         cls_idx_counter_a[classname_a]+=1
        #     else:
        #         cls_idx_counter_a[classname_a]=0
        #     idx_a = cls_idx_counter_a[classname_a]

        #     classname_p = random.choice(base_model_names) if self.base_model_name == 'random' else self.base_model_name
        #     if classname_p in cls_idx_counter_p:
        #         cls_idx_counter_p[classname_p]+=1
        #     else:
        #         cls_idx_counter_p[classname_p]=0
        #     idx_p = cls_idx_counter_p[classname_p]

        #     unprocessed_list.append(
        #         (classname_a,idx_a,classname_p,idx_p)
        #         )
        return process_list, unprocessed_list

    def process(self):
        os.makedirs(self.processed_file_name(), exist_ok=True)
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

    def process_impl(self, data):
        idx_a, idx_p = data
        filename = self.get_filename(idx_a, idx_p)
        outputpath = os.path.join(self.processed_file_name(), filename)
        if (os.path.exists(outputpath)) and not self.DEBUG_PLOT:
            return
        # fcorr = os.path.join(self.root, 'matches','FARMgt_txt',str(idx_a)+'_'+str(idx_p)+'.txt')
        # corrs = read_txt_to_list(fcorr)
        # corrs_a = {idx:int(idx) for idx,_ in enumerate(corrs)}
        # corrs_p = {idx:int(id) for idx, id in enumerate(corrs)}
        # corrs = [corrs_a, corrs_p]

        def get_model(idx):
            mat = sio.loadmat(os.path.join(self.root, "mat", str(idx) + ".mat"))
            data = mat["Shape_df"][0][0]
            vts = data[0]
            tris = data[1] - 1
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.copy(vts))
            mesh.triangles = o3d.utility.Vector3iVector(np.copy(tris))

            mesh = move_to_center(mesh)  # normalize_unit_sphere
            # t1 = o3d.geometry.get_rotation_matrix_from_axis_angle([0,0,-np.pi*0.5])
            # mesh.rotate(t1)
            return mesh

        mesh_a = get_model(idx_a)
        mesh_p = get_model(idx_p)
        if self.target_points > 0:
            mesh_a = simplify_mesh([mesh_a], self.target_points, False)[0]
            mesh_p = simplify_mesh([mesh_p], self.target_points, False)[0]
            mesh_a.remove_non_manifold_edges()
            mesh_p.remove_non_manifold_edges()
            # mesh_a, mesh_p = simplify_mesh([mesh_a,mesh_p], self.target_points, False, map_lists=corrs)

        # if self.DEBUG_PLOT:
        #     vis_graph_with_same_idx([mesh_a,mesh_p])
        #     return

        try:
            result = self.generate(mesh_a, mesh_p, name_a=str(idx_a), name_p=str(idx_p))
            torch.save(result, outputpath)
        except:
            pass


if __name__ == "__main__":
    from configs import training_configs, SHREC19_DIR

    settings = training_configs()
    # path = SMAL_DIR#'/media/sc/SSD1TB/dataset/smal/smal_online_V1.0/'
    dataset = SHREC19Dataset(SHREC19_DIR, train=False, mode=settings.data_mode)
    pass
