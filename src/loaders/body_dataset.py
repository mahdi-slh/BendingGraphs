import open3d as o3d
import numpy as np
import copy
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from loaders.augmentations import random_perturb
from graph import *
from configs import *
import timeit

from sklearn.neighbors import NearestNeighbors


SUBSAMPLE_SIZE = NUM_POINTS_SAMPLE_FAUST

nbrs_knn = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")


def create_triplet_patches(pc1, pc2, iterations, purturb=False):
    patches_anchor = []
    patches_negative = []
    patches_positive = []
    for i in range(iterations):
        index_rand = np.random.randint(0, len(pc1.points))

        pc1_np = np.asarray(pc1.points)
        pc2_np = np.asarray(pc2.points)
        nbrs_knn.fit(pc1_np)
        dist, labels1 = nbrs_knn.kneighbors(
            np.expand_dims(pc1_np[index_rand, :], 0), return_distance=True
        )

        avg_dist = (np.mean(dist) + np.max(dist)) / 2
        # print(dist>avg_dist)
        index_hard_neg = np.random.choice(labels1[dist > avg_dist], 1)[0]
        labels_hard_neg = nbrs_knn.kneighbors(
            np.expand_dims(pc1_np[index_hard_neg, :], 0), return_distance=False
        )

        nbrs_knn.fit(pc2_np)
        labels2 = nbrs_knn.kneighbors(
            np.expand_dims(pc2_np[index_rand, :], 0), return_distance=False
        )

        patch_anchor_np = pc1_np[labels1[0, :], :]
        patch_negative_np = pc1_np[labels_hard_neg[0, :], :]
        patch_positive_np = pc2_np[labels2[0, :], :]

        patch_anchor_o3d = o3d.geometry.PointCloud()
        patch_positive_o3d = o3d.geometry.PointCloud()
        patch_negative_o3d = o3d.geometry.PointCloud()

        patch_anchor_o3d.points = o3d.utility.Vector3dVector(patch_anchor_np)
        patch_positive_o3d.points = o3d.utility.Vector3dVector(patch_positive_np)
        patch_negative_o3d.points = o3d.utility.Vector3dVector(patch_negative_np)

        patches_anchor.append(patch_anchor_o3d)
        patches_negative.append(patch_negative_o3d)
        patches_positive.append(patch_positive_o3d)
    return patches_anchor, patches_positive, patches_negative


class BodyDataset(Dataset):
    def __init__(self, root, train, mode):

        self.root = root
        file_list_pose_full = sorted(os.listdir(root + "pose_ms/"))
        file_list_neut_full = [
            "_".join(n.split("_")[:-1]) + ".obj" for n in file_list_pose_full
        ]
        train_val_mask = [index % 5 < 4 for index in range(len(file_list_neut_full))]
        self.file_list_neut = [
            file_list_neut_full[i]
            for i in range(len(file_list_neut_full))
            if train_val_mask[i] == train
        ]
        self.file_list_pose = [
            file_list_pose_full[i]
            for i in range(len(file_list_pose_full))
            if train_val_mask[i] == train
        ]

        self.test_train = "train/" if train else "test/"
        self.mode = mode + "/"
        self.train = train

        self.processed_paths = self.processed_file_names()
        self.raw_paths = self.processed_file_raw_folders()

        self.process()

        self.data = []
        self.raw = {}

        path = self.processed_paths
        temp_data = torch.load(path)

        triplet_data = []

        for j in range(len(temp_data)):
            if self.train == False:
                temp_data[j]["raw"] = o3d.io.read_point_cloud(
                    self.raw_paths + temp_data[j]["name"], format="auto"
                )

            len_patches = len(temp_data[j]["a"])

            random_perm = np.random.permutation(len_patches // 2)
            for k in range(len(temp_data[j]["a"])):
                triplet_data.append(temp_data[j]["a"][k])
                triplet_data.append(temp_data[j]["p"][k])
                if k < len_patches // 4000:
                    triplet_data.append(temp_data[j]["n"][random_perm[k]])

                else:
                    triplet_data.append(temp_data[j]["n"][k])

        self.data.extend(triplet_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def processed_file_names(self):
        return (
            self.root
            + "processed/"
            + self.mode
            + self.test_train
            + "Body_-{}.pt".format(str(GRAPH_SIZE))
        )

    def processed_file_raw_folders(self):
        return (
            self.root
            + "processed/"
            + self.mode
            + "raw/"
            + self.test_train
            + "{}".format(str(GRAPH_SIZE))
        )

    def download(self):
        pass

    def process(self):

        path = self.processed_paths
        raw_path = self.raw_paths

        base_path_pose = self.root + "pose_ms/"
        base_path_neut = self.root + "neutral_ms/"

        if os.path.exists(path):
            print("Dataset already exists. Loading...")
            return
        else:
            print("Preparing dataset...")

        data = []
        raw = {}

        count = 0

        for i in tqdm(range(len(self.file_list_pose))):

            time_a = timeit.default_timer()
            file_name_neut = self.file_list_neut[i]
            file_name_pose = self.file_list_pose[i]
            count += 1

            mesh_neut = o3d.io.read_triangle_mesh(base_path_neut + file_name_neut)
            mesh_pose = o3d.io.read_triangle_mesh(base_path_pose + file_name_pose)
            pc_neut = o3d.geometry.PointCloud()
            pc_neut.points = mesh_neut.vertices

            pc_pose = o3d.geometry.PointCloud()
            pc_pose.points = mesh_pose.vertices

            np.random.seed(i)
            indexes_random = np.random.permutation(len(pc_pose.points))[:SUBSAMPLE_SIZE]
            pc_neut = pc_neut.select_by_index(indexes_random)
            pc_pose = pc_pose.select_by_index(indexes_random)
            if self.train:
                pc_pose = random_perturb(pc_pose, False, sigma=0.005, clip=0.01)
                pc_neut = random_perturb(pc_neut, False, sigma=0.005, clip=0.01)

            patches_a, patches_p, patches_n = create_triplet_patches(
                pc_neut, pc_pose, 100
            )
            # time_b = timeit.default_timer()
            graphed_patches_a = [
                create_local_graph(x, -1, True, self.train)
                for x in patches_a
                if len(x.points) == GRAPH_SIZE
            ]
            graphed_patches_p = [
                create_local_graph(x, -1, True, self.train)
                for x in patches_p
                if len(x.points) == GRAPH_SIZE
            ]
            graphed_patches_n = [
                create_local_graph(x, -1, True, self.train)
                for x in patches_n
                if len(x.points) == GRAPH_SIZE
            ]
            # time_c = timeit.default_timer()

            # for i in range(len(graphed_patches_a)):
            #     visualize_torch_graphs(graphed_patches_a[i],graphed_patches_p[i],graphed_patches_n[i])
            for i_hash in range(len(graphed_patches_a)):
                hash_code = random.getrandbits(16)
                graphed_patches_a[i_hash]["hash"] = hash_code
                graphed_patches_p[i_hash]["hash"] = hash_code
                graphed_patches_n[i_hash]["hash"] = hash_code

            result = {
                "a": graphed_patches_a,
                "p": graphed_patches_p,
                "n": graphed_patches_n,
                "name": self.file_list_pose[i] + ".pcd",
            }

            raw[self.file_list_pose[i] + ".pcd"] = pc_pose
            data.append(result)

            # print(count)

        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        for k in raw.keys():
            o3d.io.write_point_cloud(raw_path + k, raw[k])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        # time_d = timeit.default_timer()
        torch.save(data, path)
        # time_e = timeit.default_timer()
        # print('Time: ', time_b - time_a,time_c - time_b,time_d - time_c,time_e - time_d)


if __name__ == "__main__":
    root = "../../../dataset/transformed_body/"
    file_list_pose = sorted(os.listdir(root + "pose_ms/"))
    file_list_neut = [
        n.split("_")[0] + "_" + n.split("_")[1] + ".obj" for n in file_list_pose
    ]
    for file_name in tqdm(file_list_pose):
        print(file_name)

    mesh1 = o3d.io.read_triangle_mesh(root + "neutral_ms/" + file_list_neut[800])
    mesh2 = o3d.io.read_triangle_mesh(root + "pose_ms/" + file_list_pose[800])

    mesh1.translate([0, 0, 1])

    pc1 = o3d.geometry.PointCloud()
    pc1.points = mesh1.vertices
    pc1_np = np.asarray(pc1.points)
    pc2 = o3d.geometry.PointCloud()
    pc2.points = mesh2.vertices
    pc2_np = np.asarray(pc2.points)

    nbrs_knn = NearestNeighbors(n_neighbors=225, algorithm="ball_tree")

    for i in range(10):

        pc1.paint_uniform_color([0, 0, 0])
        pc2.paint_uniform_color([0, 0, 0])
        index = np.random.randint(0, len(pc1.points))
        color = copy.deepcopy(np.asarray(pc1.colors))

        pc1_np = np.asarray(pc1.points)
        pc2_np = np.asarray(pc2.points)
        nbrs_knn.fit(pc1_np)
        dist, labels1 = nbrs_knn.kneighbors(
            np.expand_dims(pc1_np[index, :], 0), return_distance=True
        )
        index_hard_neg = np.random.choice(labels1[labels1 > np.mean(labels1)], 1)[0]
        labels_hard_neg = nbrs_knn.kneighbors(
            np.expand_dims(pc1_np[index_hard_neg, :], 0), return_distance=False
        )

        color[labels_hard_neg, :] = [0, 1, 0]
        color[labels1, :] = [0, 0, 1]
        color[index, :] = [1, 0, 0]
        pc1.colors = o3d.utility.Vector3dVector(color)

        nbrs_knn.fit(pc2_np)
        labels2 = nbrs_knn.kneighbors(
            np.expand_dims(pc2_np[index, :], 0), return_distance=False
        )
        color2 = copy.deepcopy(np.asarray(pc2.colors))

        color2[labels2, :] = [0, 0, 1]
        color2[index, :] = [1, 0, 0]
        pc2.colors = o3d.utility.Vector3dVector(color2)

        o3d.visualization.draw_geometries([pc1, pc2])
