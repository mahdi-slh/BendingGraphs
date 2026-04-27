import open3d as o3d
import numpy as np
import copy
import os
from tqdm import tqdm
from torch.utils.data import Dataset


from sklearn.neighbors import NearestNeighbors


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
