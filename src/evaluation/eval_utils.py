import numpy as np
import torch

import open3d  as o3d
import tqdm
from utils.visualize_graph import (
    show_from_center,
    show_from_side,
    show_from_top,
    show_from_new,
)
from sklearn.neighbors import NearestNeighbors


import matplotlib.pyplot as plt


def add_image_tb(writter, image, text, step=0):
    if writter:
        # magma = cm.get_cmap('magma')
        # x_transformed = magma(x)
        if type(image).__name__ == "ndarray":
            image = torch.from_numpy(image)
        image_f = image.clone().float()
        max_image = image_f.max()
        min_image = image_f.min()
        if max_image - min_image > 0:

            image_f = (image - min_image) / (max_image - min_image)
        if image_f.ndim == 2:
            image_f.unsqueeze_(0)
        writter.add_image(text, image_f, step)





def test_visualize_desc(settings, net, test_data):

    net.eval()

    descriptors = {x: [] for x in test_data.object_ids}
    for data in tqdm(test_data):

        scene = data["scene"]

        pc_raw = o3d.io.read_point_cloud(
            test_data.root + "fragments/" + scene + data["name"]
        )
        pc_raw_np = np.asarray(pc_raw.points)

        desc = get_desc_from_net(settings, net, data)
        descriptors[scene].append(desc)

        all_descriptors = desc["descriptors"]
        all_keys = np.asarray([k.detach().numpy() for k in desc["positions"]])
        keypoints_all = o3d.geometry.PointCloud()
        keypoints_all.points = o3d.utility.Vector3dVector(all_keys)
        keypoints_all.paint_uniform_color([0, 0.5, 0])
        high_keys = np.asarray([k.detach().numpy() for k in desc["positions_high"]])
        high_keys = desc["positions_high"]
        keypoints_high = o3d.geometry.PointCloud()
        keypoints_high.points = o3d.utility.Vector3dVector(high_keys)
        keypoints_high.paint_uniform_color([0, 0.5, 0])
        all_positions = [x.center.detach().numpy() for x in data["a"]]
        conf = [x.confidence.detach().numpy() for x in data["a"]]

        colors = all_descriptors.cpu().detach().numpy()
        colors_pca = PCA(n_components=3).fit_transform(colors)
        colors_pca = colors_pca - np.min(colors_pca, axis=0)
        colors_pca = colors_pca / np.max(colors_pca, axis=0)

        final_positions = np.array(all_positions)
        final_conf = np.squeeze(np.array(conf), 1)
        # final_positions = np.array(final_positions)
        final_colors = np.array(colors_pca)

        nbrs_knn = NearestNeighbors(n_neighbors=4, algorithm="ball_tree")
        nbrs_knn.fit(final_positions)
        dist, labels = nbrs_knn.kneighbors(pc_raw_np, return_distance=True)
        dist = np.max(dist) / (dist + np.max(dist))

        weighted_color = final_colors[labels, :] * np.expand_dims(dist, 2)
        weighted_color = np.sum(weighted_color, 1) / np.sum(dist, 1, keepdims=True)
        pc_raw.colors = o3d.utility.Vector3dVector(weighted_color)

        show_from_center([pc_raw])

        weighted_conf = final_conf[labels, :] * np.expand_dims(dist, 2)
        weighted_conf = np.sum(weighted_conf, 1) / np.sum(dist, 1, keepdims=True)
        color_conf = np.squeeze(np.repeat(np.expand_dims(weighted_conf, 1), 3, 1))
        color_conf[:, 2] = 0
        color_conf[:, 0] = 0
        pc_raw.colors = o3d.utility.Vector3dVector(color_conf)
        show_from_center([pc_raw])

        pc_raw.paint_uniform_color([0.7, 0.7, 0.7])
        show_from_center([pc_raw])
        show_from_center([keypoints_all])
        show_from_center([keypoints_high])

        # o3d.visualization.draw_geometries([pc_raw,keypoints_all])
        # o3d.visualization.draw_geometries([keypoints_all,pc_raw])
        # o3d.visualization.draw_geometries([pc_raw,keypoints_high])





def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


import numpy as np
from sklearn.neighbors import NearestNeighbors

# code from sergeyprokudin/chamfer_distance.py

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist