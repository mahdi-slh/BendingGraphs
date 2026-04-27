import open3d as o3d
import numpy as np
import cv2
from scipy.spatial import distance
from torch_geometric.data import Data
import random
import itertools
from PIL import Image, ImageDraw
from configs import device
from loaders.bunny import create_random_patches

from graph import create_local_graph
from utils.utils import *
from loaders.augmentations import *
from utils.pointcloud_utils import permute_pointcloud


def draw_image_cropped(image, u, v):

    u = (u // WINDOW_SIZE) * WINDOW_SIZE
    v = (v // WINDOW_SIZE) * WINDOW_SIZE

    image = np.asarray(image)
    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    draw.rectangle(((u + WINDOW_SIZE, v + WINDOW_SIZE), (u, v)))

    image.show()


def _get_points_from_rgb_and_depth_image(color_image, depth_image, dataset, d_scale=1):
    # cam_int = init_camera(dataset)
    # c_x=cam_int.intrinsic_matrix[0,2]
    # c_y=cam_int.intrinsic_matrix[1,2]
    # f=90#cam_int.intrinsic_matrix[1,1]
    # depth_np=100*(np.asarray(depth_image)/252**2)
    # color_np=np.asarray(color_image)
    # w,h=depth_np.shape
    # x_space=np.linspace(0, w-1, w)
    # y_space=np.linspace(0, h-1, h)
    # x_mesh,y_mesh=np.meshgrid(x_space,y_space)

    # x_mesh=(x_mesh-c_x)/f
    # y_mesh=(y_mesh-c_y)/f
    # z_mesh=depth_np#/(1+y_mesh**2+x_mesh**2)**0.5
    # x_mesh=x_mesh*z_mesh
    # y_mesh=y_mesh*z_mesh
    # points=np.stack([np.ravel(x_mesh),np.ravel(y_mesh),np.ravel(z_mesh)])
    # points_o3d=o3d.geometry.PointCloud()
    # points_o3d.points=o3d.utility.Vector3dVector(points.T)
    # coor=o3d.geometry.TriangleMesh.create_coordinate_frame(10)
    # o3d.visualization.draw_geometries([points_o3d,coor])

    cam_int = init_camera(dataset)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1,
        depth_trunc=65535,
        convert_rgb_to_intensity=False,
    )
    points = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d.camera.PinholeCameraIntrinsic(cam_int)
    )
    points_np = np.asarray(points.points) * d_scale
    points_np[:, 2] = points_np[:, 2]
    points.points = o3d.utility.Vector3dVector(points_np)
    arg = np.argmin(points_np[:, 2])

    # colors_np=np.asarray(points.col r3dVector(colors_np )
    return points, points_np[arg, :]


def _get_window_inside_o3d_image(image, top_left, window_size=WINDOW_SIZE):

    np_image = np.asarray(image)
    np_image_filtered = np.zeros_like(np_image)
    np_image_filtered[
        top_left[1] : top_left[1] + window_size, top_left[0] : top_left[0] + window_size
    ] = np_image[
        top_left[1] : top_left[1] + window_size, top_left[0] : top_left[0] + window_size
    ]

    return o3d.geometry.Image(np_image_filtered)


def generate_points_and_feature_coord(
    feature_point_2d, color_image, depth_segmented, dataset, d_scale=1
):
    points, keypoint_pos = _get_points_from_rgb_and_depth_image(
        color_image, depth_segmented, dataset, d_scale
    )
    # coor=o3d.geometry.TriangleMesh.create_coordinate_frame(10)
    # o3d.visualization.draw_geometries([points,coor])

    points.estimate_normals()
    points.orient_normals_to_align_with_direction()

    max_bound = np.mean(points.get_max_bound()[0:-1] - points.get_min_bound()[0:-1])
    points.scale(1 / max_bound, [0, 0, 0])
    keypoint_pos = keypoint_pos / max_bound
    vox_size = 1 / 30

    points = points.voxel_down_sample(vox_size)

    points_np = np.asarray(points.points)
    colors_np = np.asarray(points.colors)
    normals_np = np.asarray(points.normals)

    if points_np.shape[0] < GRAPH_SIZE:
        return None, -1
    # depth_image_single_point_windowed = _get_window_inside_o3d_image(depth_segmented, feature_point_2d, window_size=1)

    # keypoint_to_detect = _get_points_from_rgb_and_depth_image(color_image, depth_image_single_point_windowed, dataset, d_scale)
    # keypoint_to_detect_np=np.asarray(keypoint_to_detect.points)

    nbrs_ = NearestNeighbors(n_neighbors=GRAPH_SIZE, algorithm="ball_tree")
    nbrs_.fit(points_np)
    # print(points_np.shape)
    dist, labels = nbrs_.kneighbors(
        np.expand_dims(keypoint_pos, 0), return_distance=True
    )
    # feature_index=labels[:,np.argmin(dist)].item()

    # points_selected=np.squeeze(points_np[labels,:])
    # points_selected_colors=np.squeeze(colors_np[labels,:])
    # points_selected_normals=np.squeeze(normals_np[labels,:])
    # points_selected_colors[0,:]=[1,0,0]

    points_selected_o3d = points.select_by_index(labels[0, :])
    # points_selected_o3d.points=o3d.utility.Vector3dVector(points_selected)
    # points_selected_o3d.normals=o3d.utility.Vector3dVector(points_selected_normals)
    # points_selected_o3d.colors=o3d.utility.Vector3dVector(points_selected_colors)

    # point_feat=points_np[feature_index,:]
    return points_selected_o3d, np.argmin(dist)


def generate_points_and_feature_index_from_grid_top_left_point(
    feature_point_2d, color_image, depth_segmented, dataset, d_scale=1
):

    window_top_left_i = (feature_point_2d[0] // WINDOW_SIZE) * WINDOW_SIZE
    window_top_left_j = (feature_point_2d[1] // WINDOW_SIZE) * WINDOW_SIZE

    depth_image_windowed = _get_window_inside_o3d_image(
        depth_segmented, [window_top_left_i, window_top_left_j], window_size=WINDOW_SIZE
    )
    depth_image_single_point_windowed = _get_window_inside_o3d_image(
        depth_segmented, feature_point_2d, window_size=1
    )

    points = _get_points_from_rgb_and_depth_image(
        color_image, depth_image_windowed, dataset, d_scale
    )
    if len(points.points) < GRAPH_SIZE:
        return None, -1

    points.estimate_normals()
    points.orient_normals_to_align_with_direction()

    feature_point_to_detect = _get_points_from_rgb_and_depth_image(
        color_image, depth_image_single_point_windowed, dataset, d_scale
    )

    if len(feature_point_to_detect.points) != 1:
        feature_point_index = -1
    else:
        feature_point_index = np.argmin(
            distance.cdist(points.points, feature_point_to_detect.points, "euclidean")
        )

    return points, feature_point_index


def get_graph_for_synprim(color_path, depth_path, augment=True):

    color_image = cv2.imread(color_path, -1)

    # color_image = cv2.imread('../../../dataset/syn_prim/000010.jpg')
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_image = cv2.imread(depth_path, -1)
    # depth_image = cv2.imread('../../../dataset/syn_prim/000010_d.png',-1)
    # depth_image=depth_image[:,:,1]

    # d_sub = sub_sample(d, 1.)
    # c_sub = sub_sample(c, 1.)

    random_shift = [
        0,
        0,
    ]  # random.sample(range(-WINDOW_SIZE//3 + 1, WINDOW_SIZE//3), 2)
    # d_sub_crop = crop_o3d_image_square(depth_image, WINDOW_SIZE//2, random_shift)
    # c_sub_crop = crop_o3d_image_square(color_image, WINDOW_SIZE//2, random_shift)

    d_scale = 100 / 255**2
    fixed_color, fixed_depth = convert_to_o3d_rgbd(color_image, depth_image)
    feature_loc = [
        50,
        50,
    ]  # [WINDOW_SIZE//2 - random_shift[0], WINDOW_SIZE//2 - random_shift[1]]
    g_points_r, g_keypoint_index_r = generate_points_and_feature_coord(
        feature_loc, fixed_color, fixed_depth, dataset="synprim", d_scale=d_scale
    )
    # g_points_t, g_fearture_index_t = generate_points_and_feature_index_from_grid_top_left_point([WINDOW_SIZE//2, WINDOW_SIZE//2], fixed_color_t, fixed_depth_t, dataset='synprim2', d_scale=d_scale)
    if not good_point_cloud(g_points_r):
        return None

    g_points, g_keypoint_index = permute_pointcloud(g_points_r, g_keypoint_index_r)
    g_points.colors[g_keypoint_index] = [1, 0, 0]

    g_points, g_keypoint_index = permute_pointcloud(g_points_r, g_keypoint_index_r)

    if g_keypoint_index == -1:
        return None

    if augment:

        g_points = random_perturb(g_points, coef=0.08)
        g_points, _ = random_transform(g_points, rotate=False)
        g_points = color_jitter(g_points, 0.2)
        g_points = color_noise(g_points, 0.02)

    # o3d.visualization.draw_geometries([g_points])
    g = create_local_graph(g_points, g_keypoint_index)

    return g


def get_graph_for_3dmatch(ply_path, augment=False):

    ply = o3d.io.read_point_cloud(ply_path)

    max_bound = np.median(ply.get_max_bound() - ply.get_min_bound())
    vox_size = 0.01
    ply = ply.voxel_down_sample(vox_size)
    ply.estimate_normals()
    ply.orient_normals_to_align_with_direction()

    g_fearture_index = 0

    # if augment:
    #     g_points=random_perturb(g_points,coef=0.1)
    #     g_points=random_transform(g_points,rotate=False)
    #     g_points=color_jitter(g_points,0.2)
    #     g_points=color_noise(g_points,0.05)

    patches1, c1 = create_random_patches(
        point_cloud1_sub, count=40, augment=False, uniform=False
    )
    patches2, c2 = create_random_patches(
        point_cloud2_sub, count=40, augment=False, uniform=False
    )
    patches1_2, c1 = create_random_patches(
        point_cloud1_sub, count=20, augment=False, uniform=False
    )
    patches2_2, c2 = create_random_patches(
        point_cloud2_sub, count=20, augment=False, uniform=False
    )
    patches1_3, c1 = create_random_patches(
        point_cloud1_sub, count=10, augment=False, uniform=False
    )
    patches2_3, c2 = create_random_patches(
        point_cloud2_sub, count=10, augment=False, uniform=False
    )
    patches1 = patches1 + patches1_2 + patches1_3
    patches2 = patches2 + patches2_2 + patches2_3

    graphed_patches1 = [
        create_local_graph(x, -1, True, self.train)
        for x in patches1
        if len(x.points) == GRAPH_SIZE
    ]
    graphed_patches2 = [
        create_local_graph(x, -1, True, self.train)
        for x in patches2
        if len(x.points) == GRAPH_SIZE
    ]
    # for i in range(len(graphed_patches1)):
    #     visualize_torch_graphs(graphed_patches1[i],graphed_patches2[i])
    for i_hash in range(len(graphed_patches1)):
        hash_code = random.getrandbits(16)
        graphed_patches1[i_hash]["hash"] = hash_code
        graphed_patches2[i_hash]["hash"] = hash_code

    result = {
        "a": graphed_patches1,
        "b": graphed_patches2,
        "t": random_t,
        "r": random_r,
        "e": random_euler,
        "name": off + ".pcd",
    }

    return g


def convert_to_o3d_rgbd(c, d):

    c_np = np.asarray(c)
    d_np = 256 * 256 - d

    depth_image = o3d.geometry.Image(d_np.astype(np.uint16))
    color_image = o3d.geometry.Image(c_np.astype(np.uint8))

    return color_image, depth_image


def sub_sample(image, sample_rate):

    tmp_image_mat = np.asarray(image)

    width = int(tmp_image_mat.shape[1] * sample_rate)
    height = int(tmp_image_mat.shape[0] * sample_rate)
    dim = (width, height)

    c = cv2.resize(image, dim, cv2.INTER_LINEAR)

    return c


def crop_o3d_image_square(image, margin, random_shift):

    w, h = image.shape[0] // 2 + random_shift[1], image.shape[1] // 2 + random_shift[0]
    image = image[w - margin : w + margin + 1, h - margin : h + margin + 1]
    return image


def good_point_cloud(pc):
    return not (pc is None or len(pc.points) != GRAPH_SIZE)


def get_graph_pairs(object_id, object_name, object_pres, val_poses, frame_idx, dataset):
    """Returns list of graph pairs."""

    camera_init = init_camera(dataset)

    (
        color_image,
        _,
        depth_segmented,
        seg_array,
        gray_image,
    ) = read_related_images_of_specific_class_and_index(
        object_name, object_pres, frame_idx
    )
    tra_t, tra_t_inv, tra_r, tra_r_inv = get_transformation_matrices(
        val_poses[frame_idx, :].squeeze()
    )

    nx, ny = color_image.get_max_bound()
    nx = list(range(5, int(nx), WINDOW_SIZE))
    ny = list(range(5, int(ny), WINDOW_SIZE))
    grids_centers = torch.tensor(list(itertools.product(nx, ny))).to(device)

    graphs = []
    total_tries = 0

    while total_tries < 50:

        center = random.sample(list(grids_centers), 1)[0]
        g1_points, idx1 = generate_points_and_feature_index_from_grid_top_left_point(
            center, color_image, depth_segmented, dataset
        )

        if not good_point_cloud(g1_points):
            total_tries += 1
            continue

        g1_points.transform(tra_t)
        g1_points.transform(tra_r)

        index_point_location = np.expand_dims(g1_points.points[idx1], axis=0)

        for tries in range(10):

            g2_frame_idx = random.randint(0, len(object_pres) - 1)

            (
                color_image2,
                _,
                depth_segmented2,
                _,
                _,
            ) = read_related_images_of_specific_class_and_index(
                object_name, object_pres, g2_frame_idx
            )
            tra_t2, tra_t_inv2, tra_r2, tra_r_inv2 = get_transformation_matrices(
                val_poses[g2_frame_idx, :].squeeze()
            )

            all_frame2_points = _get_points_from_rgb_and_depth_image(
                color_image2, depth_segmented2, dataset
            )

            if len(all_frame2_points.points) == 0:
                total_tries += 1
                continue

            all_frame2_points.transform(tra_t2)
            all_frame2_points.transform(tra_r2)

            distance_to_index_point = distance.cdist(
                np.array(all_frame2_points.points), index_point_location, "euclidean"
            )
            index_point_index_in_second_image_pointcloud = np.argmin(
                distance_to_index_point
            )

            if (
                distance_to_index_point[index_point_index_in_second_image_pointcloud]
                > 15
            ):
                total_tries += 1
                continue

            g1_points.transform(tra_r_inv)
            g1_points.transform(tra_t_inv)

            all_frame2_points.transform(tra_r_inv2)
            all_frame2_points.transform(tra_t_inv2)

            feature_point_in_second_image_pointcloud = all_frame2_points.points[
                index_point_index_in_second_image_pointcloud
            ]
            feature_point_in_second_image_pointcloud = (
                feature_point_in_second_image_pointcloud
                / feature_point_in_second_image_pointcloud[2]
            )

            u = (int)(
                feature_point_in_second_image_pointcloud[0]
                * FAT_CAMERA_INTRINSICS["fx"]
                + FAT_CAMERA_INTRINSICS["cx"]
            )
            v = (int)(
                feature_point_in_second_image_pointcloud[1]
                * FAT_CAMERA_INTRINSICS["fy"]
                + FAT_CAMERA_INTRINSICS["cy"]
            )

            (
                g2_points,
                idx2,
            ) = generate_points_and_feature_index_from_grid_top_left_point(
                [u, v], color_image2, depth_segmented2, dataset
            )

            if not good_point_cloud(g2_points):
                total_tries += 1
                continue

            g1 = create_local_graph(g1_points, None)
            g2 = create_local_graph(g2_points, None)

            # draw_image_cropped(color_image2, u, v)
            # draw_image_cropped(color_image, center[0], center[1])

            hash_code = random.getrandbits(16)
            g1.hash = hash_code
            g2.hash = hash_code
            graphs.extend([g1, g2])
            break

        total_tries += 1

    return graphs


def get_graphs_for_matching(color_image, depth_image, dataset):

    depth_image_np = np.asarray(depth_image)

    nx, ny = color_image.get_max_bound()
    nx = list(range(0, int(nx), WINDOW_SIZE))
    ny = list(range(0, int(ny), WINDOW_SIZE))
    grids_top_lefts = torch.tensor(list(itertools.product(nx, ny))).to(
        torch.cuda.current_device()
    )

    points_and_indexes = [
        generate_points_and_feature_index_from_grid_top_left_point(
            top_left_point, color_image, depth_image, dataset
        )
        for top_left_point in grids_top_lefts
        if depth_image_np[top_left_point[1], top_left_point[0]] != 0
    ]
    points_and_indexes = list(
        filter(lambda d: good_point_cloud(d[0]), points_and_indexes)
    )
    graphs = [create_local_graph(d[0], d[1]) for d in points_and_indexes]

    return graphs
