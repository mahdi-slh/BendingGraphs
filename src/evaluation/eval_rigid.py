import torch
import open3d as o3d
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader as UtilsDataLoader

from torch import nn
from collections import Counter
import copy
import random
from pyquaternion import Quaternion
from evaluation.eval_utils import chamfer_distance


from utils.log import log
from utils.maths import solve_svd
from utils.maths import rotate_np_points_with_t_r

from torch_scatter import scatter_mean
from evaluation.evaluator import Evaluator

from utils.visualize_graph import (
    show_from_center,
    show_from_side,
    show_from_top,
    show_from_new,
    show_with_no_light,
)




class EvaluatorRigid(Evaluator):
    def __init__(self , tb_writer=None, visualize=True, full_eval=False):
        surface='rigid'
        super().__init__(surface = surface, tb_writer=tb_writer, visualize=visualize, full_eval=full_eval)
    
    def evaluate_surface(self, dict_predictions, meta, epoch):

        eval_results = {}

        g1_seeds = meta["seeds_a"]
        g2_seeds = meta["seeds_p"]

        g1_desc = dict_predictions["a"]["descriptors"]
        g2_desc = dict_predictions["p"]["descriptors"]

        seed_a = g1_seeds.cpu().numpy()
        seed_p = g2_seeds.cpu().numpy()

        seed_a_o3d = o3d.geometry.PointCloud()
        seed_a_o3d.points = o3d.utility.Vector3dVector(seed_a)

        seed_p_o3d = o3d.geometry.PointCloud()
        seed_p_o3d.points = o3d.utility.Vector3dVector(seed_p)




        # conf_a = dict_results['a']['confidence'].cpu().numpy()
        # conf_p = dict_results['p']['confidence'].cpu().numpy()

        if "conf0" in dict_predictions["matching_results"].keys():
            conf_shape_a = dict_predictions["matching_results"]["conf0"].cpu().numpy()
            conf_shape_p = dict_predictions["matching_results"]["conf1"].cpu().numpy()

        scene_a = meta["scene_a"]  # meta.get('scene_a',None)
        scene_p = meta["scene_p"]  # meta.get('scene_p',None)

        # conf_p = conf_p-np.min(conf_p)
        # conf_p = conf_p/(np.max(conf_p)+0.1)

        # conf_a = conf_a-np.min(conf_a)
        # conf_a = conf_a/(np.max(conf_a)+0.1)

        # prop_a = dict_results['max_values'][0]
        # prop_p = dict_results['max_values'][1]

        # prop_p = prop_p-np.min(prop_p)
        # prop_p = conf_p/(np.max(prop_p)+0.1)

        # prop_a = prop_a-np.min(prop_a)
        # prop_a = prop_a/(np.max(prop_a)+0.1)

        size_a, size_p = g1_desc.shape[0], g2_desc.shape[0]

        ind_a = np.zeros([size_a, 2], dtype=int)
        ind_p = np.zeros([size_p, 2], dtype=int)

        ind_a[:, 0] = np.arange(size_a)
        ind_a[:, 1] = dict_predictions["matching_results"]["matches0"].detach().cpu().numpy()
        ind_a_valid = ind_a[ind_a[:, 1] > -1, :]

        ind_p[:, 0] = np.arange(size_p)
        ind_p[:, 1] = dict_predictions["matching_results"]["matches1"].detach().cpu().numpy()
        ind_p_valid = ind_p[ind_p[:, 1] > -1, :]

        ind_a_ot = np.copy(ind_a)

        if "matches0_ot" in dict_predictions["matching_results"].keys():
            ind_a_ot[:, 1] = (
                dict_predictions["matching_results"]["matches0_ot"].detach().cpu().numpy()
            )
        avg_dist_geo_ot = (meta["dist_mat"][ind_a_ot[:, 0], ind_a_ot[:, 1]]).mean().item()

        

        number_valid = len(ind_a_valid)
        avg_dist_geo = (meta["dist_mat"][ind_a[:, 0], ind_a[:, 1]]).mean().item()

        bijective = ind_p[ind_a[:, 1], 1] == ind_a[:, 0]

        ind_a_bijective = ind_a[bijective, :]
        rate = sum(bijective) / len(bijective)

        p_dif = meta["dist_mat"][ind_a[:, 0], ind_a[:, 1]].detach().cpu().numpy()
        p_dif = p_dif / np.max(p_dif)
        p_dif = np.tile(p_dif.reshape(-1, 1), 3)
        p_dif[:, 1] = 1 - p_dif[:, 0]
        p_dif[:, 2] = 0

        if "score_4ot" in dict_predictions["matching_results"].keys():
            self.add_image_tb(dict_predictions["matching_results"]["score_4ot"], "before ot", epoch
            )
        if "score_mat" in dict_predictions["matching_results"].keys():
            self.add_image_tb(
                np.abs(dict_predictions["matching_results"]["score_mat"]),
                "after ot",
                epoch,
            )
        if "dist_mat" in meta.keys():
            self.add_image_tb(meta["dist_mat"].squeeze(), "Dist mat", epoch)

        eval_results = {
            "bij_rate": rate,
            "number_valid": number_valid,
            "geod_avg": avg_dist_geo,
            "geod_avg_ot": avg_dist_geo_ot,
        }
        est_rot, est_t, diff = get_pose_from_sets_ransac(seed_p, seed_a[ind_p[:,1]],ransac=False)
        trans_pr = np.eye(4)
        if est_t is not None:
            trans_pr[0:3, 0:3] = est_rot
            trans_pr[0:3, 3] = est_t.squeeze()
        res = o3d.pipelines.registration.registration_icp(
            seed_p_o3d,
            seed_a_o3d,
            max_correspondence_distance=0.1,
            init=trans_pr,
            # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        est_rot_icp = res.transformation[0:3, 0:3]
        est_t_icp = np.expand_dims(res.transformation[0:3, 3], 1)
        rot_euler_pred_icp = npmat2euler(np.expand_dims(est_rot_icp,0))

        rot_euler_pred = npmat2euler(np.expand_dims(est_rot,0))
        rot_euler_gt = npmat2euler(meta["r_p"].transpose(0,1).unsqueeze(0).cpu().numpy())
        t_gt = -meta["t_p"].cpu().numpy()

        err_r_mse = np.mean(
            (rot_euler_pred - rot_euler_gt) ** 2
        )
        err_r_rmse = np.sqrt(err_r_mse)
        err_r_mae = np.mean(
            np.abs(rot_euler_pred - rot_euler_gt )
        )
        err_t_mse = np.mean(np.power(t_gt - est_t.T,2))
        err_t_rmse = np.sqrt(err_t_mse)
        err_t_mae = np.mean(np.abs(t_gt - est_t.T))
        eval_results.update({
            "err_r_mse": err_r_mse,
            "err_r_rmse": err_r_rmse,
            "err_r_mae": err_r_mae,
            "err_t_mse": err_t_mse,
            "err_t_rmse": err_t_rmse,
            "err_t_mae": err_t_mae,

        })

        if self.visualize:
            
            full_a_o3d = o3d.geometry.PointCloud()
            full_a_o3d.points = o3d.utility.Vector3dVector(meta['full_points_a'])
            full_a_o3d.paint_uniform_color([1, 0, 0])

            full_p_o3d = o3d.geometry.PointCloud()
            full_p_o3d.points = o3d.utility.Vector3dVector(meta['full_points_p'])
            full_p_o3d.paint_uniform_color([0, 0, 1])

            cols_p = np.zeros([seed_p.shape[0], 3], dtype="float32")
            cols_p = seed_p - np.min(seed_p, 0)
            cols_p = cols_p / np.max(cols_p, 0)
            seed_p_o3d.colors = o3d.utility.Vector3dVector(cols_p)


            cols_a = np.zeros([seed_a.shape[0], 3], dtype="float32")
            cols_a = cols_p[ind_a[:,1],:]
            seed_a_o3d.colors = o3d.utility.Vector3dVector(cols_a)
            # o3d.visualization.draw_geometries([seed_p_o3d.translate([1, 0, 0]), seed_a_o3d])

            show_with_no_light([seed_a_o3d, seed_p_o3d])

            cols_conf_a = cols_a
            cols_conf_a = np.tile(conf_shape_a.reshape(-1, 1), 3)
            cols_conf_a = cols_conf_a / (np.max(cols_conf_a) + 0.001)  # Normalization
            cols_conf_a[:, 0] = 1 - cols_conf_a[:, 1]
            cols_conf_a[:, 2] = 0
            seed_a_o3d.colors = o3d.utility.Vector3dVector(cols_conf_a)

            cols_conf_p = cols_p
            cols_conf_p = np.tile(conf_shape_p.reshape(-1, 1), 3)
            cols_conf_p = cols_conf_p / (np.max(cols_conf_p) + 0.001)  # Normalization
            cols_conf_p[:, 0] = 1 - cols_conf_p[:, 1]
            cols_conf_p[:, 2] = 0
            seed_p_o3d.colors = o3d.utility.Vector3dVector(cols_conf_p)

            show_with_no_light([seed_a_o3d, seed_p_o3d])

            seed_p_copy = copy.deepcopy(seed_p_o3d)
            seed_p_copy.transform(trans_pr)
            full_p_o3d.transform(trans_pr)
            seed_p_o3d.paint_uniform_color([0, 0, 1])

            seed_p_copy.paint_uniform_color([0, 0, 1])
            seed_a_o3d.paint_uniform_color([1, 0, 0])
            show_with_no_light([full_a_o3d, full_p_o3d])


            tuples = [tuple(l) for l in ind_p]
            lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
                seed_p_o3d, seed_a_o3d, tuples
            )
            show_with_no_light([lines, seed_p_o3d, seed_a_o3d])

        #for chamfer calculation outside visualization
        seed_p_o3d.transform(trans_pr)

        cd = chamfer_distance(np.asarray(seed_p_o3d.points),np.asarray(seed_a_o3d.points))
        eval_results.update({'chamfer':cd})

        return eval_results


        



def test_one_object(settings, net, test_data):

    net.eval()

    mse_icp = 0
    mae_icp = 0
    mse_grf = 0
    mae_grf = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0

    rotations_ab = []
    translations_ab = []
    rotations_grf_pred = []
    translations_grf_pred = []
    rotations_icp_pred = []
    translations_icp_pred = []

    eulers_ab = []
    failing_cases = 0
    for data in tqdm(test_data):

        rotation_ab = torch.tensor(data["r"]).unsqueeze(0)
        translation_ab = torch.tensor(data["t"]).unsqueeze(0)
        euler_ab = torch.tensor(data["e"]).unsqueeze(0)

        (
            rotation_icp_pred,
            translation_icp_pred,
            rotation_grf_pred,
            translation_grf_pred,
        ) = get_r_t_from_net(settings, net, data)
        if translation_grf_pred is None:
            continue

        rotation_grf_pred = torch.tensor(rotation_grf_pred).unsqueeze(0)
        translation_grf_pred = (
            torch.tensor(translation_grf_pred).squeeze(1).unsqueeze(0)
        )
        rotation_icp_pred = torch.tensor(rotation_icp_pred).unsqueeze(0)
        translation_icp_pred = (
            torch.tensor(translation_icp_pred).squeeze(1).unsqueeze(0)
        )

        if torch.max(translation_grf_pred) > 10:
            # print("wrong.")
            failing_cases += 1
            continue

        num_examples += 1

        rotations_ab.append(rotation_ab)
        translations_ab.append(translation_ab)
        rotations_grf_pred.append(rotation_grf_pred)
        translations_grf_pred.append(translation_grf_pred)
        rotations_icp_pred.append(rotation_icp_pred)
        translations_icp_pred.append(translation_icp_pred)
        eulers_ab.append(euler_ab)

        transformed_grf = transform_point_cloud(
            torch.tensor(data["raw"].points), rotation_grf_pred, translation_grf_pred
        )
        transformed_icp = transform_point_cloud(
            torch.tensor(data["raw"].points), rotation_icp_pred, translation_icp_pred
        )
        target = transform_point_cloud(
            torch.tensor(data["raw"].points), rotation_ab, translation_ab
        )

        # identity = torch.eye(3).unsqueeze(0).repeat(1, 1, 1)
        # loss = F.mse_loss(torch.matmul(rotation_grf_pred, rotation_ab), identity) \
        #        + F.mse_loss(translation_grf_pred, translation_ab)

        # total_loss += loss.item()

        mse_grf += torch.mean((transformed_grf - target) ** 2).item()
        mae_grf += torch.mean(torch.abs(transformed_grf - target)).item()
        mse_icp += torch.mean((transformed_icp - target) ** 2).item()
        mae_icp += torch.mean(torch.abs(transformed_icp - target)).item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_grf_pred = np.concatenate(rotations_grf_pred, axis=0)
    translations_grf_pred = np.concatenate(translations_grf_pred, axis=0)
    rotations_icp_pred = np.concatenate(rotations_icp_pred, axis=0)
    translations_icp_pred = np.concatenate(translations_icp_pred, axis=0)
    eulers_ab = np.concatenate(eulers_ab, axis=0)

    print("{} cases failed".format(failing_cases))
    return (
        mse_grf * 1.0 / num_examples,
        mae_grf * 1.0 / num_examples,
        rotations_ab,
        translations_ab,
        rotations_grf_pred,
        translations_grf_pred,
        rotations_icp_pred,
        translations_icp_pred,
        eulers_ab,
    )


def npmat2euler(mats, seq="zyx"):
    eulers = []
    from_matrix = getattr(Rotation, "from_matrix", None) or Rotation.from_dcm
    for i in range(mats.shape[0]):
        r = from_matrix(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype="float32")


def quat2mat(quat):

    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def get_r_t_from_net(settings, model, data):

    visualize = False
    if len(data["a"]) > 150 or len(data["b"]) > 150:
        print(len(data["a"]), len(data["b"]))
        return None

    with torch.no_grad():

        data_s_1 = [
            model(
                Batch.from_data_list(data["a"][i : i + settings.mb_size]).to(
                    torch.cuda.current_device()
                )
            )
            for i in range(0, len(data["a"]), settings.mb_size)
        ]
        data_s_2 = [
            model(
                Batch.from_data_list(data["b"][i : i + settings.mb_size]).to(
                    torch.cuda.current_device()
                )
            )
            for i in range(0, len(data["b"]), settings.mb_size)
        ]

    point_cloud1_np = np.asarray(data["raw"].points)
    cent1 = np.mean(point_cloud1_np, 0)

    g1_result = merge_dict(data_s_1)
    g2_result = merge_dict(data_s_2)

    confidence_thresh = 0.0
    # g1_result['descriptors']=g1_result['descriptors'][g1_result['confidence'][:,0]>0]
    # g2_result['descriptors']=g2_result['descriptors'][g2_result['confidence'][:,0]>0]
    # g1_positions = [data['a'][i].positions[np.argmax(g1_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g1_result['probabilities'])) if data['a'][i].confidence>0]
    # g2_positions = [data['b'][i].positions[np.argmax(g2_result['probabilities'][i].cpu().detach().numpy())] for i in range(len(g2_result['probabilities'])) if data['b'][i].confidence>0]

    g1_positions = [
        data["a"][i].positions[
            np.argmax(g1_result["probabilities"][i].cpu().detach().numpy())
        ]
        for i in range(len(g1_result["probabilities"]))
    ]
    g2_positions = [
        data["b"][i].positions[
            np.argmax(g2_result["probabilities"][i].cpu().detach().numpy())
        ]
        for i in range(len(g2_result["probabilities"]))
    ]

    # g1_positions = [data['a'][i].center for i in range(len(g1_result['probabilities']))]
    # g2_positions = [data['b'][i].center for i in range(len(g2_result['probabilities']))]

    final1 = {"positions": g1_positions, "descriptors": g1_result["descriptors"]}
    final2 = {"positions": g2_positions, "descriptors": g2_result["descriptors"]}

    descriptorMatcher = DescriptorMatcher()
    setA, setB, ind = descriptorMatcher.match_old(final1, final2, return_id=True)
    keysA, keysB, descA, descB, ind = descriptorMatcher.match(
        final1, final2, return_id=True
    )
    # ind = descriptorMatcher.match_id(final1, final2)

    # pc1=torch.stack(final1['positions']).cpu().detach().numpy()
    # pc1_o3d=o3d.geometry.PointCloud()
    # pc1_o3d.points=o3d.utility.Vector3dVector(pc1)
    # pc1_o3d.paint_uniform_color([1,0,0])
    # pc2=torch.stack(final2['positions']).cpu().detach().numpy()
    # pc2_o3d=o3d.geometry.PointCloud()
    # pc2_o3d.points=o3d.utility.Vector3dVector(pc2)
    # pc2_o3d.paint_uniform_color([0,1,0])

    pc1 = [data["a"][i].positions for i in range(len(data["a"]))]
    pc1 = torch.stack(pc1).view([-1, 3]).cpu().detach().numpy()
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc1_o3d.paint_uniform_color([1, 0, 0])

    pc2 = [data["b"][i].positions for i in range(len(data["b"]))]
    pc2 = torch.stack(pc2).view([-1, 3]).cpu().detach().numpy()
    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    pc2_o3d.paint_uniform_color([0, 0, 1])

    pc1_o3d.estimate_normals()
    pc2_o3d.estimate_normals()

    # f1_feat = o3d.pipelines.registration.Feature()
    # f1_feat.data = descA.T

    # f2_feat = o3d.pipelines.registration.Feature()
    # f2_feat.data = descB.T

    # distance_threshold=0.02
    # inliers=4#int(len(tuples)*0.1)
    # res=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(keypoints_o3d_a, keypoints_o3d_b, f1_feat,f2_feat,
    #  max_correspondence_distance=distance_threshold,estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #  ransac_n=inliers,
    #         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_validation=500,max_iteration=100000))

    estimatedRotation, estimatedTranslation, diff = get_pose_from_sets_ransac(
        setA, setB
    )

    if visualize:
        keypoints_o3d_a = o3d.geometry.PointCloud()
        keypoints_o3d_a.points = o3d.utility.Vector3dVector(keysA)
        keypoints_o3d_a.paint_uniform_color([0, 0, 0])
        keypoints_o3d_b = o3d.geometry.PointCloud()
        keypoints_o3d_b.points = o3d.utility.Vector3dVector(keysB)
        keypoints_o3d_b.paint_uniform_color([0, 0, 0])
        list_matches = [[i, i] for i in range(keysA.shape[0])]
        tuples = [tuple(l) for l in list_matches]
        lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            keypoints_o3d_a, keypoints_o3d_b, tuples
        )

        pc1_copy = copy.deepcopy(pc1_o3d)

    # res=o3d.pipelines.registration.registration_ransac_based_on_correspondence(pc1_o3d, pc2_o3d, corres, max_correspondence_distance=0.9,estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),ransac_n=20)
    # trans_gt=np.eye(4)
    # trans_gt[0:3,0:3]=data['r']
    # trans_gt[0:3,3]=data['t']

    trans_pr = np.eye(4)
    if estimatedTranslation is not None:
        trans_pr[0:3, 0:3] = estimatedRotation
        trans_pr[0:3, 3] = estimatedTranslation.squeeze()
    # trans_pr=res.transformation

    # estimatedRotation=res.transformation[0:3,0:3]
    # estimatedTranslation=np.expand_dims(res.transformation[0:3,3],1)
    res = o3d.pipelines.registration.registration_icp(
        pc1_o3d,
        pc2_o3d,
        max_correspondence_distance=0.1,
        init=trans_pr,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    if visualize:
        show_with_no_light([pc1_o3d, pc2_o3d])
        pc1_copy = copy.deepcopy(pc1_o3d)
        pc1_copy.transform(trans_pr)
        pc1_copy.paint_uniform_color([1, 0, 0])
        show_with_no_light([pc2_o3d, pc1_copy])
        #
        pc1_copy = copy.deepcopy(pc1_o3d)
        pc1_copy.transform(res.transformation)
        pc1_copy.paint_uniform_color([1, 0, 0])
        show_with_no_light([pc2_o3d, pc1_copy])

        pc1_o3d.paint_uniform_color([0.99, 0.7, 0.7])
        pc2_o3d.paint_uniform_color([0.7, 0.7, 0.99])
        keypoints_o3d_a.paint_uniform_color([0.0, 0.5, 0.0])
        keypoints_o3d_b.paint_uniform_color([0.0, 0.5, 0.0])
        show_with_no_light([lines, pc1_o3d, pc2_o3d, keypoints_o3d_a, keypoints_o3d_b])
        # o3d.visualization.draw_geometries([lines,pc1_o3d,pc2_o3d,keypoints_o3d_a,keypoints_o3d_b])

    estimatedRotationICP = res.transformation[0:3, 0:3]
    estimatedTranslationICP = np.expand_dims(res.transformation[0:3, 3], 1)
    return (
        estimatedRotationICP,
        estimatedTranslationICP,
        estimatedRotation,
        estimatedTranslation,
    )


def transform_point_cloud(point_cloud, rotation, translation):

    r = rotation.squeeze(0)
    t = translation.squeeze(0)

    return torch.matmul(point_cloud, r) + t


def get_pose_from_sets_ransac(a, b,ransac=True):

    n = a.shape[0]
    if a.shape != b.shape:
        print("not a pairing set.")
        return

    number_of_points = n
    # print(n)
    number_of_samples = int(number_of_points * 0.1)

    if n < 6:
        print("not enough points.")
        return None, None, None

    number_of_iterations = 1000

    min_dif = 100000
    best_tr_t = None
    best_tr_r = None
    if not ransac:
        number_of_iterations = 1
        number_of_samples = number_of_points 

    for i in range(number_of_iterations):

        indexes = random.sample(range(0, number_of_points), number_of_samples)

        a_s = a[indexes, :]
        b_s = b[indexes, :]

        cent_a = np.mean(a[range(0, n), :], 0)
        cent_b = np.mean(b[range(0, n), :], 0)
        t, r = solve_svd(a_s, b_s, cent_a, cent_b)

        a_to_b = rotate_np_points_with_t_r(a, t, r)
        dist_temp = np.linalg.norm(a_to_b - b, axis=1)
        ind = np.where(dist_temp < 30)[0]
        if (len(ind) < 6): #or abs(Quaternion(matrix=r).angle) > 1.7:
            continue
        dif = np.mean(np.linalg.norm(a_to_b - b, axis=1)[ind])

        if dif < min_dif or number_of_iterations==1:
            best_tr_t = t
            best_tr_r = r
            # print(Quaternion(matrix=r).angle)
            min_dif = dif

    # print(min_dif)

    return best_tr_r, best_tr_t, min_dif

