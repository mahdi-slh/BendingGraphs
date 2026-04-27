from copy import deepcopy
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
import os
import math
import torch



def visualize_graph_shape(gr_pc):

    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.4)

    pos_np = gr_pc.x.cpu().detach().numpy()
    edge_ind = gr_pc.edge_index.cpu().detach().numpy()
    edge_val = gr_pc.edge_attr.cpu().detach().numpy()

    pos_o3d = o3d.geometry.PointCloud()
    pos_o3d.points = o3d.utility.Vector3dVector(pos_np)

    pos_o3d.paint_uniform_color([0, 0, 0])

    lines_edges = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pos_o3d, pos_o3d, list(map(tuple, edge_ind.T))
    )

    

    # show_from_side([pos_o3d])
    lines_edges.colors = o3d.utility.Vector3dVector(
        (1 - np.repeat(np.expand_dims(edge_val, 1), 3, 1)) * 1.5
    )
    # o3d.visualization.draw_geometries([pos_o3d,lines_edges])
    show_from_side([pos_o3d, lines_edges, coord])

def visualize_graph_pair(gr_pc_pair,partial_indices_t=None):

    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.4)

    geometries = [coord]
    second=False
    gr_pc_pair[1].num_nodes
    pairs=np.zeros([gr_pc_pair[1].num_nodes,2],dtype=int) 
    pairs[:,0]=np.arange(gr_pc_pair[1].num_nodes)
    if torch.is_tensor(partial_indices_t):
        
        pairs[:,1]=partial_indices_t.cpu().detach().numpy()
    else:
        pairs[:,1]=np.arange(gr_pc_pair[0].num_nodes)

    pc_o3d_list=[]
    for gr_pc in gr_pc_pair:
        
        pos_np =deepcopy( gr_pc.x.cpu().detach().numpy())
        edge_ind = gr_pc.edge_index.cpu().detach().numpy()
        edge_val = gr_pc.edge_attr.cpu().detach().numpy()
        pos_o3d = o3d.geometry.PointCloud()
        pos_o3d.points = o3d.utility.Vector3dVector(pos_np)
        if second:
            pos_o3d.translate([1.5,0,0])
            pos_o3d.paint_uniform_color([0, 0, 0.5])
        else:
            pos_o3d.paint_uniform_color([0, 0.5, 0])
        
        lines_edges = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pos_o3d, pos_o3d, list(map(tuple, edge_ind.T))
        )
        lines_edges.colors = o3d.utility.Vector3dVector(
            (1 - np.repeat(np.expand_dims(edge_val, 1), 3, 1)) * 1.5
        )
        
        pc_o3d_list.append(pos_o3d)
        geometries.extend([pos_o3d, lines_edges])
        second=True

    tuples = [tuple(l) for l in pairs]
    lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pc_o3d_list[1], pc_o3d_list[0], tuples
    )
    lines.paint_uniform_color([0.5, 0, 0.0])



    

    # show_from_side([pos_o3d])
    
    # o3d.visualization.draw_geometries([pos_o3d,lines_edges])
    show_from_side(geometries+[lines])


def visualize_graph_batch(gr_pc):
    batch = gr_pc.batch.cpu().detach().numpy()
    pos_np = gr_pc.x.cpu().detach().numpy()[:, 3:]
    edge_ind = gr_pc.edge_index.cpu().detach().numpy()
    edge_val = gr_pc.edge_attr.cpu().detach().numpy()
    # gt_np,edge_val,edge_ind,batch
    for scene in list(set(list(batch))):

        node_indexes = np.where(batch == scene)[0]
        edge_batch = np.isin(edge_ind, node_indexes)
        edges_ind_batch = edge_ind[edge_batch].reshape([2, -1])
        edges_val_batch = edge_val[edge_batch[0, :]]
        edges_ind_batch = edges_ind_batch - np.min(edges_ind_batch)

        # rand1=np.random.rand(gt_np.shape[0],1)*2
        # rand2=np.random.rand(gt_np.shape[0],1)
        # pr_np=(gt_np*rand1+pt_np*rand2)/(rand1+rand2)
        pos_batch = pos_np[node_indexes]
        pos_o3d = o3d.geometry.PointCloud()
        pos_o3d.points = o3d.utility.Vector3dVector(pos_batch)

        pos_o3d.paint_uniform_color([0, 0, 0])

        lines_edges = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pos_o3d, pos_o3d, list(map(tuple, edges_ind_batch.T))
        )

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.4)

        show_from_side([pos_o3d])
        lines_edges.colors = o3d.utility.Vector3dVector(
            (1 - np.repeat(np.expand_dims(edges_val_batch, 1), 3, 1)) * 1.5
        )
        # o3d.visualization.draw_geometries([pos_o3d,lines_edges])
        show_from_side([pos_o3d, lines_edges])


def visualize_graph_corners(gr_pc):
    batch = gr_pc.batch.cpu().detach().numpy()
    pos_np = gr_pc.x.cpu().detach().numpy()[:, 3:]
    val_np = gr_pc.y.cpu().detach().numpy()
    edge_ind = gr_pc.edge_index.cpu().detach().numpy()
    edge_val = gr_pc.edge_attr.cpu().detach().numpy()
    # gt_np,edge_val,edge_ind,batch
    for scene in list(set(list(batch))):

        node_indexes = np.where(batch == scene)[0]
        edge_batch = np.isin(edge_ind, node_indexes)
        edges_ind_batch = edge_ind[edge_batch].reshape([2, -1])
        edges_val_batch = edge_val[edge_batch[0, :]]
        edges_ind_batch = edges_ind_batch - np.min(edges_ind_batch)

        # rand1=np.random.rand(gt_np.shape[0],1)*2
        # rand2=np.random.rand(gt_np.shape[0],1)
        # pr_np=(gt_np*rand1+pt_np*rand2)/(rand1+rand2)
        val_batch = val_np[node_indexes]
        pos_batch = pos_np[node_indexes]
        pos_o3d = o3d.geometry.PointCloud()
        pos_o3d.points = o3d.utility.Vector3dVector(pos_batch)
        color_np = np.squeeze(np.repeat(np.expand_dims(val_batch, 1), 3, 1), 2)
        color_np[:, 2] = 0
        color_np[:, 0] = 1 - color_np[:, 1]
        pos_o3d.paint_uniform_color([0, 0, 0])

        lines_edges = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pos_o3d, pos_o3d, list(map(tuple, edges_ind_batch.T))
        )

        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(0.4)

        show_from_side([pos_o3d])
        pos_o3d.colors = o3d.utility.Vector3dVector(color_np)
        lines_edges.colors = o3d.utility.Vector3dVector(
            (1 - np.repeat(np.expand_dims(edges_val_batch, 1), 3, 1)) * 1.5
        )
        # o3d.visualization.draw_geometries([pos_o3d,lines_edges])
        show_from_side([pos_o3d, lines_edges])


def show_from_center(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    pose = np.eye(4)
    pose[2, 3] = 1
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()


def show_from_side(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    top_rot = Quaternion(axis=(1.0, 0.0, 0.0), degrees=200)
    pose = np.eye(4)
    pose[2, 3] = 2.0
    # pose[2,2]=1
    pose[0:3, 0:3] = top_rot.rotation_matrix
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 8
    # vis.get_render_option().light_on=False
    vis.run()
    vis.destroy_window()


def show_from_new(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    top_rot = Quaternion(axis=(0.7, 0.2, 0.0), degrees=130)
    pose = np.eye(4)
    pose[2, 3] = 2.0
    pose[0:3, 0:3] = top_rot.rotation_matrix
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 8
    vis.run()
    vis.destroy_window()


def show_with_no_light(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    top_rot = Quaternion(axis=(1.0, 1, 0.0), degrees=135)
    pose = np.eye(4)
    pose[2, 3] = 2.5
    pose[0:3, 0:3] = top_rot.rotation_matrix
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 4
    vis.get_render_option().light_on = False
    vis.run()
    vis.destroy_window()


def show_from_top(objs):
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    [vis.add_geometry(obj) for obj in objs]
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
    top_rot = Quaternion(axis=(1.0, 0, 0.0), degrees=-90)
    pose = np.eye(4)
    pose[2, 3] = 7
    pose[0:3, 0:3] = top_rot.rotation_matrix
    cam.extrinsic = pose
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()
