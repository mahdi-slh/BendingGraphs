import numpy as np
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.utils.graph import single_source_shortest_path_length
import scipy.io as sio
import gdist
from utils.pyFM.mesh import TriMesh
import open3d as o3d


def get_mesh_graph(mesh):
    mesh.compute_adjacency_list()
    aj_list = mesh.adjacency_list
    aj_matrix = np.zeros([len(aj_list), len(aj_list)])
    for i in range(len(aj_list)):
        for elem in aj_list[i]:
            aj_matrix[i, elem] = 1
            aj_matrix[elem, i] = 1
    if not aj_matrix.flags["C_CONTIGUOUS"]:
        print("hi")
    dijkstra = graph_shortest_path(aj_matrix, directed=False)
    return dijkstra.astype("int64")


def compute_geod(mesh, seeds):

    vertices = np.asarray(mesh.vertices).astype(np.float64)
    triangles = np.asarray(mesh.triangles).astype(np.int32)
    seeds_np = np.asarray(seeds).astype(np.int32)
    mat_geod = gdist.distance_matrix_of_selected_points(vertices, triangles, seeds_np)
    return mat_geod


def normalize_unit_sphere(x):
    if isinstance(x, list):
        ele = x[0]
    else:
        ele = x
    if isinstance(ele, o3d.cpu.pybind.geometry.TriangleMesh):
        # center
        ele.translate(-ele.get_center())
        dim = ele.get_max_bound() - ele.get_min_bound()
        scale = 2.0 / np.sqrt((dim**2).sum())
        # for ele in x:
        ele.scale(scale, ele.get_center())
    else:
        raise NotImplementedError()
    return x

def normalize_unit_cube(x):
    if isinstance(x, list):
        ele = x[0]
    else:
        ele = x
    if isinstance(ele, o3d.cpu.pybind.geometry.TriangleMesh):
        # center
        ele.translate(-ele.get_center())
        dim = ele.get_max_bound() - ele.get_min_bound()

        scale = 1.0 / np.max(dim)
        # for ele in x:
        ele.scale(scale, ele.get_center())
    else:
        raise NotImplementedError()
    return x

def o3d_to_trimesh(x):
    vertices = np.asarray(x.vertices).astype(np.float64)
    triangles = np.asarray(x.triangles).astype(np.int32)
    return TriMesh(vertices=vertices, faces=triangles)


def get_dijkstra(mesh):
    """
    Get shortest path on graph with Dijkstra's algorithm

    """

    mesh.compute_adjacency_list()

    aj_list = mesh.adjacency_list
    aj_matrix = np.zeros([len(aj_list), len(aj_list)])
    for i in range(len(aj_list)):
        for elem in aj_list[i]:
            aj_matrix[i, elem] = 1
            aj_matrix[elem, i] = 1

    if not aj_matrix.flags["C_CONTIGUOUS"]:
        print("hi")
    dijkstra = graph_shortest_path(aj_matrix, directed=False)
    return dijkstra


def get_single_dijkstra(mesh, idx: int, max_dist: int = None):
    mesh.compute_adjacency_list()
    aj_list = mesh.adjacency_list
    aj_matrix = np.zeros([len(aj_list), len(aj_list)])
    for i in range(len(aj_list)):
        for elem in aj_list[i]:
            aj_matrix[i, elem] = 1
            aj_matrix[elem, i] = 1

    if not aj_matrix.flags["C_CONTIGUOUS"]:
        print("hi")

    dijkstra = single_source_shortest_path_length(aj_matrix, idx, max_dist)
    return dijkstra
