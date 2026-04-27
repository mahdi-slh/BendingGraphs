from utils.pyFM.mesh import TriMesh
import numpy as np
import os
import glob
from tqdm import tqdm
import scipy.sparse.linalg


if __name__ == "__main__":
    source_folder = "/home/data/dataset/MPI-FAUST/training/registrations_off/"
    # targ_folder = '/home/data/dataset/MPI-FAUST/training/registrations_off/'
    file_list = glob.glob(source_folder + "*.off")
    for file_path in tqdm(sorted(file_list)):

        mesh = TriMesh(file_path)
        mesh_geod = mesh.get_geodesic(verbose=False)
        np.save(file_path + ".npy", mesh_geod)
