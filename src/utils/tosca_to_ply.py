#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:11:05 2021

@author: sc
"""
import os
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import open3d as o3d
import scipy.io as sio


def Parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Convert *.tri and *.vert to *.ply.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "-f", "--foldername", type=str, help="input folder location", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output folder location", required=True
    )
    return parser


def read_txt_to_list(file):
    output = []
    with open(file, "r") as f:
        for line in f:
            entry = line.rstrip().lower()
            output.append(entry)
    return output


def list_to_numpy(x: list):
    o = np.zeros([len(x), 3])
    for i in range(len(x)):
        line = x[i]
        tokens = line.split(" ")
        o[i][0] = float(tokens[0])
        o[i][1] = float(tokens[1])
        o[i][2] = float(tokens[2])
    return o


if __name__ == "__main__":
    args = Parser().parse_args()
    Path(args.output).mkdir(exist_ok=True)

    names = [
        str(s).split("/")[-1].split(".")[0] for s in Path(args.foldername).glob("*.tri")
    ]

    p_vrt = ".vert"
    p_tri = ".tri"
    p_mat = ".mat"

    for name in sorted(names):
        fv = os.path.join(args.foldername, name + p_vrt)
        ft = os.path.join(args.foldername, name + p_tri)
        lvts = read_txt_to_list(fv)
        ltris = read_txt_to_list(ft)
        vts = list_to_numpy(lvts)
        tris = list_to_numpy(ltris).astype(np.int32)
        tris -= 1  # shift from 1 -> 0

        # load mat
        # mat = sio.loadmat(os.path.join(args.foldername,name+p_mat))
        # tris = mat['surface']['TRIV'][0][0] - 1
        # vts = np.concatenate(
        #     (mat['surface']['X'][0][0],mat['surface']['Y'][0][0],mat['surface']['Z'][0][0]),
        #     axis=1)

        tri = o3d.geometry.TriangleMesh()
        tri.vertices = o3d.utility.Vector3dVector(vts)
        tri.triangles = o3d.utility.Vector3iVector(tris)
        o3d.io.write_triangle_mesh(os.path.join(args.output, name + ".off"), tri)

        # debug plot
        # tri.compute_triangle_normals()
        # tri.compute_vertex_normals()
        # o3d.visualization.draw_geometries([tri])
        # break
