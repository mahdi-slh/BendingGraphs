#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:02:48 2021

@author: sc
"""
import os, math
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import open3d as o3d
import trimesh
import scipy.io as sio
import glob
from collections import defaultdict


def Parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Takes two meshes(point clouds), coloring both files with the same index order as the first one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=add_help,
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="",
        help="display all in the folder",
        required=False,
    )
    parser.add_argument(
        "-f1", "--file1", type=str, default="", help="reference ", required=False
    )
    parser.add_argument(
        "-f2", "--file2", type=str, default="", help="target", required=False
    )
    parser.add_argument(
        "-t", "--filetype", type=str, default=".ply", help="filetype", required=False
    )
    parser.add_argument(
        "-cls",
        "--with_class",
        type=int,
        default=0,
        help="try to seperate files into classes base on their name (CLS_**.ply)",
        required=False,
    )
    return parser


def check_tosca():
    basepath = "/media/sc/SSD1TB/dataset/tosca/nonrigid/"
    name1 = "cat0"
    name2 = "cat1"

    v1, t1 = load_mat(basepath + name1)
    v2, t2 = load_mat(basepath + name2)

    # generate clr
    pmin = v1.min(0)
    pmax = v1.max(0)
    clr_norm = (v1 - pmin) / (pmax - pmin)

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(v1)
    pc1.colors = o3d.utility.Vector3dVector(clr_norm)

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(v2)
    pc2.colors = o3d.utility.Vector3dVector(clr_norm)

    o3d.visualization.draw_geometries([pc1, pc2.translate([50, 0, 0])])


def compute_normalize_color(pts):
    pmin = pts.min(0)
    pmax = pts.max(0)
    return (pts - pmin) / (pmax - pmin)


def load_mesh(path):
    if path.find(".mat") >= 0:
        # load mat
        vts, tris = load_mat(path)
        return trimesh.Trimesh(vts, tris, process=False)
    else:
        return trimesh.load_mesh(path, process=False)


def show_pair(f1, f2):
    m1 = load_mesh(f1)
    m2 = load_mesh(f1)
    assert m1.vertices.shape == m1.vertices.shape
    clr1 = compute_normalize_color(m1.vertices)
    m1.visual.vertex_colors = clr1
    m2.visual.vertex_colors = clr1

    # calculate offset for display purpose
    offset = m1.bounding_box.bounds[1] - m1.bounding_box.bounds[0]
    offset[0] = 0
    offset[2] = 0
    offset *= 2
    m2.apply_translation(offset)
    trimesh.Scene([m1, m2]).show()


def show_all(paths):
    meshes = list()
    for path in paths:
        mesh = load_mesh(path)
        mesh.apply_translation(-mesh.centroid)  # move to center
        diameter = (mesh.bounding_sphere.bounds[1] - mesh.bounding_sphere.bounds[0])[0]
        mesh.apply_scale(1 / diameter)
        meshes.append(mesh)
    # use the first one for coloring
    assert len(meshes) > 1
    mesh = meshes[0]
    clr = compute_normalize_color(mesh.vertices)
    # mean_off = np.zeros([3])
    for mesh in meshes:
        mesh.visual.vertex_colors = clr
        # mean_off += mesh.bounding_box.bounds[1]-mesh.bounding_box.bounds[0]
    # offset = mean_off / len(meshes)
    # compute offset
    # offset = mesh.bounding_box.bounds[1]-mesh.bounding_box.bounds[0]

    # apply all
    x_incre = 0
    y_incre = 0
    split = int(math.sqrt(len(meshes)))
    for i in range(len(meshes)):
        # y
        # off_ = np.copy(offset)
        off_ = np.zeros([3])
        off_[0] = 1.0 * x_incre
        off_[1] = 1.0 * y_incre
        off_[2] = 0
        meshes[i].apply_translation(off_)

        y_incre += 1
        if (i + 1) % split == 0:
            # print('plus')
            x_incre += 1
            y_incre = 0
    trimesh.Scene(meshes).show()


if __name__ == "__main__":
    args = Parser().parse_args()
    # pair
    if len(args.file1) > 0 and len(args.file2) > 0:
        show_pair(args.file1, args.file2)

    if len(args.folder) > 0:
        files = glob.glob(os.path.join(args.folder, "*" + args.filetype))
        filenames = [f.split("/")[-1].split(".")[0] for f in files]

        if args.with_class > 0:
            classnames = [
                "".join([i for i in name if not i.isdigit()]) for name in filenames
            ]
            classnames = np.unique(classnames)
            print(classnames)
            seqmap = defaultdict(list)
            for name in classnames:
                for path in filenames:
                    if name in path:
                        seqmap[name].append(path)
                seqmap[name] = sorted(seqmap[name])

            for k, v in seqmap.items():
                if len(v) <= 1:
                    continue
                paths = [os.path.join(args.folder, name + args.filetype) for name in v]
                show_all(paths)
        else:
            paths = [
                os.path.join(args.folder, name + args.filetype) for name in filenames
            ]
            show_all(paths)

    pass
