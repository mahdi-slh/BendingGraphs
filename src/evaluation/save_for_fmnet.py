import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from torch_geometric.data import DataLoader, Batch
from torch.utils.data import DataLoader as UtilsDataLoader
from torch.utils.data import Sampler, BatchSampler
from torch import nn
from utils.visualize_graph import show_from_center
import copy

# from typing_extensions import final

from loaders.modelnet_dataset import ModelNetDataset
from loaders.augmentations import copy_pc
from utils.pointcloud_utils import create_random_patches
from graph import visualize_torch_graphs_local

from utils.maths import *
from utils.utils import *
from utils.log import log
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def test(settings, net, test_data):

    net.eval()

    # descriptors={x:[] for x in test_data.object_ids}
    for data in tqdm(test_data):

        # scene = data['scene']
        desc = get_desc_from_net(settings, net, data)

        file_name = data["name"][:-8] + ".npy"
        filepath = "utils/FMNet/data/graphite/train/" + file_name
        np.save(filepath, desc.cpu().numpy())


def get_desc_from_net(settings, model, data):

    with torch.no_grad():
        data_s_1 = [
            model(
                Batch.from_data_list(data["a"][i : i + settings.mb_size]).to(
                    torch.cuda.current_device()
                )
            )
            for i in range(0, len(data["a"]), settings.mb_size)
        ]

    g_result = merge_dict(data_s_1)
    g_desc = g_result["descriptors"]

    return g_desc
