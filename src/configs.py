import torch
import random
import json
import string
from os import path
import os
from threading import Thread
import time
import sys, select


FEATURE_SIZE = 64


class training_configs:
    def __init__(
        self,
        data_mode="clean",
        alpha=10,
        epoch_size=50,
        multi_gpu=0,
        mb_size=72,
        report_size=25,
        learning_rate=1e-3,
        dataset="",
        m=1,
    ):

        # cuda_available = self.device == torch.device("cuda:0")

        if torch.cuda.is_available:
            torch.cuda.set_device(torch.cuda.current_device())
            print("Running on GPU...")

        else:
            print("Running on CPU...")

        self.data_mode = data_mode
        self.m = m
        self.alpha = alpha
        self.epoch_size = epoch_size
        self.multi_gpu = multi_gpu
        self.dataset = ""
        # if VERSION == 1:

        #     self.mb_size = mb_size
        #     self.mb_size = 1

        # else:
        self.mb_size = 1
        self.report_size = report_size
        self.learning_rate = learning_rate

        self.identifier = "".join(random.choices(string.ascii_lowercase, k=5))
        self.model_folder = MODEL_PATH + self.identifier
        self.file_tag = self.identifier
        # self.save_config()

    def timed_input(self):
        answer = None
        time.sleep(2)
        if answer != None:
            return
        print("Continue")

    def input_params(self):
        print(self.__dict__)
        print("Set the identifier or pass: ")

        i, o, e = select.select([sys.stdin], [], [], 10)

        if i:
            entry = sys.stdin.readline().strip()
            if entry is not None and entry != "":
                self.identifier = entry
                print("Setting model name: {}".format(self.to_string()))
        # if entry is not "" and entry is not None:
        #     for param in self.__dict__:

        #         print("{} with value {} , Enter new value or pres Enter to pass:".format(
        #             param, self.__dict__[param]))
        #         entry = input()

        #         if entry is not "" and entry is not None:
        #             if param == 'data_mode' or param == 'file_tag':
        #                 self.__dict__[param] = entry
        #             else:

        #                 self.__dict__[param] = float(
        #                     entry) if '.' in entry else int(entry)
        #             print("Parameter updated to {}".format(
        #                 self.__dict__[param]))

    def to_string(self, epoch=None):
        if FIXED_GRAPH:
            if epoch:
                return "{}-{}a{}m{}lr{}f{}k{}e{}".format(
                    self.identifier,
                    self.dataset,
                    self.alpha,
                    self.m,
                    self.learning_rate,
                    FEATURE_SIZE,
                    GRAPH_SIZE,
                    epoch,
                )
            else:
                return "{}-{}a{}m{}lr{}f{}k{}".format(
                    self.identifier,
                    self.dataset,
                    self.alpha,
                    self.m,
                    self.learning_rate,
                    FEATURE_SIZE,
                    GRAPH_SIZE,
                )
        else:
            if epoch:
                return "{}-{}a{}m{}lr{}f{}r{}e{}".format(
                    self.identifier,
                    self.dataset,
                    self.alpha,
                    self.m,
                    self.learning_rate,
                    FEATURE_SIZE,
                    GEODESIC_CUT,
                    epoch,
                )
            else:
                return "{}-{}a{}m{}lr{}f{}r{}".format(
                    self.identifier,
                    self.dataset,
                    self.alpha,
                    self.m,
                    self.learning_rate,
                    FEATURE_SIZE,
                    GEODESIC_CUT,
                )

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_config(self):

        if not path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        with open(self.model_folder + "/configs.txt", "w") as outfile:
            json.dump(self.toJSON(), outfile)


FAT_CAMERA_INTRINSICS = {
    "width": 960,
    "height": 540,
    "fx": 768.16058349609375,
    "fy": 768.16058349609375,
    "cx": 480,
    "cy": 270,
}

SYN_PRIM_CAMERA_INTRINSICS = {
    "width": 100,
    "height": 100,
    "fx": 100,  # 155.77,
    "fy": 100,  # 155.77,
    "cx": 49.5,
    "cy": 49.5,
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PREFIX = "../../../dataset/"
# PREFIX = '/media/sc/SSD1TB/dataset/'
COLOR_SRC = PREFIX + "fat/single/{}/{}.jpg"
DEPTH_SRC = PREFIX + "fat/single/{}/{}.depth.png"
SEG_SRC = PREFIX + "fat/single/{}/{}.seg.png"
IMG_SRC = PREFIX + "fat/single/{}/{}.jpg"

OBJECT_CLASS_LABELS = PREFIX + "fat/classes.txt"
NUMBER_OF_OBJECTS = 20

OBJECT_PRESENT_FRAMES_PATH = PREFIX + "fat/object_pres/{}.txt"
OBJECT_GROUND_TRUTH_POSE_PATH = PREFIX + "fat/poses/{}.txt"

SYN_PRIM_DIR = PREFIX + "syn_prim/"
BODY_DIR = PREFIX + "transformed_body/"
MPIFAUST_DIR = PREFIX + "MPI-FAUST/"
SURREAL_DATA_DIR = PREFIX + "/SURREAL/smpl_data/"
SURREAL_MODEL_DIR = PREFIX + "/SMPL/SMPL_python_v.1.1.0/smpl/models"
SURREAL_MODEL_DIR = PREFIX + "/SMPL/SMPL_python_v.1.0.0/smpl/models"
TOSCA_DIR = PREFIX + "tosca/toscahires-mat/"
SMAL_DIR = PREFIX + "smal/smal_online_V1.0/"
SHREC19_DIR = PREFIX + "/SHREC19_matching_humans/"
SHREC_PARTIAL_DIR = PREFIX + "/shrec_partial/"
MATCH3D_DIR = PREFIX + "3dmatch/scenes/"
MODELNET_DIR = PREFIX + "modelnet40_aligned/"
GRAPH_DATASET = PREFIX + "graph/"
FAT_DIR = PREFIX + "fat/graphed/"
MODEL_PATH = "../model/"
PRETRAINED_MODEL_PATH = "../checkpoints/pretrained/"
TRAINED_MODEL_PATH = "../checkpoints/models/"
LOG_PATH = "../checkpoints/logs/"

WINDOW_SIZE = 15
GRAPH_SIZE = WINDOW_SIZE**2
FIXED_GRAPH = False
GEODESIC_CUT = 7
EUCLIDEAN_CUT = 0.15
VALIDATION_SPLIT = 0.9
KNN = False


NUM_POINTS_FAUST = 6890
NUM_POINTS_SAMPLE_FAUST = 4096
# VERSION = 2
