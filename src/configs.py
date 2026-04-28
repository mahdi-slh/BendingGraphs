"""Project-wide configuration and dataset paths.

Reorganised to follow the Graphite layout:

* every path is repo-relative by default, override via env vars.
* ``training_configs`` exposes an argparse-driven ``input_params``.
* ``apply_compat`` from ``_compat.py`` runs as a side-effect of importing this
  module, so legacy ``from configs import *`` call sites pick up the modern
  Open3D / NumPy / scikit-learn / SciPy / PyG APIs without further changes.
"""
import argparse
import json
import os
import random
import string
import sys
import select
import time
from os import path
from threading import Thread

import torch

# Apply Open3D / numpy / sklearn compatibility shims as a side-effect of
# importing ``configs``.  This avoids touching every call site.
from _compat import apply_compat  # noqa: E402
apply_compat()


# ---------------------------------------------------------------------------
# Hyper-parameters / sizing constants
# ---------------------------------------------------------------------------
FEATURE_SIZE = 64
WINDOW_SIZE = 15
GRAPH_SIZE = WINDOW_SIZE ** 2
FIXED_GRAPH = False
GEODESIC_CUT = 7
EUCLIDEAN_CUT = 0.15
VALIDATION_SPLIT = 0.9
KNN = False

NUM_POINTS_FAUST = 6890
NUM_POINTS_SAMPLE_FAUST = 4096


# ---------------------------------------------------------------------------
# Filesystem paths.  Roots are relative to the repo root by default but can
# be overridden through env vars (handy when running on different machines).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_ROOT = os.environ.get("BENDING_DATA_ROOT", os.path.join(_REPO_ROOT, "data"))
_MODELS_ROOT = os.environ.get("BENDING_MODELS_ROOT", os.path.join(_REPO_ROOT, "models"))
_LOG_ROOT = os.environ.get("BENDING_LOG_ROOT", os.path.join(_REPO_ROOT, "logs"))
OUTPUTS_DIR = os.environ.get("BENDING_OUTPUTS_DIR", os.path.join(_REPO_ROOT, "outputs"))

os.makedirs(_MODELS_ROOT, exist_ok=True)
os.makedirs(_LOG_ROOT, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Per-dataset roots
MPIFAUST_DIR = os.environ.get("MPIFAUST_DIR", os.path.join(_DATA_ROOT, "MPI-FAUST") + "/")
SMAL_DIR = os.environ.get("SMAL_DIR", os.path.join(_DATA_ROOT, "smal") + "/")
TOSCA_DIR = os.environ.get("TOSCA_DIR", os.path.join(_DATA_ROOT, "tosca") + "/")
SHREC19_DIR = os.environ.get("SHREC19_DIR", os.path.join(_DATA_ROOT, "shrec19") + "/")
SHREC_PARTIAL_DIR = os.environ.get("SHREC_PARTIAL_DIR", os.path.join(_DATA_ROOT, "shrec_partial") + "/")
SURREAL_DATA_DIR = os.environ.get("SURREAL_DATA_DIR", os.path.join(_DATA_ROOT, "surreal", "smpl_data") + "/")
SURREAL_MODEL_DIR = os.environ.get("SURREAL_MODEL_DIR", os.path.join(_DATA_ROOT, "surreal", "smpl_models") + "/")
MATCH3D_DIR = os.environ.get("MATCH3D_DIR", os.path.join(_DATA_ROOT, "match3d") + "/")
MODELNET_DIR = os.environ.get("MODELNET_DIR", os.path.join(_DATA_ROOT, "modelnet") + "/")
GRAPH_DATASET = os.environ.get("GRAPH_DATASET", os.path.join(_DATA_ROOT, "graph") + "/")
SYN_PRIM_DIR = os.environ.get("SYN_PRIM_DIR", os.path.join(_DATA_ROOT, "syn_prim") + "/")
BODY_DIR = os.environ.get("BODY_DIR", os.path.join(_DATA_ROOT, "transformed_body") + "/")
FAT_DIR = os.environ.get("FAT_DIR", os.path.join(_DATA_ROOT, "fat") + "/")

# Model + log directories.  Names are kept as they are referenced from
# `utils.utils.get_model_path` / `get_log_path`.
MODEL_PATH = _MODELS_ROOT + "/"
PRETRAINED_MODEL_PATH = os.path.join(_MODELS_ROOT, "pretrained_")
TRAINED_MODEL_PATH = os.path.join(_MODELS_ROOT, "trained_")
LOG_PATH = _LOG_ROOT + "/"

# FAT helper paths (preserved for the FAT-specific code paths).
OBJECT_CLASS_LABELS = os.path.join(FAT_DIR, "classes.txt")
OBJECT_PRESENT_FRAMES_PATH = os.path.join(FAT_DIR, "object_pres", "{}.txt")
OBJECT_GROUND_TRUTH_POSE_PATH = os.path.join(FAT_DIR, "poses", "{}.txt")
COLOR_SRC = os.path.join(FAT_DIR, "single", "{}", "{}.jpg")
DEPTH_SRC = os.path.join(FAT_DIR, "single", "{}", "{}.depth.png")
SEG_SRC = os.path.join(FAT_DIR, "single", "{}", "{}.seg.png")
IMG_SRC = os.path.join(FAT_DIR, "single", "{}", "{}.jpg")
NUMBER_OF_OBJECTS = 20

FAT_CAMERA_INTRINSICS = {
    "width": 960, "height": 540,
    "fx": 768.16058349609375, "fy": 768.16058349609375,
    "cx": 480, "cy": 270,
}
SYN_PRIM_CAMERA_INTRINSICS = {
    "width": 100, "height": 100,
    "fx": 100, "fy": 100, "cx": 49.5, "cy": 49.5,
}
SYN_PRIM_CAMERA_INTRINSICS_2 = dict(SYN_PRIM_CAMERA_INTRINSICS)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Settings object
# ---------------------------------------------------------------------------
class training_configs:
    """Hyper-parameters bag.

    Defaults follow the original BendingGraphs paper protocol; pass
    ``input_params(argv)`` to override from the command line.
    """

    def __init__(
        self,
        data_mode: str = "clean",
        alpha: float = 10.0,
        epoch_size: int = 50,
        multi_gpu: int = 0,
        mb_size: int = 1,
        report_size: int = 25,
        learning_rate: float = 1e-3,
        dataset: str = "",
        m: float = 1.0,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        lr_schedule: str = "none",
        file_tag: str = "",
        identifier: str = "",
        cuda: bool = None,
    ):
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
            print("Running on GPU...")
        else:
            print("Running on CPU...")

        self.data_mode = data_mode
        self.alpha = alpha
        self.m = m
        self.epoch_size = epoch_size
        self.multi_gpu = multi_gpu
        # The original code forces mb_size = 1 because the matching head
        # operates on a single shape pair at a time; keep that behaviour but
        # let it be overridden via CLI for experiments.
        self.mb_size = mb_size
        self.report_size = report_size
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule
        self.cuda = torch.cuda.is_available() if cuda is None else bool(cuda)

        self.identifier = identifier or "".join(random.choices(string.ascii_lowercase, k=5))
        self.file_tag = file_tag or self.identifier
        self.model_folder = MODEL_PATH + self.identifier

    # ----- argparse-driven CLI overrides ------------------------------------
    def input_params(self, argv=None):
        """Optionally parse CLI arguments and update fields in-place.

        If ``argv`` is None and stdin is interactive, we also honour the
        original "press enter to keep the random tag" prompt for backwards
        compatibility, but with a 2-second timeout so unattended runs don't
        hang.
        """
        parser = argparse.ArgumentParser(description="BendingGraphs training configuration")
        parser.add_argument("--identifier", default=self.identifier,
                            help="Run id used for checkpoint / log filenames.")
        parser.add_argument("--file-tag", dest="file_tag", default=self.file_tag)
        parser.add_argument("--dataset", default=self.dataset,
                            help="Dataset name appended to to_string().")
        parser.add_argument("--mb-size", dest="mb_size", type=int, default=self.mb_size)
        parser.add_argument("--epoch-size", dest="epoch_size", type=int, default=self.epoch_size)
        parser.add_argument("--report-size", dest="report_size", type=int, default=self.report_size)
        parser.add_argument("--alpha", type=float, default=self.alpha)
        parser.add_argument("--m", type=float, default=self.m)
        parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                            default=self.learning_rate)
        parser.add_argument("--data-mode", dest="data_mode", default=self.data_mode,
                            choices=["clean", "noise", "patch", "patchnoise"])
        parser.add_argument("--multi-gpu", dest="multi_gpu", type=int, default=self.multi_gpu)
        parser.add_argument("--weight-decay", dest="weight_decay", type=float,
                            default=self.weight_decay)
        parser.add_argument("--grad-clip", dest="grad_clip", type=float,
                            default=self.grad_clip,
                            help="Max gradient norm; <=0 disables clipping.")
        parser.add_argument("--lr-schedule", dest="lr_schedule", default=self.lr_schedule,
                            choices=["cosine", "step", "none"])
        parser.add_argument("--cuda", type=int, default=int(self.cuda))
        # Allow training routines to add more flags later by parsing
        # known-only.
        args, _ = parser.parse_known_args(argv)

        self.identifier = args.identifier
        self.file_tag = args.file_tag or self.identifier
        self.dataset = args.dataset
        self.mb_size = args.mb_size
        self.epoch_size = args.epoch_size
        self.report_size = args.report_size
        self.alpha = args.alpha
        self.m = args.m
        self.learning_rate = args.learning_rate
        self.data_mode = args.data_mode
        self.multi_gpu = args.multi_gpu
        self.weight_decay = args.weight_decay
        self.grad_clip = args.grad_clip
        self.lr_schedule = args.lr_schedule
        self.cuda = bool(args.cuda) and torch.cuda.is_available()

        self.model_folder = MODEL_PATH + self.identifier
        return self

    # ----- legacy interactive prompt (used to be `input_params`) -----------
    def timed_input(self):
        time.sleep(2)
        print("Continue")

    def to_string(self, epoch=None):
        if FIXED_GRAPH:
            base = "{}-{}a{}m{}lr{}f{}k{}".format(
                self.identifier, self.dataset, self.alpha, self.m,
                self.learning_rate, FEATURE_SIZE, GRAPH_SIZE,
            )
        else:
            base = "{}-{}a{}m{}lr{}f{}r{}".format(
                self.identifier, self.dataset, self.alpha, self.m,
                self.learning_rate, FEATURE_SIZE, GEODESIC_CUT,
            )
        if epoch:
            base = "{}e{}".format(base, epoch)
        return base

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_config(self):
        if not path.exists(self.model_folder):
            os.makedirs(self.model_folder, exist_ok=True)
        with open(os.path.join(self.model_folder, "configs.txt"), "w") as outfile:
            json.dump(self.toJSON(), outfile)
