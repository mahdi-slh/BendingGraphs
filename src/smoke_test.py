"""Tiny synthetic-mesh smoke test for the training/eval pipeline.

Runs a single forward + backward pass through ``trainer.run_epoch`` and
one ``EvaluatorDeform.eval`` round on hand-crafted Open3D primitives.
Exercises everything the real FAUST / SURREAL / SMAL / TOSCA pipeline
does *except* dataset download, so it's the cheapest way to confirm the
pipeline is wired correctly on a fresh checkout.

Usage:
    python src/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile

import configs  # noqa: F401  applies compat shims
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from configs import VALIDATION_SPLIT, training_configs
from evaluation.eval_deform import EvaluatorDeform
from loaders.matching_dataset_base import Matching_Dataset
from models.model import init_model
from trainer import trainer
from utils.utils import MaskedSampler, create_validation_masks


class _SyntheticPair(Matching_Dataset):
    """A small dataset built from perturbed Open3D spheres.

    Inherits from ``Matching_Dataset`` so we get the patch + graph
    construction (geodesic seeds, FPS, Dijkstra, etc.) for free.  Always
    runs the train-style ``generate`` so the ``n`` (negative) patches are
    populated; eval just ignores them.
    """

    def __init__(self, num_pairs: int = 2):
        super().__init__(train=True, mode="clean")
        self.points_per_shape = 64
        self.THREADS = 0
        self.DEBUG_PLOT = False
        self._cache = []
        for i in range(num_pairs):
            mesh_a = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            mesh_p = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
            verts = np.asarray(mesh_p.vertices)
            verts[:, 0] += 0.05 * np.sin(verts[:, 1] * 3 + i)
            mesh_p.vertices = o3d.utility.Vector3dVector(verts)
            mesh_a.compute_vertex_normals()
            mesh_p.compute_vertex_normals()
            data = self.generate(mesh_a, mesh_p, name_a=f"sphere_a_{i}", name_p=f"sphere_p_{i}")
            self._cache.append(data)

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx):
        data = self._cache[idx]
        frame_data = {"a": data["a"], "p": data["p"]}
        if data.get("n"):
            frame_data["n"] = data["n"]
        meta = {k: v for k, v in data.items() if k not in ("a", "p", "n")}
        # Drop any None values to keep the PyG collator happy.
        meta = {k: v for k, v in meta.items() if v is not None}
        return frame_data, meta


def _passthrough_collate(samples):
    # mb_size is always 1 for the deformable pipeline; the trainer expects
    # a (frame_data, meta) tuple from ``next(loader)`` rather than a batched
    # collation thereof, so we hand the single sample back unchanged.
    assert len(samples) == 1
    return samples[0]


def _build_loader(dataset: Dataset, mb_size: int = 1):
    train_mask, _val_mask = create_validation_masks(dataset, VALIDATION_SPLIT)
    if not train_mask:
        train_mask = list(range(len(dataset)))
    return DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=BatchSampler(
            MaskedSampler(len(dataset), train_mask),
            batch_size=mb_size,
            drop_last=False,
        ),
        collate_fn=_passthrough_collate,
    )


def main() -> None:
    print("==> Building synthetic spheres dataset...")
    train_set = _SyntheticPair(num_pairs=2)
    test_set = _SyntheticPair(num_pairs=1)

    settings = training_configs(
        data_mode="clean",
        epoch_size=1,
        mb_size=1,
        alpha=1.0,
        m=0.7,
        learning_rate=1e-4,
        identifier="smoke",
        file_tag="smoke",
        dataset="synth",
    )

    train_loader = _build_loader(train_set)
    val_loader = _build_loader(train_set)

    print("==> Constructing trainer (deformable surface_type)...")
    feat_trainer = trainer(
        settings,
        train_loader,
        val_loader,
        dataset_test=test_set,
        surface_type="deform",
        pretrain=False,
        model_path=None,
    )

    print("==> Running 1 epoch (forward + backward)...")
    feat_trainer.train()

    print("==> Running stand-alone EvaluatorDeform...")
    evaluator = EvaluatorDeform(visualize=False)
    metrics = evaluator.eval(test_data=test_set, net=feat_trainer.model, epoch=0)
    print("==> Eval metrics:")
    for k in sorted(metrics):
        print(f"    {k}: {metrics[k]:.5f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    sys.exit(main() or 0)
