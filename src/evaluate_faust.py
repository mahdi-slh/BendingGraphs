"""Evaluate a BendingGraphs checkpoint on MPI-FAUST.

Loads a checkpoint produced by ``train_surreal.py``, builds the FAUST
test split, and prints the deformable-shape metrics — bijective rate,
geodesic average, geodesic average after one OT round, and Chamfer
distance.
"""
from __future__ import annotations

import argparse
import os
import sys

import configs
from configs import MPIFAUST_DIR, training_configs

from evaluation.eval_deform import EvaluatorDeform
from loaders.faust_syn_dataset import FaustSynDataset
from models.model import init_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BendingGraphs on FAUST")
    parser.add_argument("--model", required=True,
                        help="Path to a trained .model checkpoint.")
    parser.add_argument("--data-mode", default="clean",
                        choices=["clean", "noise", "patch", "patchnoise"])
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="If >0, only evaluate the first N test pairs.")
    parser.add_argument("--visualize", action="store_true",
                        help="Pop up Open3D windows for each pair (needs DISPLAY).")
    args = parser.parse_args()

    if not os.path.isdir(MPIFAUST_DIR):
        sys.exit(f"FAUST root not found at {MPIFAUST_DIR}")

    settings = training_configs(data_mode=args.data_mode)
    print(f"==> Loading checkpoint from {args.model}")
    model, _, epoch = init_model(args.model, settings)
    model.eval()

    print("==> Building FAUST test dataset...")
    test_data = FaustSynDataset(root=MPIFAUST_DIR, train=False, mode=args.data_mode)
    if args.max_pairs > 0:
        test_data.process_list = test_data.process_list[: args.max_pairs]
        test_data.data = test_data.data[: args.max_pairs]
        test_data.meta = test_data.meta[: args.max_pairs]

    evaluator = EvaluatorDeform(visualize=args.visualize)
    metrics = evaluator.eval(test_data=test_data, net=model, epoch=epoch)
    print("==> Metrics:")
    for k in sorted(metrics):
        print(f"    {k}: {metrics[k]:.5f}")


if __name__ == "__main__":
    main()
