"""Evaluate a BendingGraphs checkpoint on TOSCA.

Loads a checkpoint produced by ``train_smal.py``, builds the TOSCA test
split for a given target species (cat / horse / dog / centaur / david / ...)
and prints the deformable-shape metrics.
"""
from __future__ import annotations

import argparse
import os
import sys

import configs
from configs import TOSCA_DIR, training_configs

from evaluation.eval_deform import EvaluatorDeform
from loaders.tosca_dataset import ToscaDataset, base_model_names
from models.model import init_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BendingGraphs on TOSCA")
    parser.add_argument("--model", required=True,
                        help="Path to a trained .model checkpoint.")
    parser.add_argument("--target-shape", default="horse",
                        choices=base_model_names + ["random"])
    parser.add_argument("--data-mode", default="clean",
                        choices=["clean", "noise", "patch", "patchnoise"])
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="If >0, only evaluate the first N test pairs.")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(TOSCA_DIR):
        sys.exit(f"TOSCA root not found at {TOSCA_DIR}")

    settings = training_configs(data_mode=args.data_mode)
    print(f"==> Loading checkpoint from {args.model}")
    model, _, epoch = init_model(args.model, settings)
    model.eval()

    print(f"==> Building TOSCA test dataset (target_shape={args.target_shape})...")
    test_data = ToscaDataset(
        root=TOSCA_DIR, train=False, mode=args.data_mode,
        target_shape=args.target_shape,
    )
    if args.max_pairs > 0 and hasattr(test_data, "process_list"):
        test_data.process_list = test_data.process_list[: args.max_pairs]

    evaluator = EvaluatorDeform(visualize=args.visualize)
    metrics = evaluator.eval(test_data=test_data, net=model, epoch=epoch)
    print("==> Metrics:")
    for k in sorted(metrics):
        print(f"    {k}: {metrics[k]:.5f}")


if __name__ == "__main__":
    main()
