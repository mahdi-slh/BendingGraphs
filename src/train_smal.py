"""Train BendingGraphs on SMAL → evaluate on TOSCA.

This is the cross-dataset animal-shape pipeline from the *Bending Graphs*
paper: SMAL pairs are generated on the fly from the SMAL parametric model
(one shape per ``cluster_means`` row), and evaluation runs after each
epoch on the TOSCA high-resolution meshes for the same target species.

Examples
--------

    python src/train_smal.py --target-shape horse
    python src/train_smal.py --target-shape cat
    python src/train_smal.py --target-shape random   # mix all 5 SMAL species
    python src/train_smal.py --no-eval               # train only
"""
from __future__ import annotations

import argparse
import os
import sys

import configs
from configs import SMAL_DIR, TOSCA_DIR, VALIDATION_SPLIT, training_configs

from torch.utils.data import BatchSampler, DataLoader

import wandb_logger
from loaders.smal_dataset import SMALDataset, base_model_names
from loaders.tosca_dataset import ToscaDataset
from utils.utils import MaskedSampler, create_validation_masks, get_model_path
from trainer import trainer


def _passthrough_collate(samples):
    if len(samples) != 1:
        raise ValueError(
            f"_passthrough_collate expects mb_size=1; got {len(samples)} samples."
        )
    return samples[0]


def _build_settings() -> training_configs:
    parser = argparse.ArgumentParser(description="BendingGraphs SMAL training")
    parser.add_argument("--data-mode", default="clean",
                        choices=["clean", "noise", "patch", "patchnoise"])
    parser.add_argument("--target-shape", default="horse",
                        choices=base_model_names + ["random"],
                        help=("SMAL species to train on (or 'random').  Per-epoch "
                              "TOSCA evaluation uses the same target shape; pick a "
                              "TOSCA-compatible class (cat, dog, horse) for cross-"
                              "dataset numbers."))
    parser.add_argument("--mb-size", dest="mb_size", type=int, default=1)
    parser.add_argument("--epoch-size", dest="epoch_size", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--identifier", default="")
    parser.add_argument("--file-tag", dest="file_tag", default="smal")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip per-epoch evaluation on TOSCA.")
    parser.add_argument("--eval-on", default="tosca",
                        choices=["tosca", "smal"],
                        help=("Where to run per-epoch evaluation.  Default is "
                              "TOSCA (cross-dataset, paper protocol)."))
    parser.add_argument("--wandb-project",
                        default=os.environ.get("WANDB_PROJECT", "bending-graphs"))
    args = parser.parse_args()

    s = training_configs(
        data_mode=args.data_mode,
        mb_size=args.mb_size,
        epoch_size=args.epoch_size,
        alpha=args.alpha,
        m=args.m,
        learning_rate=args.learning_rate,
        dataset="smal",
        file_tag=args.file_tag,
        identifier=args.identifier or "",
    )
    s.target_shape = args.target_shape
    s.resume = args.resume
    s.no_eval = args.no_eval
    s.eval_on = args.eval_on
    s.wandb_project = args.wandb_project
    return s


def main() -> None:
    settings = _build_settings()
    print("==> Settings:", settings.to_string())

    smal_pkl = os.path.join(SMAL_DIR, "smal_CVPR2017.pkl")
    smal_data_pkl = os.path.join(SMAL_DIR, "smal_CVPR2017_data.pkl")
    if not (os.path.exists(smal_pkl) and os.path.exists(smal_data_pkl)):
        sys.exit(
            f"SMAL model not found under {SMAL_DIR}.\n"
            "  Set SMAL_DIR=... or run scripts/download_smal.sh after\n"
            "  registering at https://smal.is.tue.mpg.de/.\n"
            f"  Required files: {smal_pkl}, {smal_data_pkl}"
        )

    print(f"==> Building SMAL train/val dataset (target_shape={settings.target_shape})...")
    smal_train = SMALDataset(SMAL_DIR, train=True, mode=settings.data_mode,
                             target_shape=settings.target_shape)
    print(f"==> Train pairs: {len(smal_train)}")

    test_data = None
    if not settings.no_eval:
        if settings.eval_on == "tosca":
            if not os.path.isdir(TOSCA_DIR):
                sys.exit(
                    f"TOSCA root not found at {TOSCA_DIR}.\n"
                    "  Pass --eval-on smal (in-domain eval) or --no-eval, or set\n"
                    "  TOSCA_DIR=... and run scripts/download_tosca.sh after\n"
                    "  registering at http://tosca.cs.technion.ac.il/."
                )
            print(f"==> Building TOSCA test dataset (target_shape={settings.target_shape})...")
            test_data = ToscaDataset(
                root=TOSCA_DIR, train=False, mode=settings.data_mode,
                target_shape=settings.target_shape,
            )
        else:  # 'smal'
            print("==> Building SMAL test dataset...")
            test_data = SMALDataset(SMAL_DIR, train=False, mode=settings.data_mode,
                                    target_shape=settings.target_shape)
        print(f"==> Test pairs:  {len(test_data)}")

    train_mask, val_mask = create_validation_masks(smal_train, VALIDATION_SPLIT)
    train_loader = DataLoader(
        smal_train, num_workers=0, collate_fn=_passthrough_collate,
        batch_sampler=BatchSampler(
            MaskedSampler(len(smal_train), train_mask),
            batch_size=settings.mb_size, drop_last=False,
        ),
    )
    val_loader = DataLoader(
        smal_train, num_workers=0, collate_fn=_passthrough_collate,
        batch_sampler=BatchSampler(
            MaskedSampler(len(smal_train), val_mask),
            batch_size=settings.mb_size, drop_last=False,
        ),
    )

    wandb_logger.init(
        config={
            **{k: v for k, v in vars(settings).items() if not k.startswith("_")},
            "stack": "smal",
        },
        name=settings.to_string(),
        project=settings.wandb_project,
    )

    trained_path = settings.resume or get_model_path(False, settings)
    print(f"==> Trained checkpoint will be written to: {trained_path}")

    feat_trainer = trainer(
        settings,
        train_loader,
        val_loader,
        dataset_test=test_data,
        surface_type="deform",
        pretrain=False,
        model_path=trained_path if os.path.exists(trained_path) else None,
    )
    feat_trainer.train()
    wandb_logger.finish()


if __name__ == "__main__":
    main()
