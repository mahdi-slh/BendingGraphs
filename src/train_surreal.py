"""Train BendingGraphs on SURREAL → evaluate on MPI-FAUST.

Mirrors ``Graphite``'s ``train_modelnet.py`` style.  This is the
deformable-body pipeline from the *Bending Graphs* paper:

* train pairs are generated from the SURREAL parametric SMPL model;
* validation runs after each epoch on the MPI-FAUST registrations.

Outputs:
    models/trained_<tag>.model
    logs/<tag>/                  TensorBoard
    Weights & Biases (auto-disabled if ``wandb`` not configured)
"""
from __future__ import annotations

import argparse
import os
import sys

import configs
from configs import (
    MPIFAUST_DIR,
    SURREAL_DATA_DIR,
    SURREAL_MODEL_DIR,
    VALIDATION_SPLIT,
    training_configs,
)

from torch.utils.data import BatchSampler, DataLoader

import wandb_logger
from loaders.faust_syn_dataset import FaustSynDataset
from loaders.surreal_dataset import SURREALDataset
from utils.utils import (
    MaskedSampler,
    create_validation_masks,
    get_model_path,
)
from trainer import trainer


def _passthrough_collate(samples):
    """The dataset already yields ``(frame_data, meta)`` tuples whose
    leaves are heterogeneous (``Data`` lists, dicts, tensors).  PyG's
    default collator can't recurse through that structure, so for the
    deformable mb=1 pipeline we just hand the single sample back."""
    if len(samples) != 1:
        raise ValueError(
            f"_passthrough_collate expects mb_size=1; got {len(samples)} samples."
        )
    return samples[0]


def _build_settings() -> training_configs:
    parser = argparse.ArgumentParser(
        description="BendingGraphs SURREAL training (eval on FAUST)",
    )
    parser.add_argument("--data-mode", default="clean",
                        choices=["clean", "noise", "patch", "patchnoise"])
    parser.add_argument("--mb-size", dest="mb_size", type=int, default=1)
    parser.add_argument("--epoch-size", dest="epoch_size", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                        default=1e-4)
    parser.add_argument("--identifier", default="")
    parser.add_argument("--file-tag", dest="file_tag", default="surreal_faust")
    parser.add_argument("--resume", default="",
                        help="Path to a saved .model to resume from.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip per-epoch evaluation on FAUST.")
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
        dataset="surreal",
        file_tag=args.file_tag,
        identifier=args.identifier or "",
    )
    s.resume = args.resume
    s.no_eval = args.no_eval
    s.wandb_project = args.wandb_project
    return s


def main() -> None:
    settings = _build_settings()
    print("==> Settings:", settings.to_string())

    # ----- Sanity-check dataset roots ---------------------------------------
    if not os.path.isfile(os.path.join(SURREAL_DATA_DIR, "smpl_data.npz")):
        sys.exit(
            f"SURREAL ``smpl_data.npz`` not found under {SURREAL_DATA_DIR}.\n"
            "  Set SURREAL_DATA_DIR=... or run scripts/download_surreal.sh\n"
            "  after registering at https://www.di.ens.fr/willow/research/surreal/."
        )
    if not os.path.isdir(SURREAL_MODEL_DIR):
        sys.exit(
            f"SMPL model directory not found at {SURREAL_MODEL_DIR}.\n"
            "  Set SURREAL_MODEL_DIR=... and place the official\n"
            "  basicModel_*_lbs_10_207_0_v1.0.0.pkl files there\n"
            "  (download from https://smpl.is.tue.mpg.de/)."
        )
    if (not settings.no_eval) and not os.path.isdir(MPIFAUST_DIR):
        sys.exit(
            f"FAUST root not found at {MPIFAUST_DIR}.\n"
            "  Pass --no-eval to skip the per-epoch evaluation, or\n"
            "  set MPIFAUST_DIR=... / run scripts/download_faust.sh."
        )

    # ----- Datasets ---------------------------------------------------------
    print("==> Building SURREAL train dataset (preprocesses on first run)...")
    surreal_train = SURREALDataset(
        SURREAL_DATA_DIR, SURREAL_MODEL_DIR,
        train=True, mode=settings.data_mode,
    )
    print(f"==> Train pairs: {len(surreal_train)}")

    faust_test = None
    if not settings.no_eval:
        print("==> Building FAUST test dataset...")
        faust_test = FaustSynDataset(
            root=MPIFAUST_DIR, train=False, mode=settings.data_mode,
            frame_range=range(90, 99), sample_per_frame=1,
        )
        print(f"==> Test pairs:  {len(faust_test)}")

    train_mask, val_mask = create_validation_masks(surreal_train, VALIDATION_SPLIT)
    train_loader = DataLoader(
        surreal_train, num_workers=0, collate_fn=_passthrough_collate,
        batch_sampler=BatchSampler(
            MaskedSampler(len(surreal_train), train_mask),
            batch_size=settings.mb_size, drop_last=False,
        ),
    )
    val_loader = DataLoader(
        surreal_train, num_workers=0, collate_fn=_passthrough_collate,
        batch_sampler=BatchSampler(
            MaskedSampler(len(surreal_train), val_mask),
            batch_size=settings.mb_size, drop_last=False,
        ),
    )

    wandb_logger.init(
        config={
            **{k: v for k, v in vars(settings).items() if not k.startswith("_")},
            "stack": "surreal->faust",
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
        dataset_test=faust_test,
        surface_type="deform",
        pretrain=False,
        model_path=trained_path if os.path.exists(trained_path) else None,
    )
    feat_trainer.train()
    wandb_logger.finish()


if __name__ == "__main__":
    main()
