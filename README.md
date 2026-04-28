# BendingGraphs

Reference implementation of:

> **Bending Graphs: Hierarchical Shape Matching using Gated Optimal Transport**
> Mahdi Saleh, Shun-Cheng Wu, Luca Cosmo, Nassir Navab, Benjamin Busam, Federico Tombari
> *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.

A hierarchical shape-matching framework that pairs a Graphite-style
graph-induced descriptor extractor with a *Gated Optimal Transport* (GOT)
matching head. GOT itself is the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
middle-end re-purposed for graph-structured shape descriptors and run for
multiple Sinkhorn rounds with a confidence-gated GRU regulariser between
rounds (see [src/models/matching.py](src/models/matching.py) — the Magic
Leap copyright is preserved verbatim).

> **Note on relation to Graphite.** BendingGraphs reuses the Graphite GNN
> as its per-shape descriptor backbone but adds: (i) hierarchical patch
> aggregation, (ii) the GOT matching head, (iii) hard-negative sampling,
> and (iv) partial use of mesh structure (geodesic distances + Dijkstra
> seeds) where Graphite was point-cloud-only. The reference Graphite
> implementation lives at [github.com/mahdi-slh/Graphite](https://github.com/mahdi-slh/Graphite).
> Originally written for PyTorch 1.8 / CUDA 10.2 / Python 3.8; this fork
> brings it up to PyTorch 2.x / CUDA 11.8 / Python 3.10.

```
src/
├── train_surreal.py         train on SURREAL (eval on FAUST per epoch)
├── train_smal.py            train on SMAL    (eval on TOSCA per epoch)
├── evaluate_faust.py        FAUST geodesic-error eval
├── evaluate_tosca.py        TOSCA geodesic-error eval
├── smoke_test.py            sanity check on synthetic Open3D spheres
├── configs.py               paths + training_configs (Graphite-style)
├── _compat.py               Open3D / NumPy / SciPy / sklearn shims
├── wandb_logger.py          optional W&B logging
├── trainer.py               main training loop
├── models/
│   ├── model.py             Net + Graphite + NetMatch (Graphite + GOT)
│   ├── matching.py          GOT matcher (SuperGlue middle-end + Reg)
│   └── regularizer.py       GraphNodeGrad + WeightedGCN for GOT
├── loaders/
│   ├── faust_syn_dataset.py / smal_dataset.py / tosca_dataset.py
│   ├── surreal_dataset.py   SURREAL+SMPL pair generator
│   └── matching_dataset_base.py   patch/seed/Dijkstra plumbing
├── evaluation/
│   ├── evaluator.py         Evaluator base
│   ├── eval_rigid.py        EvaluatorRigid (rotation + Chamfer)
│   └── eval_deform.py       EvaluatorDeform (geodesic error + bij rate)
└── ...
scripts/
├── download_surreal.sh      SURREAL + SMPL helper
├── download_faust.sh        MPI-FAUST helper
├── download_smal.sh         SMAL parametric model helper
└── download_tosca.sh        TOSCA high-resolution helper
data/{surreal,MPI-FAUST,smal,tosca}/   datasets land here
models/                                trained checkpoints
logs/                                  per-tag training logs
outputs/                               last eval metrics
```

## 1. Install

```bash
conda create -y -n bending python=3.10
conda activate bending

pip install --no-cache-dir 'numpy<2' \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install --no-cache-dir torch-geometric==2.5.3

pip install --no-cache-dir 'open3d==0.18.0' 'opencv-python==4.10.0.84' \
    pyquaternion h5py trimesh tvb-gdist
pip install --no-cache-dir --no-build-isolation chumpy
```

`chumpy` ships SMPL/SMAL parametric model loaders. The
`--no-build-isolation` flag is required because `chumpy`'s `setup.py`
imports `numpy` at build time.

Sanity check:

```bash
python -c "import torch, torch_geometric, open3d, chumpy; print(torch.__version__, torch.cuda.is_available())"
```

The same wheel pinning constraint as Graphite applies — `torch-{scatter,
sparse,cluster}` only ship for specific torch / CUDA combos at
`data.pyg.org`. The `bending` env is otherwise self-contained.

If you already have a working `graphite` env, you can clone it and just
add the BendingGraphs-only extras:

```bash
conda create -y -n bending --clone graphite
conda activate bending
pip install --no-cache-dir trimesh tvb-gdist
pip install --no-cache-dir --no-build-isolation chumpy
```

### Smoke test (no datasets needed)

`smoke_test.py` builds two perturbed Open3D spheres in memory and runs
one forward + backward pass through the trainer plus one
`EvaluatorDeform.eval` round. It exercises the same pipeline as
SURREAL→FAUST and SMAL→TOSCA, minus the gated downloads:

```bash
python src/smoke_test.py
# ... SMOKE TEST PASSED
```

## 2. Datasets

All four datasets are gated and require manual registration. The
`scripts/download_*.sh` helpers create the expected directory layout
and (if you point them at a local archive via env var) extract it into
place.

### SURREAL + SMPL — train backbone for FAUST eval

* SURREAL: register at https://www.di.ens.fr/willow/research/surreal/data/
* SMPL: register at https://smpl.is.tue.mpg.de/ and download
  `SMPL_python_v.1.0.0.zip`.

```bash
SMPL_ZIP=/path/to/SMPL_python_v.1.0.0.zip \
SURREAL_USER=foo SURREAL_PASS=bar \
    ./scripts/download_surreal.sh
```

This populates `data/surreal/smpl_data/smpl_data.npz` and
`data/surreal/smpl_models/basic*model_*.pkl`.

### MPI-FAUST — eval target for SURREAL training

Register at http://faust.is.tue.mpg.de/ and download `MPI-FAUST.zip`.

```bash
FAUST_ZIP=/path/to/MPI-FAUST.zip ./scripts/download_faust.sh
```

### SMAL — train backbone for TOSCA eval

Register at https://smal.is.tue.mpg.de/ and download
`smal_online_V1.0.zip`.

```bash
SMAL_ZIP=/path/to/smal_online_V1.0.zip ./scripts/download_smal.sh
```

### TOSCA — eval target for SMAL training

Register at http://tosca.cs.technion.ac.il/ and download
`toscahires-mat.zip`.

```bash
TOSCA_ZIP=/path/to/toscahires-mat.zip ./scripts/download_tosca.sh
```

## 3. Train

The paper trains in two regimes:

| Backbone | Eval target | Entry point |
|----------|-------------|-------------|
| SURREAL (synthetic humans) | MPI-FAUST | `python src/train_surreal.py` |
| SMAL (parametric animals)  | TOSCA     | `python src/train_smal.py --target-shape horse` |

Both follow the same loop: per-epoch ``trainer`` runs all train pairs,
saves a checkpoint to `models/trained_<tag>.model`, then runs the
deformable evaluator (geodesic average error, bijective rate, Chamfer
distance, geodesic-after-OT) on the eval target.

```bash
conda activate bending

# Train on SURREAL, evaluate on FAUST after each epoch.
python src/train_surreal.py            # paper-protocol defaults
python src/train_surreal.py --no-eval  # skip per-epoch FAUST eval

# Train on SMAL, evaluate on TOSCA after each epoch (paper protocol).
python src/train_smal.py --target-shape horse
python src/train_smal.py --target-shape cat --eval-on smal   # in-domain eval
```

The first run lazily preprocesses each shape into patch graphs (FPS
seeds → Dijkstra patches → KNN graph) and caches the result under
`data/<dataset>/processed/`; subsequent runs skip preprocessing.

### CLI flags (most useful)

| Flag | Default | Notes |
|------|---------|-------|
| `--epoch-size` | 50 | Number of epochs |
| `--mb-size` | 1 | Per-pair training; each pair already produces ~64 patches |
| `--learning-rate` | 1e-4 | Optimizer base LR |
| `--alpha` | 10.0 | Weight on the descriptor (triplet) loss |
| `--m` | 1.0 | Triplet margin |
| `--data-mode` | `clean` | `clean` / `noise` / `patch` / `patchnoise` |
| `--target-shape` | `horse` | (SMAL only) `cat` / `dog` / `horse` / `cow` / `hippos` / `random` |
| `--eval-on` | `tosca` | (SMAL only) eval target — `tosca` (paper) or `smal` |
| `--no-eval` | – | Skip per-epoch evaluation |
| `--resume PATH` | – | Continue from a saved `.model` |
| `--file-tag` | `surreal_faust` / `smal` | Used in checkpoint and log filenames |

### Optional: Weights & Biases tracking

Training logs every per-epoch loss and deformable-eval metric to W&B if
it's configured. The default project is `bending-graphs`.

```bash
pip install wandb && wandb login
python src/train_surreal.py
WANDB_PROJECT=my-project python src/train_smal.py
WANDB_MODE=disabled    python src/train_smal.py   # silence
```

`wandb_logger.py` is a no-op if `wandb` isn't installed.

### Useful environment variables

| Variable | Effect |
|----------|--------|
| `BENDING_DATA_ROOT` / `BENDING_MODELS_ROOT` / `BENDING_LOG_ROOT` | Override default `data/`, `models/`, `logs/` |
| `BENDING_OUTPUTS_DIR` | Where `metrics.txt` is written (default `outputs/`) |
| `MPIFAUST_DIR` / `SMAL_DIR` / `TOSCA_DIR` | Per-dataset roots |
| `SURREAL_DATA_DIR` / `SURREAL_MODEL_DIR` | SURREAL `smpl_data.npz` and SMPL `.pkl` roots |
| `WANDB_PROJECT` / `WANDB_MODE` | Standard W&B knobs |

## 4. Evaluation

Two benchmarks are wired up. Both load a checkpoint produced by
`train_surreal.py` or `train_smal.py`.

### 4.1 MPI-FAUST geodesic-error eval

```bash
python src/evaluate_faust.py \
    --model models/trained_surreal_faust_a10.0m1.0lr0.0001f64r7.model
```

Reports bijective rate, geodesic average (raw), geodesic average after
one OT round, number of valid matches, and Chamfer distance on the
matched seeds. Pass `--max-pairs N` to evaluate only the first `N`
pairs, or `--visualize` to pop up Open3D windows (requires `DISPLAY`).

### 4.2 TOSCA cross-dataset eval

```bash
python src/evaluate_tosca.py \
    --model models/trained_smal_a10.0m1.0lr0.0001f64r7.model \
    --target-shape horse
```

Same metrics; `--target-shape` selects the species
(`cat` / `centaur` / `david` / `dog` / `gorilla` / `horse` / `michael` / `victoria` / `wolf`).

## 5. Loading a checkpoint manually

```python
import configs
from configs import training_configs
from models.model import init_model

settings = training_configs(file_tag='surreal_faust',
                            data_mode='clean')
model, optimizer, epoch = init_model(
    'models/trained_surreal_faust_a10.0m1.0lr0.0001f64r7.model', settings)
model.eval()
```

## 6. Licensing

* The matching block in `src/models/matching.py` is the **SuperGlue**
  middle-end from the Magic Leap reference implementation, modified to
  consume graph-structured shape descriptors and to perform
  Gated Optimal Transport rounds. The Magic Leap copyright header is
  preserved verbatim at the top of that file. SuperGlue is released for
  *non-commercial use only* under the Magic Leap [LICENSE](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE);
  any commercial use of BendingGraphs needs to clear that licence first.
* Everything else in this repository is © the BendingGraphs authors and
  shipped as-is.

If you use this codebase, please cite the Bending Graphs paper *and* the
SuperGlue paper for the matching block.
