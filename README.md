<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>AdaPixFlow: Learning Continuous-Time 3D Dynamics Across Scenes from Sparse Multi-view Observations</h1>
      <p><a href="https://github.com/HrishikeshVish">Hrishikesh Viswanath</a> &emsp; Aniket Bera</p>
      <p>Department of Computer Science, Purdue University</p>
      <p><a href="https://arxiv.org/abs/2611.26433">arXiv</a> &emsp; <a href="https://hrishikeshvish.github.io/adapixflow/">Project page</a></p>
    </summary>
  </ul>
</div>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-598BE7?style=for-the-badge&logo=python&logoColor=598BE7&labelColor=F0F0F0"/></a> &emsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=EE4C2C&labelColor=F0F0F0"/></a> &emsp;
<a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=76B900&labelColor=F0F0F0"/></a> &emsp;
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge&labelColor=F0F0F0"/></a>

<br><br>
<img src="assets/hero.webp" width="850px" alt="AdaPixFlow advecting object pixels on a held-out scene"/>
<br>
<em>A held-out scene (PlasticineFall): the object pixels of an observation move along the predicted velocity field to a step between observations.</em>
<br><br>

<div id="toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h2><a href="#overview">Overview</a> &emsp; <a href="#data">Data</a> &emsp; <a href="#reproducing-the-paper">Reproducing the Paper</a> &emsp; <a href="#training">Training</a> &emsp; <a href="#evaluation">Evaluation</a> &emsp; <a href="#baselines">Baselines</a> &emsp; <a href="#citation">Citation</a></h2>
    </summary>
  </ul>
</div>

</div>

---

## Overview

Predicting how objects move requires recovering their 3D motion from visual observations and predicting their positions at times we never observed, both between sparse observations and after the last one. Existing methods either optimize a velocity field for each scene or impose a predefined physical or kinematic model.

**AdaPixFlow** learns a continuous-time dynamics operator across scenes. We back-project the pixels of all calibrated RGB-D cameras and fuse them into one voxel grid in world coordinates. From the fused observations around a gap, a 3D U-Net predicts a continuous-time velocity field, and we advect the object pixels along it to any target time:

$$\frac{d\mathbf{x}(t)}{dt} = \mathbf{v}_\theta(\mathbf{x}(t), t \mid \mathbf{c}), \qquad \mathbf{c} = (\mathbf{z}_{k_0}, \mathbf{z}_{k_p}, \mathbf{z}_{k_1}, e, s).$$

The operator is trained once, through the advection, by comparing the advected points with the observed object points as unordered sets. It needs no velocity labels, point correspondences, or per-scene optimization.

### Highlights

- **One model for in-between and future prediction.** The context holds the observations on both sides of a gap when they exist, and the past ones otherwise. One forward pass gives the velocity at all 13 integer times of a pass.
- **Closed-form fusion.** The voxel grid is computed directly from depth, masks and the known cameras, so the network only learns how the fused state moves.
- **Point-set supervision.** A Chamfer loss normalized by the error of a static prediction, plus local distance preservation. Neither needs tracks.
- **One command per result.** The configs in `config/` are the exact settings behind the paper's 3D results on PhysInOne and on the three physics benchmarks, and the released model reproduces them.

---

## Results

<div align="center">
<img src="assets/teaser.jpg" width="850px" alt="AdaPixFlow on a held-out scene against the compared methods"/>
<br>
<em>Left: observed states and our pathlines. Right: predicted 3D object points at a step between observations, against the compared methods, shaded by distance to the ground truth.</em>
</div>

On the six PhysInOne comparison scenes (Tab. 1 of the paper; F1 in %, with views 0 and 1 held out of every input):

| Method | Input | In-between | Future |
| :--- | :--- | :-: | :-: |
| FreeGave | RGB, D<sub>0</sub>, M | 69.1 | 31.5 |
| TRACE | RGB, D<sub>0</sub>, M | 29.9 | 28.4 |
| Shape of Motion | RGB-D, M, 1 view | 67.3 | 31.2 |
| MoSca | RGB-D, M, 1 view | 43.3 | 28.6 |
| SpatialTrackerV2 | RGB-D, M | 87.8 | 50.7 |
| OpenD4RT | RGB-D, M | 84.0 | 45.9 |
| **AdaPixFlow** | RGB-D, M | **90.0** | **58.4** |

Zero-shot on three physics benchmarks (Tab. 2; F1 in %, pooled over the targets of the scenes every method was run on):

| Benchmark | Linear | Non-linear | Future, 4–10 steps |
| :--- | :-: | :-: | :-: |
| Spring-Gaus | 95.7 | 72.3 | 52.7 |
| PhysGaia | 98.3 | 61.4 | 62.2 |
| Kubric-4D | 98.7 | 91.6 | 79.5 |

On the held-out Test (15) and Val (8) scenes (Tab. 3; F1 in %, Chamfer distance in mm):

| Scenes | s = 2 | s = 4 | s = 8 | Future (10 steps) |
| :--- | :-: | :-: | :-: | :-: |
| Test | 90.9 / 10.4 | 84.2 / 15.8 | 78.7 / 24.0 | 68.3 / 41.5 |
| Val | 99.9 / 4.3 | 97.6 / 5.2 | 90.8 / 9.2 | 55.8 / 42.3 |

---

## Method

<div align="center">
<img src="assets/pipeline.jpg" width="850px" alt="AdaPixFlow pipeline"/>
</div>

1. **Volumetric fusion** (`adapixflow/fusion.py`). The voxel grid covers the region in which the objects of a clip move, at 1.25 cm. Each voxel stores color, objectness, occupancy, and an unseen flag. Occupancy rounds each point to its nearest voxel, color and objectness are trilinear averages, and a voxel is unseen when every camera sees it behind its observed depth or not at all.
2. **Dynamics operator** (`adapixflow/model.py`). A 3D U-Net (widths 32, 64, 96, 128, LayerNorm per voxel) maps the three fused states, the flag $e$ and the spacing $s$ to 13 node fields $V_0, \dots, V_{12}$. The velocity is linear in time between nodes and trilinear in space.
3. **Advection** (`adapixflow/flow.py`). A midpoint scheme with both stages at the substep's mid-time integrates the field forward or backward. In-between targets are advected from the nearer observation, future targets from the last one.
4. **Training** (`train.py`, `adapixflow/training.py`). We train on crops of 48³ voxels, over the spacings and horizons we evaluate plus random ones, with a horizon curriculum and a redraw of windows in which too few voxels move.

---

## Repository Structure

```
adapixflow/
├── train.py                      # training entry point (one YAML config per run)
├── evaluate.py                   # held-out (Tab. 3) and comparison-protocol (Tab. 1, 2, 9, 10) evaluation
├── evaluate_points.py            # scores another method's predicted points with the same protocol
├── adapixflow/
│   ├── fusion.py                 # volumetric fusion of multi-view RGB-D, sparse fused scenes
│   ├── model.py                  # the velocity U-Net and checkpoint loading
│   ├── flow.py                   # node fields, tiled inference, midpoint advection
│   ├── training.py, losses.py    # the batch loss: relative Chamfer + distance preservation
│   ├── data.py                   # splits, moving voxels, evaluation windows, training sampler
│   ├── evaluation.py, metrics.py # the evaluation protocols, motion split, F1 and Chamfer
│   └── config.py                 # every config key and its default
├── dataset/
│   ├── fetch_physinone.py, fuse.py   # PhysInOne download and conversion, fusion CLI
│   ├── benchmarks/               # Spring-Gaus, PhysGaia and Kubric-4D converters
│   └── estimated/                # Depth Anything 3 depth, SAM 2.1 masks, and their fusion
├── config/                       # physinone/, springgaus/, physgaia/, kubric4d/: the paper's settings
├── splits/                       # scenes of every split, motion-split labels, estimated-input depth limits
├── scripts/                      # download / prepare / train / evaluate
├── third_party/                  # pinned submodules (Depth Anything 3, baselines) and our patches
└── tests/                        # unit checks (fusion, advection, model, losses)
```
---

## Installation

Tested on Ubuntu with Python 3.12, PyTorch 2.11 and CUDA 12.8. Install PyTorch first, since its wheel must match your CUDA version.

```bash
git clone https://github.com/HrishikeshVish/adapixflow.git
cd adapixflow
conda create -n adapixflow python=3.12 -y
conda activate adapixflow
pip install torch --index-url https://download.pytorch.org/whl/cu128   # replace cu128 with your CUDA version
pip install -r requirements.txt
python tests/test_core.py                                              # check the install (CPU, a few seconds)
```

**Pretrained model.** Download `adapixflow_physinone.pt` (12 MB) from the [releases page](https://github.com/HrishikeshVish/adapixflow/releases) into `checkpoints/`. It is the model behind every result in the paper.

---

## Data

All data goes under `data/`. Configs use repo-relative paths, so run every command from the repository root.

| Dataset | Scenes | Get it | Prepare |
| :--- | :--- | :--- | :--- |
| PhysInOne | 152 training, 15 Test, 8 Val (6 of them comparison scenes) | `bash scripts/download_physinone.sh` | `bash scripts/prepare_physinone.sh` |
| [Spring-Gaus](https://github.com/Colmar-zlicheng/Spring-Gaus) | 7 `mpm_synthetic` scenes | `bash scripts/download_benchmarks.sh springgaus` | `bash scripts/prepare_benchmarks.sh springgaus` |
| [PhysGaia](https://huggingface.co/datasets/mijeongkim/PhysGaia) | `visco_bouncing_balls` | `bash scripts/download_benchmarks.sh physgaia` | `bash scripts/prepare_benchmarks.sh physgaia` |
| [Kubric-4D](https://gcd.cs.columbia.edu/) | 6 GCD scenes | `bash scripts/download_benchmarks.sh kubric4d` | `bash scripts/prepare_benchmarks.sh kubric4d` |

`download_physinone.sh` fetches the 175 scenes listed in `splits/physinone_sources.json` from Hugging Face (121 GB of zips). It converts each one to a 560 px cache (48 GB in total) and deletes the zip. `prepare_physinone.sh` then fuses every scene into `data/physinone/fused` (9 GB). It also fuses the six comparison scenes without views 0 and 1 into `data/physinone/fused_holdout`, on the same grids.

The benchmark converters render depth and masks from the released particles (Spring-Gaus, PhysGaia) or read them from the release (Kubric-4D), and write the PhysInOne cache layout. Spring-Gaus and Kubric-4D frames are written twice, so that one step of the model is one native frame. Spring-Gaus `mpm_synthetic.zip` comes from the authors' Google Drive folder, which the script names if the zip is missing.

<details>
<summary><b>Notes on the splits</b></summary>

- We selected the training pool from PhysInOne's Train release, balancing phenomena within five tiers: A continuum, B1/B2/C rigid mechanics with one, two or three phenomena, D invisible forces. Granular and light phenomena are excluded. We held out 10% of each tier (seed 0), and trained on the 152 scenes that had been fused when training started.
- The Val scenes are ElasticFall and PlasticineFall on backgrounds 901–904 from `vLAR/PhysInOne` (`Val/SinglePhysics`). Neither their backgrounds nor PlasticineFall appear in training.
- The comparison scenes are DropInLiquid, CrankPush, PanelSpin, DoublePendulum and BalloonLift (five Test scenes), plus ElasticFall (a Val scene).

</details>

---

## Reproducing the Paper

With the pretrained model and the prepared data, each command prints the numbers of one table and writes per-scene and per-step results to `experiments/`. `evaluate.py <config>` runs the same thing directly.

| Result | Data | Command |
| :--- | :--- | :--- |
| Tab. 1, comparison scenes, and Tab. 9, motion split | PhysInOne | `bash scripts/evaluate.sh physinone/comparison` |
| Tab. 2 and Tab. 10, physics benchmarks | Spring-Gaus | `bash scripts/evaluate.sh springgaus/comparison` |
| | PhysGaia | `bash scripts/evaluate.sh physgaia/comparison` |
| | Kubric-4D | `bash scripts/evaluate.sh kubric4d/comparison` |
| Tab. 3, held-out Test and Val scenes | PhysInOne | `bash scripts/evaluate.sh physinone/heldout` |
| Estimated depth and masks (78.5 / 53.3) | PhysInOne + `prepare_estimated.sh` | `bash scripts/evaluate.sh physinone/estimated` |

For the benchmarks, Tab. 2 pools the motion subsets over the scenes every method was run on (`common` in `splits/<benchmark>.json`), and Tab. 10 lists the per-scene F1. The motion subsets are fixed by the ground truth alone, from the F1 that an oracle linear interpolation of the ground-truth states reaches on each target (`splits/motion_split_*.json`).

<details>
<summary><b>Estimated inputs</b></summary>

The estimated-input result replaces the dataset depth and masks of the comparison scenes with estimates from the RGB of the observed views: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (giant, posed with our cameras) and [SAM 2.1](https://github.com/facebookresearch/sam2) (`facebook/sam2.1-hiera-large`, prompted automatically from color and depth changes). Depth Anything 3 is a pinned submodule. SAM 2.1 runs through its `transformers` implementation.

```bash
git submodule update --init third_party/Depth-Anything-3
pip install -e third_party/Depth-Anything-3          # in its own environment (DA3_PYTHON below)
pip install transformers==5.17.0                     # in the AdaPixFlow environment, for SAM 2.1
DA3_PYTHON=/path/to/da3/env/bin/python bash scripts/prepare_estimated.sh 0
bash scripts/evaluate.sh physinone/estimated
```

`prepare_estimated.sh` runs both models on the training views only, and fuses their outputs on the grids of the dataset fusion. In-between and future prediction get separate fusions, since the masks see only that task's observations.

</details>

---

## Training

```bash
bash scripts/train.sh config/physinone/train.yaml 0     # -> experiments/physinone/
```

`train.sh <config> [gpu]` runs `python train.py <config>`. Training takes about 4 h on one RTX 5080, including an evaluation every 3,000 steps on a subset of the held-out scenes. The final evaluation at step 24,000 covers every Test and Val scene, and `--resume` continues from `last.pt`. Set `logging.wandb: true` to log to Weights & Biases, and `memory_fraction` to share a GPU. `adapixflow/config.py` lists every key with its default.

---

## Evaluation

| Protocol | Config | Input | Ground truth | Window | Chamfer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Comparison scenes and benchmarks (Tab. 1, 2, 9, 10) | `*/comparison.yaml` | fusion without views 0 and 1 | fusion of all views | whole moving region (128³ tiles) | median |
| Held-out scenes (Tab. 3) | `physinone/heldout.yaml` | fusion of all views | the same | moving region, capped at 128 voxels | mean |

Both protocols score F1 at 1 and 2 voxels (the paper reports 2 voxels, 2.5 cm). They score only the target steps at which the ground-truth object moved by more than one voxel since the observation before the gap (in-between) or the last observation (future). Results are averaged over the scored steps of a scene, then over scenes. On a GPU, the forward pass runs in bfloat16 as in the paper. On a CPU, it runs in float32, which changes F1 by about 0.1 point.

---

## Baselines

The compared methods are pinned submodules in `third_party/`, at the commits we ran. Pull the ones you need:

```bash
git submodule update --init third_party/FreeGave        # or TRACE, shape-of-motion, MoSca, SpaTrackerV2, Open-d4rt
git -C third_party/FreeGave apply ../patches/FreeGave.patch   # FreeGave, TRACE, shape-of-motion and MoSca only
```

| Method | Repository | Our patch |
| :--- | :--- | :--- |
| FreeGave | [vLAR-group/FreeGave](https://github.com/vLAR-group/FreeGave) | missing `<cstdint>` / `<cfloat>` includes (build fix) |
| TRACE | [vLAR-group/TRACE](https://github.com/vLAR-group/TRACE) | the same build fix |
| Shape of Motion | [vye16/shape-of-motion](https://github.com/vye16/shape-of-motion) | scikit-learn fallback for cuML, gsplat ≥ 1.5 radii, opt-in masking of invalid depth |
| MoSca | [JiahuiLei/MoSca](https://github.com/JiahuiLei/MoSca) | build fixes, NumPy 2, the `camera.py` symlink |
| SpatialTrackerV2 | [henry123-boy/SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2) | none |
| OpenD4RT | [Lijiaxin0111/Open-d4rt](https://github.com/Lijiaxin0111/Open-d4rt) (unofficial D4RT) | none |

The appendix of the paper lists the inputs and settings we gave each method (tables *Per-scene dynamics baselines* and *Adapting 4D methods*). `evaluate_points.py` scores any method's predicted object points with the protocol of Tab. 1 and 2, so that a prediction file in its documented format is scored exactly as AdaPixFlow is:

```bash
python evaluate_points.py config/physinone/comparison.yaml --pred <dir>   # <dir>/<task>/<scene>/pred.npz
```

---

## Citation

```bibtex
@misc{viswanath2026adapixflowlearningcontinuoustime3d,
  title={AdaPixFlow: Learning Continuous-Time 3D Dynamics Across Scenes from Sparse Multi-view Observations},
  author={Hrishikesh Viswanath and Aniket Bera},
  year={2026},
  eprint={2611.26433},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.26433},
}
```

## Acknowledgments

We thank the authors of [PhysInOne](https://huggingface.co/datasets/vLAR/PhysInOne), [Spring-Gaus](https://github.com/Colmar-zlicheng/Spring-Gaus), [PhysGaia](https://huggingface.co/datasets/mijeongkim/PhysGaia) and [Kubric-4D](https://gcd.cs.columbia.edu/) for their data. Each dataset is downloaded from its own release, under its own license (PhysGaia: CC BY-NC 4.0).

## License

MIT. See [LICENSE](LICENSE).
