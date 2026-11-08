# Prune, Quantize, or Both? Honest Accuracy–Size Curves for a Vision Transformer

*[Versión en español](README.es.md)*

## Summary

Compression of Vision Transformers is usually reported with nominal figures, such as bits per weight or parameter counts, not with the real size of the model. Each published method uses its own protocol (a different number of calibration images, a different full-precision reference accuracy), so their results are not directly comparable. They are rarely compared with a smaller uncompressed model, and almost never evaluated beyond ImageNet-val, for example on new photos of the same classes (ImageNet-V2) or on other visual styles (ImageNet-R).

For DeiT-Small on ImageNet-1k, we ask in which ranges of the accuracy–size curve it pays to prune, to quantize, or to combine both, and which components of the model explain the difference.

To answer with a common measurement, every configuration is placed on the same axes by its real size in bytes, with paired-bootstrap confidence intervals. It sits next to DeiT-Tiny and to published post-training compression methods, all run with their authors' code under one protocol. We also test whether the conclusions hold on ImageNet-V2 and ImageNet-R.

On that basis we propose a post-training procedure. It measures how sensitive each block's attention and MLP are to structured pruning and to weight quantization. It removes attention heads and MLP neurons, compensating the next layer by least squares, then quantizes the remaining weights, and allocates these decisions per block to reach a series of target sizes. Restricting it to quantization only, or to pruning only, gives curves that are directly comparable with each other.

## How to navigate

The sections below follow the notebooks in order: the two reference models, the dense baselines saved as real files, the published quantization methods, the published pruning methods, our allocation with its three curves, and the activations. Each notebook holds the full tables and charts of its section, and the decisions behind the protocol are listed, with their reasons, in [`DECISIONS.md`](DECISIONS.md).

| Path | Role |
|---|---|
| `01_rebuild_deit.ipynb` | DeiT rebuilt from scratch and checked against timm |
| `02_dense_baselines.ipynb` | DeiT-S and DeiT-Ti at FP32, FP16, INT8 and INT4 as real files |
| `03_known_methods.ipynb` | the published quantization methods, as published and weights-only, against their papers |
| `04_known_pruning.ipynb` | the published pruning methods, without retraining |
| `05_fragility.ipynb` | which part of FIMA-Q is fragile, against the predictions in `ABLATIONS.md` |
| `06_prune_quantize_curves.ipynb` | pruned models at low bits, the three curves of our allocation, and the costs besides size |
| `deitc/` | the code the notebooks and scripts share: model, data, packed files, scoring and result files, the readers for published checkpoints, costs, pruning and rounding |
| `scripts/setup_third_party.sh` | the published methods' code, cloned at the commits we used, with our patches |
| `scripts/data/prepare_imagenet.py`, `scripts/data/make_calib_list.py` | the data, and the calibration images shared by every method (`splits/calib_seed0.txt`) |
| `scripts/quantize/pack_published.py`, `scripts/quantize/evaluate_file.py` | from a published method's checkpoint to a checked file, and from a file to its scores |
| `methods/adalog/` | our patch to the AdaLog code and the script that runs it |
| `methods/aphq/`, `methods/fimaq/`, `methods/repq/`, `methods/ptq4vit/` | the same for APHQ-ViT, FIMA-Q, RepQ-ViT and PTQ4ViT, and FIMA-Q's activation tests |
| `deitc/pruning.py`, `scripts/prune/prune_points.py`, `scripts/prune/combine_points.py` | pruning without retraining (our CORP reimplementation, GRAIL and the pruning stage of VBP), and pruned models taken to low bits |
| `scripts/allocate/sensitivity.py`, `scripts/allocate/allocate.py` | the sensitivity of each layer and block, and our per-block allocation under a target size |
| `scripts/measure/measure_costs.py`, `scripts/measure/measure_latency.py` | MACs, BOPs, latency and memory |
| `scripts/quantize/w8a8_torchao.py`, `scripts/quantize/a8_score.py`, `scripts/measure/gemlite_test.py`, `scripts/prune/quantize_then_prune.py` | the INT8 files, 8-bit activations on packed models, the GemLite kernels and the check of compression order |
| `ABLATIONS.md` | our predictions for the ablations and the activation tests, written before they ran |
| `outputs/*_summary.json` | the numbers in this page |

## 1. Rebuilding DeiT-S

DeiT-S is re-implemented in plain PyTorch, with the attention written out so that whole heads can be removed, and loaded with the official weights. The run in the notebook used an RTX 4070, torch 2.11 and timm 1.0.30.

| Check | Result |
|---|---|
| Parameters | 22,050,664; the official `state_dict` loads with `strict=True` |
| Same function as timm | identical predictions on all 10,000 ImageNet-V2 images |
| ImageNet-V2 top-1, DeiT-S | 68.52 % [67.63, 69.48], published 68.53 % |
| ImageNet-V2 top-1, DeiT-Ti | 59.81 % [58.86, 60.84], published 59.92 % |
| Where the weights are | 96.5 % in the block linears (MLP 64.3 %, attention 32.2 %) |
| Size overhead over params × bits / 8 | +4 % at INT8, +23 % at INT4, +32 % at INT3, from scales, zero-points and FP16 layers |

Brackets are 95 % bootstrap intervals. The INT sizes in notebook 01 are computed; from notebook 02 on, every size is that of a file on disk.

## 2. The dense baselines as real files

Each model is saved in each format, read back, checked bit for bit against what was written, and evaluated from the file. The quantizer is plain round-to-nearest on the weights of the 48 block linears, in groups of 64 for INT4, with the rest in FP16. It uses no data, so it is the floor every published method should improve on.

| Model | File (MB) | ImageNet-1k val | ImageNet-V2 | ImageNet-R |
|---|---|---|---|---|
| DeiT-S FP32 | 88.2 | 79.86 | 68.52 | 42.24 |
| DeiT-S FP16 | 44.1 | 79.86 | 68.53 | 42.25 |
| DeiT-S INT8 | 23.0 | 79.84 | 68.41 | 42.27 |
| DeiT-S INT4 | 13.6 | 79.18 | 67.76 | 41.68 |
| DeiT-Ti FP32 | 22.9 | 72.17 | 59.81 | 32.70 |
| DeiT-Ti FP16 | 11.4 | 72.17 | 59.80 | 32.70 |
| DeiT-Ti INT8 | 6.2 | 72.15 | 59.77 | 32.84 |
| DeiT-Ti INT4 | 3.8 | 70.43 | 57.87 | 31.53 |

Top-1 in %. ImageNet-R is scored over its 200 classes, as in Hendrycks et al. (2021). DeiT-S at INT4 is 2.2 MB larger than DeiT-Ti at FP16 and 7 points more accurate on ImageNet-1k val, which is why DeiT-Ti stays on every curve.

## 3. The published quantization methods

Each published method runs with its authors' code and settings, so that the point on the chart is their method and not our version of it. It runs twice, as published, with weights and activations quantized at the bit widths its paper reports, and weights-only, with activations at full precision, since only the weights set the size of the file. Between methods only two things change, the calibration images, which are the first N lines of `splits/calib_seed0.txt` where N is the number the authors use, and the three test sets. A weights-only run is scored from a real file, built from the method's own codes and grid, read back and checked bit for bit against the weights the method's code uses. A run as published is scored inside the method's own code on the same three sets.

| Weights-only on DeiT-S | File (MB) | ImageNet-1k val | ImageNet-V2 | ImageNet-R |
|---|---|---|---|---|
| FIMA-Q W4 | 11.5 | 79.54 | 68.23 | 41.56 |
| APHQ-ViT W4 | 11.5 | 79.16 | 67.71 | 41.16 |
| AdaLog W4 | 11.5 | 78.87 | 66.71 | 40.62 |
| RepQ-ViT W4 | 11.5 | 77.34 | 65.42 | 38.25 |
| FIMA-Q W3 | 8.8 | 78.66 | 66.96 | 39.91 |
| APHQ-ViT W3 | 8.8 | 77.96 | 66.33 | 38.97 |
| AdaLog W3 | 8.8 | 74.18 | 61.56 | 37.53 |
| FIMA-Q W2 | 6.0 | 73.10 | 60.42 | 33.53 |

FIMA-Q W2 is not a published setting, it is its W3 configuration with 2-bit weights. APHQ-ViT keeps its MLP reconstruction, which swaps GELU for ReLU, so its files are scored with ReLU. PTQ4ViT, whose lowest published width is 6 bits, and the W6 points of the others are in notebook 03. Run as published at W4A4, the same methods lose between 2.4 points (APHQ-ViT) and 8.2 points (RepQ-ViT) on val against their weights-only runs, which is the share of their loss that comes from the activations. The paired differences against FP32 and their intervals are in `outputs/published_summary.json`.

## 4. Pruning without retraining

CORP, GRAIL and the pruning stage of VBP run without retraining. GRAIL and VBP run from their authors' code, and CORP from our reimplementation of its paper, since it has no public code. Each pruned model is saved as a real FP16 and FP32 file, 63 in all. CORP pruning attention and MLP at 10 to 50 % gives 78.82, 77.46, 75.27, 72.26 and 57.07 on val, against 78.48, 76.78, 74.53, 70.84 and 58.37 in its paper, and VBP at 50 % gives 64.60 against the 64.44 its paper reports before fine-tuning. GRAIL's paper has no DeiT numbers to compare with.

No pruned model in FP16 beats an unpruned model of the same size or smaller. CORP at 10 % takes 40.6 MB for 78.82 on val, while DeiT-S INT8 takes 23.0 MB for 79.84.

## 5. Our allocation and the three curves

Our procedure measures how much each layer's bits and each block's pruning change the output on calibration images, one at a time with the rest in FP32 (`scripts/allocate/sensitivity.py`). `scripts/allocate/allocate.py` turns that map into a plan for a target size, and every plan is built and scored as a whole model, FIMA-Q with per-layer bits for the bits-only curve, CORP in FP16 for the pruning-only curve, and CORP then FIMA-Q for the combined one.

Every claim compares two models on the same images and has to hold on ImageNet-V2 and ImageNet-R as well. Quantization alone keeps 79.75 % on val at 14.0 MB, against 79.86 % for FP32 at 88.2 MB. Adding pruning to quantization brings no gain between 6.7 and 10 MB, and at 7.4 MB and below it loses, by up to 1.8 points on val. Near 6 MB, DeiT-Ti INT8 (72.15 % at 6.2 MB) catches up with DeiT-S. Notebook 06 also checks the other compression order and reports MACs, BOPs, latency, memory and time to compress for every point.

## 6. Activations

With FIMA-Q, quantizing the activations costs little at 8 bits but 2.8 points on val at 4 bits and 8.6 at 3. Pruning 10 % with CORP before FIMA-Q, run as published, takes a further 1.1 points at W4A4 and 1.4 at W3A3. On twelve of our models, rounding the inputs of the block linears to 8 bits, simulated in floating point, moves top-1 by at most 0.16 points on any of the three sets. Notebook 05 tests which of FIMA-Q's activations break first, returning one point at a time to full precision, and compares the results with the predictions written in `ABLATIONS.md` before any test ran.

## Third-party code

The published methods are not part of this repository, since each has its own licence. `bash scripts/setup_third_party.sh` clones each one next to this folder, in `../third_party/`, at the commit we used, and applies our patch from `methods/<name>/` where there is one, under the folder names that `methods/*/run.sh` and `deitc/pruning.py` look for. VBP is under AGPL-3.0, so our code only calls it from that folder and none of it is copied here; check each repository's licence before redistributing anything.

| Method | Repository | Commit |
|---|---|---|
| AdaLog | https://github.com/GoatWu/AdaLog | 75a57d2 |
| APHQ-ViT | https://github.com/GoatWu/APHQ-ViT | 84c9fde |
| FIMA-Q | https://github.com/ShiheWang/FIMA-Q | 64c4b00 |
| RepQ-ViT | https://github.com/zkkli/RepQ-ViT | a160561 |
| PTQ4ViT | https://github.com/hahnyuan/PTQ4ViT | b42c790 |
| GRAIL | https://github.com/TWWinde/GRAIL_Compensation | ba5e48b |
| VBP (pruning stage only) | https://github.com/boschresearch/variance-based-pruning | b6834b4 |
| GemLite (kernels for one check) | https://github.com/dropbox/gemlite | f649e80 |

CORP has no public code; `deitc/pruning.py` is our reimplementation from its paper.

## Running it

```bash
# Python 3.10 and a CUDA GPU; we used torch 2.11 and timm 1.0.30
pip install torch torchvision timm torchao safetensors scipy matplotlib jupyter huggingface_hub pyarrow
```

The published methods need older versions, so they run in a second environment with Python 3.10, torch 2.2.2 and timm 0.9.2, plus what each repository lists. Its interpreter goes in `ADALOG_PY`, and `PY` points to the first environment for the steps where a method's script calls ours.

ImageNet-V2 matched-frequency is public ([Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2)), and ImageNet-R is available from its authors. ImageNet-1k is gated on Hugging Face; once its terms are accepted, `python scripts/data/prepare_imagenet.py --out ../data/imagenet` writes the 50,000 validation images and a pool of training images for calibration. The notebooks read the three sets from `../data/` or from `IMAGENET_VAL_DIR`, `IMAGENETV2_DIR` and `IMAGENETR_DIR`.

```bash
jupyter nbconvert --to notebook --execute 01_rebuild_deit.ipynb    # about a minute on an RTX 4070
jupyter nbconvert --to notebook --execute 02_dense_baselines.ipynb
python scripts/data/make_calib_list.py --train ../data/imagenet/train --seed 0 # the shared calibration list
bash scripts/setup_third_party.sh                                 # the published methods' code
```

Notebooks 03 to 06 read the results in `outputs/`, so they show every table and chart from a fresh clone without running anything on the GPU. Setting `RERUN = True` at the top of one makes it call the scripts that produced those results instead, which takes hours, since each FIMA-Q or APHQ-ViT search runs from 40 minutes to 2 hours on our RTX 4070 with 12 GB. On that card two searches at once finish no sooner than one after the other, so run one at a time, which also keeps each time to compress valid. The latency in notebook 06 was measured on that card with nothing else running, and the low-bit kernels we tried were not built for it, so those timings describe this GPU only. Every script runs from the repository root.

## Roadmap

- [x] Rebuild DeiT-S, verify it and account for its size
- [x] The dense baselines as real files on ImageNet-1k val, V2 and R
- [x] The shared calibration list and the pipeline for published methods
- [x] The other published methods under the same protocol: APHQ-ViT, FIMA-Q, RepQ-ViT and PTQ4ViT, as published and weights-only (FQ-ViT, I&S-ViT and ERQ left out)
- [x] Pruning without retraining: CORP, reimplemented with its least-squares compensation, GRAIL and the pruning stage of VBP
- [x] Sensitivity of each layer and block to pruning and to weight bits
- [x] Per-block allocation and the bits-only, pruning-only and combined curves
- [x] The W8A8 series and 8-bit activations on our models
- [x] Which activations of FIMA-Q are fragile, against the predictions in `ABLATIONS.md`
- [ ] Future work: other orders of pruning and quantization, quantization-aware training, specialized kernels for low-bit activations, and fine-tuning as a final step to recover accuracy

## AI declaration

Claude Opus 5.5 was used as an assistant for coding, for queuing experiments remotely and for drafting text. The team validated all the code and results, and provided the writing style and guides that keep the documents uncluttered and easy to follow. The research direction, the methodology, the proofreading and everything else were done by the team.

## Key references

- Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? *MLSys*.
- Hendrycks, D., et al. (2021). The many faces of robustness: A critical analysis of out-of-distribution generalization. *ICCV*.
- Li, Z., Xiao, J., Yang, L., & Gu, Q. (2023). RepQ-ViT: Scale reparameterization for post-training quantization of vision transformers. *ICCV*.
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*.
- Touvron, H., et al. (2021). Training data-efficient image transformers & distillation through attention. *ICML*.
- Wu, Z., Chen, J., Zhong, H., Huang, D., & Wang, Y. (2024). AdaLog: Post-training quantization for vision transformers with adaptive logarithm quantizer. *ECCV*.
- Wu, Z., Wang, S., Zhang, J., Chen, J., & Wang, Y. (2025). FIMA-Q: Post-training quantization for vision transformers by Fisher information matrix approximation. *CVPR*.
- Yuan, Z., et al. (2022). PTQ4ViT: Post-training quantization for vision transformers with twin uniform quantization. *ECCV*.
- Zhang, B., & Yang, B. (2026). CORP: Closed-form one-shot structured pruning for transformers. *arXiv*.
