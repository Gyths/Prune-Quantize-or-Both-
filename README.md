# Prune, Quantize, or Both? Honest Accuracy–Size Curves for a Vision Transformer

*[Versión en español](README.es.md)*

## Summary

Compression of Vision Transformers is usually reported with nominal figures, such as bits per weight or parameter counts, not with the real size of the model. Each published method uses its own protocol (a different number of calibration images, a different full-precision reference accuracy), so their results are not directly comparable. They are rarely compared with a smaller uncompressed model, and almost never evaluated beyond ImageNet-val, for example on new photos of the same classes (ImageNet-V2) or on other visual styles (ImageNet-R).

For DeiT-Small on ImageNet-1k, we ask in which ranges of the accuracy–size curve it pays to prune, to quantize, or to combine both, and which components of the model explain the difference.

To answer with a common measurement, every configuration is placed on the same axes by its real size in bytes, with paired-bootstrap confidence intervals. It sits next to DeiT-Tiny and to published post-training compression methods, all run with their authors' code under one protocol. We also test whether the conclusions hold on ImageNet-V2 and ImageNet-R.

On that basis we propose a post-training procedure. It measures how sensitive each block's attention and MLP are to structured pruning and to weight quantization. It removes attention heads and MLP neurons, compensating the next layer by least squares, then quantizes the remaining weights, and allocates these decisions per block to reach a series of target sizes. Restricting it to quantization only, or to pruning only, gives curves that are directly comparable with each other.

## What is here

The work so far covers the two reference models, the dense baselines saved as real files, and the first published method placed on the chart. The decisions behind the protocol are listed, with their reasons, in [`DECISIONS.md`](DECISIONS.md).

| Path | Role |
|---|---|
| `01_rebuild_deit.ipynb` | DeiT rebuilt from scratch and checked against timm |
| `02_dense_baselines.ipynb` | DeiT-S and DeiT-Ti at FP32, FP16, INT8 and INT4 as real files |
| `deitc/` | the code the notebooks and scripts share: model, data, packed files, scoring, and the readers for published checkpoints |
| `prepare_imagenet.py`, `make_calib_list.py` | the data, and the calibration images shared by every method (`splits/calib_seed0.txt`) |
| `pack_published.py`, `evaluate_file.py` | from a published method's checkpoint to a checked file, and from a file to its scores |
| `methods/adalog/` | our patch to the AdaLog code and the script that runs it |
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

## 3. The first published method, AdaLog

Each published method runs with its authors' code and settings, so that the point on the chart is their method and not our version of it. Three things change, and they are the same for every method. The calibration images are the first N lines of `splits/calib_seed0.txt`, one training image per class shuffled with seed 0, where N is the number the authors use. The activations stay in full precision, because they do not change the file size. The accuracy comes from a real file, built from the method's own codes and grid, read back, checked against the weights the method's code uses, and scored on the three test sets.

AdaLog (Wu et al., 2024) is the first method run this way, with its 32 calibration images and no block reconstruction. Its log quantizer acts on activations, so what is measured here is its weight quantizer and the search that sets each row's grid. It quantizes the 48 block linears, the patch embedding and the head, with one grid per output row.

| AdaLog on DeiT-S | File (MB) | ImageNet-1k val | ImageNet-V2 | ImageNet-R |
|---|---|---|---|---|
| W6 | 17.0 | 79.76 | 68.46 | 42.23 |
| W4 | 11.5 | 78.87 | 66.71 | 40.62 |
| W3 | 8.8 | 74.18 | 61.56 | 37.53 |

In all three files the weights read back are identical to AdaLog's, and the ImageNet-1k val accuracy from the file matches the one AdaLog's own code reports. The paired differences against FP32 and their intervals are in `outputs/published_summary.json`.

## Running it

```bash
# Python ≥ 3.10, CUDA GPU recommended
pip install torch torchvision timm safetensors matplotlib jupyter
```

ImageNet-V2 matched-frequency is public ([Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2)), and ImageNet-R is available from its authors. ImageNet-1k is gated on Hugging Face; once its terms are accepted, `python prepare_imagenet.py --out ../data/imagenet` writes the 50,000 validation images and a pool of training images for calibration. The notebooks read the three sets from `../data/` or from `IMAGENET_VAL_DIR`, `IMAGENETV2_DIR` and `IMAGENETR_DIR`.

```bash
jupyter nbconvert --to notebook --execute 01_rebuild_deit.ipynb    # about a minute on an RTX 4070
jupyter nbconvert --to notebook --execute 02_dense_baselines.ipynb
python make_calib_list.py --train ../data/imagenet/train --seed 0 # the shared calibration list
bash methods/adalog/run.sh                                        # AdaLog, see the script for its environment
```

## Roadmap

- [x] Rebuild DeiT-S, verify it and account for its size
- [x] The dense baselines as real files on ImageNet-1k val, V2 and R
- [x] The shared calibration list and the pipeline for published methods, with AdaLog as the first one
- [ ] The other published methods with public code (RepQ-ViT, PTQ4ViT, FQ-ViT, APHQ-ViT, FIMA-Q, I&S-ViT, ERQ) and post-training pruning
- [ ] Sensitivity of each block to pruning and to weight bits
- [ ] Structured pruning of heads and MLP neurons with least-squares compensation
- [ ] Per-block allocation and the quantize, prune and combine curves

## Key references

- Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? *MLSys*.
- Hendrycks, D., et al. (2021). The many faces of robustness: A critical analysis of out-of-distribution generalization. *ICCV*.
- Li, Z., Xiao, J., Yang, L., & Gu, Q. (2023). RepQ-ViT: Scale reparameterization for post-training quantization of vision transformers. *ICCV*.
- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*.
- Touvron, H., et al. (2021). Training data-efficient image transformers & distillation through attention. *ICML*.
- Wu, Z., Chen, J., Zhong, H., Huang, D., & Wang, Y. (2024). AdaLog: Post-training quantization for vision transformers with adaptive logarithm quantizer. *ECCV*.
- Wu, Z., Wang, S., Zhang, J., Chen, J., & Wang, Y. (2025). FIMA-Q: Post-training quantization for vision transformers by Fisher information matrix approximation. *CVPR*.
