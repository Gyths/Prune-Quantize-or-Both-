"""Scoring on the three test sets, with bootstrap intervals and paired differences on the same images."""
import os
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config, create_transform

from .data import WNID_TO_INDEX, imagenet_v2, by_wnid
from .model import TIMM_NAMES, from_state_dict

REPO = Path(__file__).resolve().parents[1]
# Relative defaults are read from the repository root, so that scripts running from a method's folder find them.
DATA = {k: p if p.is_absolute() else (REPO / p).resolve() for k, p in {
    "ImageNet-1k val": Path(os.environ.get("IMAGENET_VAL_DIR", "../data/imagenet/val")),
    "ImageNet-V2": Path(os.environ.get("IMAGENETV2_DIR", "../data/imagenetv2-matched-frequency-format-val")),
    "ImageNet-R": Path(os.environ.get("IMAGENETR_DIR", "../data/imagenet-r")),
}.items()}


def test_loader(ds, batch_size):
    """Loader for the test sets: 8 workers, file order, no pinned memory. Pinning would only speed up the copy to
    the GPU, which is not the bottleneck here, and the label tensors kept for the whole set held their pinned blocks
    alive: on 2026-11-02 one scoring pass inside APHQ-ViT's process grew shared memory to 12.5 GB."""
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=8, pin_memory=False)


@torch.no_grad()
def predict(models, loader, device="cuda"):
    for m in models.values(): m.to(device).eval()
    logits = {k: [] for k in models}; labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        for k, m in models.items(): logits[k].append(m(x).float().cpu())
        labels.append(y.clone())   # a private copy, so the worker's shared-memory batch can be released
    return {k: torch.cat(v) for k, v in logits.items()}, torch.cat(labels)


def bootstrap_ci(correct, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(1)
    return np.percentile(accs, [2.5, 97.5]) * 100


def paired_diff_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(1) * 100
    return (a.mean() - b.mean()) * 100, np.percentile(d, [2.5, 97.5])


def score(sd, reference, model="DeiT-S", device="cuda", sets=None, mlp_act="gelu"):
    """Top-1 of a state_dict on each set (all three unless `sets` names some), and its paired difference with
    `reference` ({set: per-image 0/1}). Layer sizes come from the state_dict, so pruned models score the same way."""
    m = from_state_dict({k: v.float() for k, v in sd.items()}, mlp_act)
    tfm = create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES[model], pretrained=False)))
    res, correct = {}, {}
    for dname, root in DATA.items():
        if sets and dname not in sets:
            continue
        ds = imagenet_v2(root, tfm) if dname == "ImageNet-V2" else by_wnid(root, tfm)
        loader = test_loader(ds, 250)
        logits, labels = predict({"m": m}, loader, device)
        lg = logits["m"]
        if dname == "ImageNet-R":   # scored over R's 200 classes, as in Hendrycks et al. (2021)
            classes = sorted({WNID_TO_INDEX[p.name] for p in root.iterdir() if p.is_dir()})
            mask = torch.full((1000,), float("-inf")); mask[classes] = 0
            lg = lg + mask
        c = (lg.argmax(1) == labels).numpy().astype(float)
        diff, (dlo, dhi) = paired_diff_ci(c, reference[dname].astype(float))
        res[dname] = {"top1": c.mean() * 100, "ci95": list(bootstrap_ci(c)), "vs_fp32": {"diff": diff, "ci95": [dlo, dhi]}}
        correct[dname] = c.astype(np.uint8)
    return res, correct


@torch.no_grad()
def score_in_process(model, transform, name, summary_path, per_image_path=None, extra=None, device="cuda",
                     batch_size=200):
    """Score a model as it is in memory (for example inside a method's own code, with its activations quantized)
    on the three sets, pair it with DeiT-S FP32, and add it to `summary_path` (and the per-image 0/1 results to
    `per_image_path`). Returns the accuracy dict. Batches of 200 in file order, as the methods' own validation."""
    from .results import add_point, fp32_reference
    base = fp32_reference()
    model.eval()
    res, correct = {}, {}
    for dname, root in DATA.items():
        ds = imagenet_v2(root, transform) if dname == "ImageNet-V2" else by_wnid(root, transform)
        loader = test_loader(ds, batch_size)
        lg, labels = [], []
        for x, y in loader:
            lg.append(model(x.to(device)).float().cpu()); labels.append(y.clone())
        lg, labels = torch.cat(lg), torch.cat(labels)
        if dname == "ImageNet-R":
            classes = sorted({WNID_TO_INDEX[p.name] for p in root.iterdir() if p.is_dir()})
            mask = torch.full((1000,), float("-inf")); mask[classes] = 0
            lg = lg + mask
        c = (lg.argmax(1) == labels).numpy().astype(float)
        diff, (dlo, dhi) = paired_diff_ci(c, base[dname].astype(float))
        res[dname] = {"top1": c.mean() * 100, "ci95": list(bootstrap_ci(c)), "vs_fp32": {"diff": diff, "ci95": [dlo, dhi]}}
        correct[dname] = c.astype(np.uint8)
        print(f"  {name} | {dname:<16} top-1 {res[dname]['top1']:6.2f} %  vs FP32 {diff:+.2f} [{dlo:+.2f}, {dhi:+.2f}]",
              flush=True)
    add_point(summary_path, name, {**(extra or {}), "accuracy": res}, per_image_path, correct)
    return res
