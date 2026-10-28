"""Scoring on the three test sets, with bootstrap intervals and paired differences on the same images."""
import os
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config, create_transform

from .data import WNID_TO_INDEX, imagenet_v2, by_wnid
from .model import BUILDERS, TIMM_NAMES

DATA = {
    "ImageNet-1k val": Path(os.environ.get("IMAGENET_VAL_DIR", "../data/imagenet/val")),
    "ImageNet-V2": Path(os.environ.get("IMAGENETV2_DIR", "../data/imagenetv2-matched-frequency-format-val")),
    "ImageNet-R": Path(os.environ.get("IMAGENETR_DIR", "../data/imagenet-r")),
}


@torch.no_grad()
def predict(models, loader, device="cuda"):
    for m in models.values(): m.to(device).eval()
    logits = {k: [] for k in models}; labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        for k, m in models.items(): logits[k].append(m(x).float().cpu())
        labels.append(y)
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


def score(sd, reference, model="DeiT-S", device="cuda"):
    """Top-1 of a state_dict on each set, and its paired difference with `reference` ({set: per-image 0/1})."""
    m = BUILDERS[model]()
    m.load_state_dict(sd, strict=True)
    tfm = create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES[model], pretrained=False)))
    res, correct = {}, {}
    for dname, root in DATA.items():
        ds = imagenet_v2(root, tfm) if dname == "ImageNet-V2" else by_wnid(root, tfm)
        loader = torch.utils.data.DataLoader(ds, batch_size=250, num_workers=8, pin_memory=True)
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
