"""The bookkeeping every script shares: the FP32 reference to pair with, adding a point to a `*_summary.json` and its
per-image results to the matching `.npz`, and printing a point's scores. Paths are taken as given, so the scripts keep
running from the repository root and the files they record stay relative to it."""
import json
from pathlib import Path

import numpy as np

SETS = ("ImageNet-1k val", "ImageNet-V2", "ImageNet-R")
REPO = Path(__file__).resolve().parents[1]


def fp32_reference(model="DeiT-S", sets=SETS):
    """{set: per-image 0/1} of the FP32 model, as notebook 02 saved it, for paired differences on the same images."""
    base = np.load(REPO / "outputs" / "baselines_per_image.npz")
    return {d: base[f"{d}|{model} FP32"] for d in sets}


def add_point(summary_path, name, entry, per_image_path=None, correct=None):
    """Add `entry` under `name` to the summary (created if missing), and `correct` ({set: 0/1 per image}) to the npz."""
    summary_path = Path(summary_path)
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary[name] = entry
    json.dump(summary, open(summary_path, "w"), indent=2, default=float)
    if per_image_path is not None and correct is not None:
        per_image_path = Path(per_image_path)
        old = dict(np.load(per_image_path)) if per_image_path.exists() else {}
        old.update({f"{d}|{name}": c for d, c in correct.items()})
        np.savez_compressed(per_image_path, **old)


def report(res):
    """One line per set: top-1 with its interval, and the paired difference with FP32 with its interval."""
    for d, r in res.items():
        v = r["vs_fp32"]
        print(f"  {d:<16} top-1 {r['top1']:6.2f} % [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
              f"  vs FP32 {v['diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]", flush=True)
