"""Score a packed model file on the three test sets and pair it with DeiT-S FP32 on the same images.

    python evaluate_file.py outputs/artifacts/adalog_w4.safetensors --name "AdaLog W4"

The weights come from the file, so the accuracy belongs to the stored model. The FP32 per-image results are the
ones notebook 02 saved in `outputs/baselines_per_image.npz`. Results are added to `outputs/published_summary.json`
and `outputs/published_per_image.npz`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deitc.evaluate import score
from deitc.packing import load_packed_rows

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="DeiT-S")
    args = ap.parse_args()

    base = np.load(OUT / "baselines_per_image.npz")
    reference = {d: base[f"{d}|{args.model} FP32"] for d in ("ImageNet-1k val", "ImageNet-V2", "ImageNet-R")}
    res, correct = score(load_packed_rows(args.file), reference, args.model)
    size = Path(args.file).stat().st_size
    summary_path = OUT / "published_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary[args.name] = {"file": str(args.file), "size_mb": size / 1e6, "accuracy": res}
    json.dump(summary, open(summary_path, "w"), indent=2, default=float)
    per_image = OUT / "published_per_image.npz"
    old = dict(np.load(per_image)) if per_image.exists() else {}
    old.update({f"{d}|{args.name}": c for d, c in correct.items()})
    np.savez_compressed(per_image, **old)
    print(f"{args.name}: {size / 1e6:.2f} MB")
    for d, r in res.items():
        v = r["vs_fp32"]
        print(f"  {d:<16} top-1 {r['top1']:6.2f} % [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
              f"  vs FP32 {v['diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")


if __name__ == "__main__":
    main()
