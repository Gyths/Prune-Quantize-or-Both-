"""Score a packed model file on the three test sets and pair it with DeiT-S FP32 on the same images.

    python scripts/quantize/evaluate_file.py outputs/artifacts/adalog_w4.safetensors --name "AdaLog W4"

The weights come from the file, so the accuracy belongs to the stored model. The FP32 per-image results are the
ones notebook 02 saved in `outputs/baselines_per_image.npz`. Results are added to `outputs/published_summary.json`
and `outputs/published_per_image.npz`.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
from pathlib import Path

import torch

from deitc.evaluate import score
from deitc.packing import load_packed_rows
from deitc.results import add_point, fp32_reference, report

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="DeiT-S")
    ap.add_argument("--mlp-act", default="gelu", choices=["gelu", "relu"], help="relu for APHQ-ViT's MLP reconstruction")
    args = ap.parse_args()

    res, correct = score(load_packed_rows(args.file), fp32_reference(args.model), args.model, mlp_act=args.mlp_act)
    size = Path(args.file).stat().st_size
    add_point(OUT / "published_summary.json", args.name, {"file": str(args.file), "size_mb": size / 1e6, "accuracy": res},
              OUT / "published_per_image.npz", correct)
    print(f"{args.name}: {size / 1e6:.2f} MB")
    report(res)


if __name__ == "__main__":
    main()
