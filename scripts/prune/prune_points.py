"""Prune DeiT-S without retraining, write the model to a real file and score it from that file.

    python scripts/prune/prune_points.py vbp --sparsity 0.5
    python scripts/prune/prune_points.py grail-mag-l2 --sparsity 0.5
    python scripts/prune/prune_points.py corp --sparsity 0.5 --scope both

VBP and GRAIL run the authors' code from `../third_party` (VBP at commit b6834b4, of which only the pruning stage runs
here because fine-tuning is out of scope; GRAIL at ba5e48b, whose ViT module map is pointed at DeiT's MLP layers). CORP
is our reimplementation in `deitc/pruning.py`, because no code for it is public. Calibration images are the first N
lines of `splits/calib_seed0.txt`: 128 for GRAIL and 4 000 for CORP, as their papers say, and for VBP, whose authors use
the whole training set, the longest list we have (4 000). VBP keeps the authors' training transform; the other two use
the test transform, as GRAIL's code asks for and CORP's paper does not contradict.

Each model is saved in FP32 and FP16, like the dense baselines, and scored from the FP16 file. Results are added to
`outputs/pruned_summary.json` and `outputs/pruned_per_image.npz`. `--val-only` scores on ImageNet-1k val without saving
to the summary, for checking against a paper.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from deitc.evaluate import score
from deitc.model import load_pretrained
from deitc.packing import save_float
from deitc.pruning import run_corp, run_grail, run_vbp
from deitc.results import add_point, fp32_reference, report

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")
DEFAULT_N = {"vbp": 4000, "grail-mag-l1": 128, "grail-mag-l2": 128, "grail-wanda": 128, "corp": 4000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=DEFAULT_N)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--scope", default="both", choices=["both", "mlp", "attn"], help="CORP only")
    ap.add_argument("--no-compensation", action="store_true", help="GRAIL only: the selection alone, as a floor")
    ap.add_argument("--calib", type=int, help="number of calibration images, if not the method's own")
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    n = args.calib or DEFAULT_N[args.method]
    model = load_pretrained("DeiT-S").cuda()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    if args.method == "vbp":
        pruned = run_vbp(model, args.sparsity, n)
    elif args.method == "corp":
        pruned = run_corp(model, args.sparsity, n, args.scope)
    else:
        pruned = run_grail(model, args.method[6:], args.sparsity, n, not args.no_compensation)
    seconds = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    sd = {k: v.detach().cpu() for k, v in pruned.state_dict().items()}
    params = sum(v.numel() for v in sd.values())
    label = args.method + (f"-{args.scope}" if args.method == "corp" else "") + ("-nocomp" if args.no_compensation else "")
    pct = round(args.sparsity * 100)
    tag = f"{label}_s{pct}" + (f"_n{n}" if args.calib else "")
    fp32, fp16 = OUT / "artifacts" / f"{tag}_fp32.safetensors", OUT / "artifacts" / f"{tag}_fp16.safetensors"
    size32, size16 = save_float(sd, fp32, torch.float32), save_float(sd, fp16, torch.float16)

    names = ("ImageNet-1k val",) if args.val_only else ("ImageNet-1k val", "ImageNet-V2", "ImageNet-R")
    res, correct = score(load_file(fp16), fp32_reference(sets=names), sets=names)
    print(f"{tag}: {params / 1e6:.2f} M params, FP16 {size16 / 1e6:.2f} MB, FP32 {size32 / 1e6:.2f} MB, "
          f"{seconds:.0f} s, peak {peak:.0f} MiB")
    report(res)
    if args.val_only:
        return
    add_point(OUT / "pruned_summary.json", tag,
              {"method": args.method, "scope": "mlp" if args.method != "corp" else args.scope,
               "compensated": not args.no_compensation, "sparsity": args.sparsity, "calib_images": n,
               "params": params, "file": str(fp16), "size_mb": size16 / 1e6, "size_mb_fp32": size32 / 1e6,
               "calib_seconds": seconds, "calib_peak_gpu_mib": peak, "accuracy": res},
              OUT / "pruned_per_image.npz", correct)


if __name__ == "__main__":
    main()
