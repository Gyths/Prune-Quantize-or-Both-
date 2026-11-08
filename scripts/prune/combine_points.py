"""Quantize a pruned model and score it from the packed file: pruning and low bits together, on the same axes.

    python scripts/prune/combine_points.py rtn corp-both_s10 --bits 3 4        # round-to-nearest, per row, as the floor
    python scripts/prune/combine_points.py file corp-both_s10 fimaq 3 FILE     # a file `scripts/quantize/pack_published.py` wrote from a method's run

The base is an entry of `outputs/pruned_summary.json`, taken from its FP32 file. Round-to-nearest uses the same layers
and grid as `scripts/quantize/validate_low_bits.py rtn-rows`. Results go to `outputs/combined_summary.json` and
`outputs/combined_per_image.npz`, paired with DeiT-S FP32 like every other point.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from deitc.evaluate import score
from deitc.packing import load_packed_rows, save_packed_rows
from deitc.quant import adalog_layer, rtn_rows
from deitc.results import add_point, fp32_reference, report

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def add(base, quantizer, bits, file, seconds=None):
    pruned = json.load(open(OUT / "pruned_summary.json"))[base]
    res, correct = score(load_packed_rows(file), fp32_reference())
    name, size = f"{base}+{quantizer}_w{bits}", Path(file).stat().st_size
    add_point(OUT / "combined_summary.json", name,
              {"base": base, "quantizer": quantizer, "bits": bits, "file": str(file), "size_mb": size / 1e6,
               "prune_seconds": pruned["calib_seconds"], "quantize_seconds": seconds, "accuracy": res},
              OUT / "combined_per_image.npz", correct)
    print(f"{name}: {size / 1e6:.2f} MB")
    report(res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    r = sub.add_parser("rtn"); r.add_argument("base"); r.add_argument("--bits", nargs="+", type=int, default=[3, 4])
    f = sub.add_parser("file"); f.add_argument("base"); f.add_argument("quantizer"); f.add_argument("bits", type=int)
    f.add_argument("path"); f.add_argument("--seconds", type=float)
    args = ap.parse_args()
    if args.what == "file":
        return add(args.base, args.quantizer, args.bits, args.path, args.seconds)
    sd = load_file(OUT / "artifacts" / f"{args.base}_fp32.safetensors")
    for b in args.bits:
        q = {k: rtn_rows(v.float(), b) if adalog_layer(k, v) else v.float().half() for k, v in sd.items()}
        file = OUT / "artifacts" / f"{args.base}_rtn_w{b}.safetensors"
        save_packed_rows(q, file, b, method="rtn-rows")
        add(args.base, "rtn", b, file, 0.0)


if __name__ == "__main__":
    main()
