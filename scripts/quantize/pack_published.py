"""Pack a published method's checkpoint into a real file and check it against the method's own rounding.

    python scripts/quantize/pack_published.py --method adalog --bits 4 --ckpt <path>.pth --out outputs/artifacts/adalog_w4.safetensors

The check: every quantized weight read back from the file equals the weight the method's forward pass uses,
bit for bit (`torch.equal`). If it fails, nothing is evaluated.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import json

import torch

from deitc.packing import save_packed_rows, load_packed_rows
from deitc.published import adalog_to_packable, adalog_fake_weights

# APHQ-ViT and FIMA-Q are built on AdaLog's code and keep the same per-row grid; after block reconstruction the
# weights in their checkpoint are already AdaRound's hard-rounded values, which the same rounding reproduces.
METHODS = {m: (adalog_to_packable, adalog_fake_weights) for m in ("adalog", "aphq", "fimaq")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--bits", type=int, default=0, help="one width for all layers; see --layer-bits")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--as-published", action="store_true", help="also store the activation quantizers")
    ap.add_argument("--layer-bits", help="json {layer: width} of a run that mixes widths; replaces --bits per layer")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    to_packable, fake = METHODS[args.method]
    bits = args.bits
    if args.layer_bits:
        bits = {k: v for k, v in json.load(open(args.layer_bits)).items()}
    size = save_packed_rows(to_packable(ckpt, bits, args.as_published), args.out, bits, method=args.method)
    back, ref = load_packed_rows(args.out), fake(ckpt, bits)
    bad = [k for k in ref if not torch.equal(back[k], ref[k])]
    if bad:
        raise SystemExit(f"round trip failed on {len(bad)} layers, e.g. {bad[:3]}")
    print(f"{args.out}: {size / 1e6:.3f} MB, {len(ref)} quantized layers identical to the method's weights")


if __name__ == "__main__":
    main()
