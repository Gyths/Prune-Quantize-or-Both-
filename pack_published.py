"""Pack a published method's checkpoint into a real file and check it against the method's own rounding.

    python pack_published.py --method adalog --bits 4 --ckpt <path>.pth --out outputs/artifacts/adalog_w4.safetensors

The check: every quantized weight read back from the file equals the weight the method's forward pass uses,
bit for bit (`torch.equal`). If it fails, nothing is evaluated.
"""
import argparse

import torch

from deitc.packing import save_packed_rows, load_packed_rows
from deitc.published import adalog_to_packable, adalog_fake_weights

METHODS = {"adalog": (adalog_to_packable, adalog_fake_weights)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    to_packable, fake = METHODS[args.method]
    size = save_packed_rows(to_packable(ckpt, args.bits), args.out, args.bits, method=args.method)
    back, ref = load_packed_rows(args.out), fake(ckpt, args.bits)
    bad = [k for k in ref if not torch.equal(back[k], ref[k])]
    if bad:
        raise SystemExit(f"round trip failed on {len(bad)} layers, e.g. {bad[:3]}")
    print(f"{args.out}: {size / 1e6:.3f} MB, {len(ref)} quantized layers identical to the method's weights")


if __name__ == "__main__":
    main()
