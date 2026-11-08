"""Checks that the low-bit points make sense, written to `outputs/validation_summary.json`.

    python scripts/quantize/validate_low_bits.py rtn-rows            # round-to-nearest on AdaLog's own grid, 3, 4 and 6 bits
    python scripts/quantize/validate_low_bits.py weights-of FILE...  # a packed file's weights with activations at full precision

`rtn-rows` quantizes the same 50 layers AdaLog does (patch embedding, the 48 block linears and the head) with the
same grid, one asymmetric min-max grid per output row, but no search. The distance between it and AdaLog at the
same bit width is what AdaLog's search gains, and it should grow as the bits go down.

`weights-of` scores the weights stored in a file as if the activations were left at full precision, ignoring any
activation quantizers the file also holds. For a run as published, the distance between this and the run itself
is the part of its loss that comes from quantizing the activations. AdaLog shifts the input of each fc2 by a
constant s before its log quantizer and folds -s into the fc2 bias (`reparam_bias`: b' = b - s * W_q 1), so for
float activations the bias is put back as b = b' + s * W_q 1; the LayerNorm reparameterization of qkv and fc1 is
exact in float and needs nothing.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
from pathlib import Path

import torch

from deitc.evaluate import score
from deitc.model import load_pretrained
from deitc.packing import save_packed_rows, load_packed_rows
from deitc.quant import rtn_rows, adalog_layer
from deitc.results import add_point, fp32_reference

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def record(name, res, extra):
    add_point(OUT / "validation_summary.json", name, {**extra, "accuracy": res})
    print(name, {d: round(r["top1"], 2) for d, r in res.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rtn-rows", "weights-of"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--bits", nargs="+", type=int, default=[3, 4, 6])
    args = ap.parse_args()

    reference = fp32_reference()
    if args.what == "rtn-rows":
        sd = load_pretrained("DeiT-S").state_dict()
        for b in args.bits:
            q = {k: rtn_rows(v.float(), b) if adalog_layer(k, v) else v.float().half() for k, v in sd.items()}
            f = OUT / "artifacts" / f"rtn_rows_w{b}.safetensors"
            size = save_packed_rows(q, f, b, method="rtn-rows")
            res, _ = score(load_packed_rows(f), reference)
            record(f"RTN W{b} per row, AdaLog's layers", res, {"file": str(f), "size_mb": size / 1e6})
    else:
        keys = set(load_pretrained("DeiT-S").state_dict())
        for f in args.files:
            raw = load_packed_rows(f)
            for k in [k for k in raw if k.endswith("fc2.a_quantizer.shift")]:
                layer = k[: -len(".a_quantizer.shift")]
                if raw.get(layer + ".a_quantizer.bias_reparamed", torch.tensor(0.)).item():
                    raw[layer + ".bias"] = raw[layer + ".bias"] + raw[k].item() * raw[layer + ".weight"].sum(1)
            sd = {k: v for k, v in raw.items() if k in keys}
            res, _ = score(sd, reference)
            record(f"weights of {Path(f).stem}, activations FP32", res, {"file": f})


if __name__ == "__main__":
    main()
