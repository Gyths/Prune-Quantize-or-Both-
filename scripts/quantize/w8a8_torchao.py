"""The torchao W8A8 points: INT8 weights and dynamically quantized INT8 activations on the 48 block linears.

    python scripts/quantize/w8a8_torchao.py                          # DeiT-S and DeiT-Ti
    python scripts/quantize/w8a8_torchao.py --pruned p40 corp-both_s10    # pruned models, from outputs/artifacts/NAME_fp32.safetensors

This is the format `scripts/measure/measure_latency.py` times, the one low-bit format our GPU computes with natively. The activations are
quantized inside torchao's forward pass, so each point is scored in memory, on an FP32 base as the latency's first row is.
The file holds the torchao codes with their scales, and the rest of the model in FP16. Results go to
`outputs/w8a8_summary.json`, next to which PTQ4ViT W8A8 sits in `outputs/published_summary.json`. Every point pairs with
DeiT-S FP32 on the same images, DeiT-Ti included, like every other point on the charts.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import time
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from timm.data import create_transform, resolve_data_config

from deitc.evaluate import score_in_process
from deitc.model import TIMM_NAMES, from_state_dict, load_pretrained
from deitc.packing import load_packed_rows, save_packed_rows

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def run(m, label, stem, tfm, pack_only=False):
    """Quantize `m` with torchao, pack it, check the round trip and score it under `label`."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    floats = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    start = time.time()
    quantize_(m, Int8DynamicActivationInt8WeightConfig(),
              filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and fqn.startswith("blocks."))
    seconds = time.time() - start
    packable, ref = {}, {}
    for fqn, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear) and fqn.startswith("blocks."):
            w = mod.weight
            k = fqn + ".weight"
            q, scale, zero = w.qdata.detach().cpu(), w.scale.detach().cpu().float(), w.zero_point.detach().cpu()
            rows = q.shape[0]
            scale, zero = scale.reshape(rows, 1), zero.reshape(rows, 1).float() + 128
            packable[k] = {"q": (q.to(torch.int16) + 128).to(torch.uint8), "scale": scale, "zero": zero}
            ref[k] = (q.float().reshape(rows, -1) - w.zero_point.detach().cpu().float().reshape(rows, 1)).mul(scale).reshape(q.shape)
    for k, v in floats.items():
        packable.setdefault(k, v.half())
    f = OUT / "artifacts" / f"{stem}.safetensors"
    size = save_packed_rows(packable, f, 8, method="torchao")
    back = load_packed_rows(f)
    bad = [k for k in ref if not torch.equal(back[k], ref[k])]
    if bad:
        raise SystemExit(f"round trip failed on {len(bad)} layers, e.g. {bad[:3]}")
    print(f"{f.name}: {size / 1e6:.3f} MB, {len(ref)} layers identical", flush=True)
    if pack_only:   # pruned widths that miss the GPU kernel's multiples of 8 fall back to the CPU, so scripts/quantize/a8_score.py scores them
        return
    score_in_process(m, tfm, label, OUT / "w8a8_summary.json", OUT / "w8a8_per_image.npz",
                     {"run": "W8A8", "file": str(f), "size_mb": size / 1e6, "quantize_s": seconds})
    del m
    torch.cuda.empty_cache()


def transform(name):
    return create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES[name], pretrained=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pruned", nargs="+", help="pruned bases, by the stem of their _fp32 file")
    ap.add_argument("--pack-only", action="store_true", help="write the files and do not score them")
    args = ap.parse_args()
    if args.pruned:
        tfm = transform("DeiT-S")
        for base in args.pruned:
            sd = load_file(OUT / "artifacts" / f"{base}_fp32.safetensors")
            run(from_state_dict(sd).cuda().eval(), f"DeiT-S W8A8 torchao on {base}", f"{base}_w8a8_torchao", tfm, args.pack_only)
        return
    for name in ("DeiT-S", "DeiT-Ti"):
        run(load_pretrained(name).cuda().eval(), f"{name} W8A8 torchao", f"{name.lower()}_w8a8_torchao", transform(name))


if __name__ == "__main__":
    main()
