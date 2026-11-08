"""Latency and peak GPU memory for the formats that have a real kernel on our GPU, in `outputs/latency_summary.json`.

    python scripts/measure/measure_latency.py                 # every format, eager and compiled, batch 1 and 256, one process each
    python scripts/measure/measure_latency.py --check-w8a8    # also confirm the W8A8 model computes with integer kernels
    python scripts/measure/measure_latency.py --pruned        # one pruned model per set of layer sizes, with DeiT-S FP16 as reference,
                                              # in `outputs/latency_pruned_summary.json`

Latency depends on the format (bit width, grid shape, layer sizes) and not on the weight values, so one
measurement serves every method that shares a format. The formats with a real kernel are FP32, FP16, W8A8
(torchao, on an FP32 base as scored and on an FP16 base as it would be deployed; INT8 weights and dynamically quantized INT8 activations on the 48 block linears; the head stays in float
because the integer matmul needs more than 16 rows and the head sees one class token per image) and the smaller dense model;
pruned models are added as they appear. Each format is timed in eager mode and under `torch.compile`, since
torchao's INT8 path reaches the integer matmul only once compiled. TF32 is off, as in the accuracy runs, so the
time belongs to the arithmetic that was scored. Each format runs in its own process, so that memory left by one
format cannot count towards the next. Run it with nothing else on the GPU.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from deitc.costs import time_model
from deitc.model import from_state_dict, load_pretrained

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def pruned_groups():
    """{tag of the first model with those layer sizes: every tag that shares them}. Same sizes, same latency."""
    groups = {}
    for tag, r in json.load(open(OUT / "pruned_summary.json")).items():
        with safe_open(r["file"], "pt") as f:
            sizes = tuple(sorted((k, tuple(f.get_slice(k).get_shape())) for k in f.keys()))
        groups.setdefault(sizes, []).append(tag)
    return {tags[0]: tags for tags in groups.values()}


def build(fmt):
    model_name, kind = fmt.split()
    if model_name == "Pruned":
        sd = {k: v.float() for k, v in load_file(OUT / "artifacts" / f"{kind}_fp16.safetensors").items()}
        return from_state_dict(sd).cuda().eval().half(), torch.float16
    if model_name == "PrunedW8A8":   # a pruned model in the format the W8A8 files hold: INT8 block linears, the rest FP16
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        sd = {k: v.float() for k, v in load_file(OUT / "artifacts" / f"{kind}_fp16.safetensors").items()}
        m = from_state_dict(sd).cuda().eval().half()
        quantize_(m, Int8DynamicActivationInt8WeightConfig(),
                  filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and fqn.startswith("blocks."))
        return m, torch.float16
    m = load_pretrained(model_name).cuda().eval()
    if kind == "FP16":
        return m.half(), torch.float16
    if kind.startswith("W8A8"):
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        dtype = torch.float16 if kind == "W8A8-FP16" else torch.float32   # the base the rest of the model runs in
        m = m.to(dtype)
        quantize_(m, Int8DynamicActivationInt8WeightConfig(),
                  filter_fn=lambda mod, fqn: isinstance(mod, torch.nn.Linear) and fqn.startswith("blocks."))
        return m, dtype
    return m, torch.float32


def uses_int_mm(model, dtype):
    """True if the compiled forward pass calls an integer matmul (aten._int_mm or a Triton int8 kernel)."""
    from torch.profiler import profile, ProfilerActivity
    x = torch.randn(64, 3, 224, 224, device="cuda", dtype=dtype)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        model(x); torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    return sorted(n for n in names if "int_mm" in n or "int8" in n.lower() or "imma" in n.lower())[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+",
                    default=["DeiT-S FP32", "DeiT-S FP16", "DeiT-S W8A8", "DeiT-S W8A8-FP16",
                             "DeiT-Ti FP32", "DeiT-Ti FP16", "DeiT-Ti W8A8", "DeiT-Ti W8A8-FP16"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 256])
    ap.add_argument("--check-w8a8", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["eager", "compiled"])
    ap.add_argument("--pruned", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)   # one format and mode, results to this json
    args = ap.parse_args()
    summary = OUT / "latency_summary.json"
    extra = {}
    if args.pruned:
        groups = pruned_groups()
        args.formats = ["DeiT-S FP16"] + [f"Pruned {t}" for t in groups]
        summary, extra = OUT / "latency_pruned_summary.json", {"same_layer_sizes": groups}

    if args.worker is None:                    # parent: one fresh process per format, then merge
        out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, **extra, "results": {}}
        for fmt in args.formats:
            for mode in args.modes:
                tmp = OUT / ".latency_part.json"
                cmd = [sys.executable, __file__, "--formats", fmt, "--modes", mode,
                       "--batches", *map(str, args.batches), "--worker", str(tmp)]
                subprocess.run(cmd + (["--check-w8a8"] if args.check_w8a8 else []), check=True)
                out["results"].update(json.load(open(tmp))["results"]); tmp.unlink()
                json.dump(out, open(summary, "w"), indent=2)
        return

    out = {"results": {}}
    for fmt in args.formats:
        for compiled in [m == "compiled" for m in args.modes]:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            m, dtype = build(fmt)
            torch.cuda.synchronize()
            model_mib = (torch.cuda.memory_allocated() - before) / 2**20
            if compiled:
                m = torch.compile(m)
            for b in args.batches:
                r = time_model(m, b, dtype)
                r["model_mib"] = model_mib
                key = f"{fmt} | {'compiled' if compiled else 'eager'} | batch {b}"
                out["results"][key] = r
                print(f"{key:<36} {r['median_ms']:8.2f} ms [{r['p25_ms']:.2f}, {r['p75_ms']:.2f}]"
                      f"  {r['images_per_s']:8.0f} img/s  peak {r['peak_mem_mib']:7.0f} MiB (weights {model_mib:.0f})")
            if args.check_w8a8 and "W8A8" in fmt and compiled:
                out["results"][f"{fmt} | integer kernels"] = k = uses_int_mm(m, dtype)
                print(f"  integer kernels seen: {k}")
            del m; torch.cuda.empty_cache()
            json.dump(out, open(args.worker, "w"), indent=2)


if __name__ == "__main__":
    main()
