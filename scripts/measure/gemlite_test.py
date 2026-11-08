"""Can GemLite's GPU kernels run our low-bit DeiT-S files? Fidelity, top-1, latency and memory, per format.

    PYTHONPATH=../third_party/gemlite python scripts/measure/gemlite_test.py check W4
    PYTHONPATH=../third_party/gemlite python scripts/measure/gemlite_test.py run "W4 A16" [--score]

GemLite is a set of Triton kernels for 1, 2, 4 and 8-bit weights, with FP16 or dynamic 8-bit activations, written for
language models. Weights come from the FIMA-Q packed files. Their per-row scales are GemLite's grouped scales with the
group as wide as the layer. W3 runs in the 4-bit kernel and W6 in the 8-bit one, since no 3 or 6-bit kernel exists, so
those two take the memory and compute of the larger container at run time. The patch embedding and the head stay FP16.
Results go to `outputs/gemlite_summary.json`.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_data_config

from deitc.evaluate import score_in_process
from deitc.model import TIMM_NAMES, load_pretrained
from deitc.packing import load_entries, load_packed_rows
from gemlite.helper import A8W8_int8_dynamic, A8Wn_HQQ_INT_dynamic, A16W8_INT8, A16Wn
from deitc.costs import time_model

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")
CONTAINER = {2: 2, 3: 4, 4: 4, 6: 8, 8: 8}
SUMMARY = OUT / "gemlite_summary.json"


def kernel(entry, bias, activation):
    """One GemLite linear from a packed layer. `activation` is 16 (FP16) or 8 (dynamic FP8), as GemLite defines it."""
    q, scale, zero = entry["q"], entry["scale"].float().half(), entry["zero"].float().half()
    bits = CONTAINER[int(entry["bits"])]
    in_features = q.shape[1]
    if activation == 16:
        return A16Wn(device="cuda:0", dtype=torch.float16).from_weights(q, scale, zero, bits, in_features, bias=bias)
    return A8Wn_HQQ_INT_dynamic(device="cuda:0", dtype=torch.float16, W_nbits=bits).from_weights_(
        q, scale, zero, bits, in_features, bias=bias)


def load_layers(bits):
    entries, raw, _ = load_entries(OUT / "artifacts" / f"fimaq_w{bits}.safetensors")
    for e in entries.values():
        e["bits"] = bits
    return entries, raw


def build(bits, activation):
    """W8 quantizes the FP32 weights with GemLite's own channel-wise INT8 (FIMA-Q has no W8 file), with INT8 activations
    at A8, the same format as torchao's W8A8. Bits 0 is plain FP16, the reference for the same session."""
    if bits in (0, 8):
        m = load_pretrained("DeiT-S").cuda().eval().half()
        if bits == 8:
            make = A16W8_INT8 if activation == 16 else A8W8_int8_dynamic
            for fqn, mod in list(m.named_modules()):
                if isinstance(mod, nn.Linear) and fqn.startswith("blocks."):
                    setattr(m.get_submodule(fqn.rpartition(".")[0]), fqn.rpartition(".")[2],
                            make(device="cuda:0", dtype=torch.float16).from_linear(mod, del_orig=False))
        torch.cuda.empty_cache()
        return m
    entries, raw = load_layers(bits)
    sd = load_packed_rows(OUT / "artifacts" / f"fimaq_w{bits}.safetensors")
    m = load_pretrained("DeiT-S")
    m.load_state_dict(sd, strict=True)
    m = m.cuda().eval().half()
    for fqn, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and fqn.startswith("blocks."):
            parent = m.get_submodule(fqn.rpartition(".")[0])
            bias = mod.bias.data if mod.bias is not None else None
            setattr(parent, fqn.rpartition(".")[2], kernel(entries[fqn + ".weight"], bias, activation))
    torch.cuda.empty_cache()
    return m


def check(args):
    """Each kernel against the same dequantized weights in a plain FP16 linear, on activation-like inputs."""
    bits = int(args.fmt[1:])
    entries, raw = load_layers(bits)
    sd = load_packed_rows(OUT / "artifacts" / f"fimaq_w{bits}.safetensors")
    torch.manual_seed(0)
    for k in ("blocks.3.mlp.fc1", "blocks.3.mlp.fc2", "blocks.3.attn.qkv", "blocks.3.attn.proj"):
        w, b = sd[k + ".weight"].cuda().half(), sd[k + ".bias"].cuda().half() if k + ".bias" in sd else None
        x = (torch.randn(4, 197, w.shape[1]).cuda() * 0.5).half()
        ref = nn.functional.linear(x, w, b)
        for a in (16, 8):
            try:
                out = kernel(entries[k + ".weight"], b, a)(x)
                err = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
                print(f"{k:20s} W{bits} A{a}: relative error {err:.5f}", flush=True)
            except Exception as e:
                print(f"{k:20s} W{bits} A{a}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


class HalfInput(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x.half()).float()


def run(args):
    w, a = args.fmt.split()
    bits, activation = int(w[1:]), int(a[1:])
    m = build(bits, activation)
    label = f"GemLite W{bits} A{activation}" if bits else "FP16, same session"
    summary = json.load(open(SUMMARY)) if SUMMARY.exists() else {}
    record = summary.setdefault(label, {"weights": f"fimaq_w{bits}", "kernel_bits": CONTAINER.get(bits), "activation_bits": activation})
    start = time.time()
    with torch.no_grad():
        m(torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16))
        m(torch.randn(256, 3, 224, 224, device="cuda", dtype=torch.float16))
    record["first_calls_s"] = time.time() - start
    for mode in ("eager", "compiled"):
        net = m if mode == "eager" else torch.compile(m)
        for b in (1, 256):
            r = time_model(net, b, torch.float16)
            record[f"{mode} batch {b}"] = {k: r[k] for k in ("median_ms", "p25_ms", "p75_ms", "images_per_s", "peak_mem_mib")}
            print(f"{label} | {mode} | batch {b}: {r['median_ms']:.2f} ms, {r['images_per_s']:.0f} img/s, "
                  f"peak {r['peak_mem_mib']:.0f} MiB", flush=True)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    if args.score:
        tfm = create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES["DeiT-S"], pretrained=False)))
        acc = score_in_process(HalfInput(m), tfm, label, OUT / "gemlite_scores.json", OUT / "gemlite_per_image.npz", {})
        summary = json.load(open(SUMMARY))
        summary[label]["accuracy"] = acc
        json.dump(summary, open(SUMMARY, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("fmt")
    r = sub.add_parser("run"); r.add_argument("fmt"); r.add_argument("--score", action="store_true")
    args = ap.parse_args()
    {"check": check, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
