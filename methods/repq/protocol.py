"""RepQ-ViT (Li et al., 2023) on DeiT-S under our protocol, from the authors' code at commit a160561, unchanged.

    cd <RepQ-ViT>/classification && $ADALOG_PY <repo>/methods/repq/protocol.py --w 4 --a 4 --dataset <imagenet>  # as published
    cd <RepQ-ViT>/classification && $ADALOG_PY <repo>/methods/repq/protocol.py --w 4 --dataset <imagenet>        # weights-only

The authors' main() is reproduced step by step (initial quantization, scale reparameterization, re-calibration) with only
the calibration batch changed to the first 32 lines of our list. Weights-only leaves every activation in float, so the
reparameterization, which moves activation scales, has nothing to act on and is skipped; what remains is the percentile
search for each row's weight grid. The weights go to a packed file, checked bit for bit against the method's quantizer.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.nn import Parameter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [os.getcwd(), str(REPO)]
from quant import quant_model, set_quant_state              # the authors'
from quant.quant_modules import QuantConv2d, QuantLinear
from utils.build_dataset import build_transform
from utils.build_model import build_model

from deitc.evaluate import score_in_process
from deitc.model import deit_small
from deitc.packing import load_packed_rows, save_packed_rows

RAW = REPO.parent / "third_party/FIMA-Q-protocol/checkpoints/vit_raw/deit_small_patch16_224.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--a", type=int, default=32, help="32 leaves activations in float")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib", type=int, default=32)
    args = ap.parse_args()
    published = args.a < 32
    tag = f"repq_w{args.w}" + (f"a{args.a}" if published else "")
    name = f"RepQ-ViT W{args.w}" + (f"A{args.a}" if published else "")

    torch.manual_seed(0)
    create = timm.create_model
    timm.create_model = lambda n, pretrained=False, **k: create(n, checkpoint_path=str(RAW))   # no download
    model = build_model("deit_small_patch16_224").cuda().eval()
    tfm = build_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_pct=0.875)
    lines = (REPO / "splits/calib_seed0.txt").read_text().split()[:args.calib]
    calib = torch.stack([tfm(Image.open(Path(args.dataset) / "train" / f).convert("RGB")) for f in lines]).cuda()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    wq, aq = {"n_bits": args.w, "channel_wise": True}, {"n_bits": min(args.a, 8), "channel_wise": False}
    q = quant_model(model, input_quant_params=aq, weight_quant_params=wq).cuda().eval()
    set_quant_state(q, input_quant=published, weight_quant=True)
    with torch.no_grad():
        q(calib)
        if published:   # test_quant.py, scale reparameterization
            modules = {}
            for n, m in q.blocks.named_modules():
                modules[n] = m
                father = modules.get(n[:max(n.rfind("."), 0)])
                if "norm1" in n or "norm2" in n:
                    nxt = father.attn.qkv if "norm1" in n else father.mlp.fc1
                    delta, zero = nxt.input_quantizer.delta.reshape(-1), nxt.input_quantizer.zero_point.reshape(-1)
                    act_min = -zero * delta
                    t_delta, t_zero = delta.mean(), zero.mean()
                    r = delta / t_delta
                    b = act_min / r - (-t_zero * t_delta)
                    m.weight.data = m.weight.data / r
                    m.bias.data = m.bias.data / r - b
                    nxt.weight.data = nxt.weight.data * r
                    nxt.bias.data = nxt.bias.data + torch.mm(nxt.weight.data, b.reshape(-1, 1)).reshape(-1)
                    nxt.input_quantizer.channel_wise = False
                    nxt.input_quantizer.delta, nxt.input_quantizer.zero_point = t_delta, t_zero
                    nxt.weight_quantizer.inited = False
            set_quant_state(q, input_quant=True, weight_quant=True)
            q(calib)
    torch.cuda.synchronize()
    seconds = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    # The packed file, read back and checked against the method's own quantizer
    sd = q.state_dict()
    entries, ref = {}, {}
    for n, m in q.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            wq_ = m.weight_quantizer
            w = m.weight.data
            delta, zero = wq_.delta.reshape(w.shape[0], 1), wq_.zero_point.reshape(w.shape[0], 1)
            codes = (torch.round(w.reshape(w.shape[0], -1) / delta) + zero).clamp(0, wq_.n_levels - 1)
            entries[n + ".weight"] = {"q": codes.to(torch.uint8).reshape(w.shape).cpu(), "scale": delta.cpu(), "zero": zero.cpu()}
            ref[n + ".weight"] = wq_(w).cpu()
    packable = {}
    if published:   # the activation quantizers the point needs at inference
        for n, m in q.named_modules():
            quant = getattr(m, "input_quantizer", None)
            for nm, qz in (("input_quantizer", quant), ("quantizer_A", getattr(m, "quantizer_A", None)), ("quantizer_B", getattr(m, "quantizer_B", None))):
                if qz is not None and getattr(qz, "delta", None) is not None:
                    packable[f"{n}.{nm}.delta"] = torch.as_tensor(qz.delta).float().reshape(-1).cpu()
                    z = getattr(qz, "zero_point", None)
                    if z is not None:
                        packable[f"{n}.{nm}.zero_point"] = torch.as_tensor(z).float().reshape(-1).cpu()
    for k in deit_small().state_dict():
        packable[k] = entries[k] if k in entries else sd[k].float().half().cpu()
    out = REPO / "outputs/artifacts" / f"{tag}.safetensors"
    size = save_packed_rows(packable, out, args.w, method="repq")
    back = load_packed_rows(out)
    bad = [k for k in ref if not torch.equal(back[k], ref[k])]
    if bad:
        raise SystemExit(f"round trip failed on {len(bad)} layers, e.g. {bad[:3]}")
    print(f"{out.name}: {size / 1e6:.3f} MB, {len(ref)} quantized layers identical to the method's weights", flush=True)

    record = {"run": "as published" if published else "weights-only", "w_bit": args.w, "a_bit": args.a, "calib_size": args.calib,
              "file": str(out.relative_to(REPO)), "calibration_s": seconds, "peak_gpu_mib_after_calibration": peak}
    if published:
        score_in_process(q, tfm, name, REPO / "outputs/published_summary.json", REPO / "outputs/published_per_image.npz", record)
    else:
        score_in_process(q, tfm, f"{name}, in the method's forward", REPO / "outputs/validation_summary.json", None, record)


if __name__ == "__main__":
    main()
