"""PTQ4ViT (Yuan et al., 2022) on DeiT-S under our protocol, from the authors' code at commit b42c790, unchanged.

    cd <PTQ4ViT> && $ADALOG_PY <repo>/methods/ptq4vit/protocol.py --w 8 --a 8 --dataset <imagenet>   # as published
    cd <PTQ4ViT> && $ADALOG_PY <repo>/methods/ptq4vit/protocol.py --w 8 --dataset <imagenet>         # weights-only

This follows the authors' `experiment_basic` (their PTQ4ViT config, Hessian calibration, batch size 4 on 32 images) with
the calibration images changed to the first 32 lines of our list. As in the paper, all of it is parallel: a layer's search
sees the others in float. Weights-only keeps every activation in float: the input quantizers become the identity and the
two attention products leave the calibration, so what remains is the Hessian-weighted search of the weight scales. Those
scales cover the whole layer (three groups in qkv, one per channel in the patch embedding), so the file stores one per
group. The weights go to a packed file, checked bit for bit against the method's own rounding.
"""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [os.getcwd(), str(REPO)]
import utils.net_wrap as net_wrap                            # the authors'
from quant_layers.conv import MinMaxQuantConv2d
from quant_layers.linear import MinMaxQuantLinear
from utils.models import get_net
from utils.quant_calib import HessianQuantCalibrator

from deitc.evaluate import score_in_process
from deitc.model import deit_small
from deitc.packing import load_packed_rows, save_packed_rows

RAW = REPO.parent / "third_party/FIMA-Q-protocol/checkpoints/vit_raw/deit_small_patch16_224.bin"


class Calib(torch.utils.data.Dataset):
    def __init__(self, root, n, tfm):
        self.files, self.root, self.tfm = (REPO / "splits/calib_seed0.txt").read_text().split()[:n], Path(root), tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.tfm(Image.open(self.root / "train" / self.files[i]).convert("RGB")), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--a", type=int, default=32, help="32 leaves activations in float")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib", type=int, default=32)
    args = ap.parse_args()
    published = args.a < 32
    tag = f"ptq4vit_w{args.w}" + (f"a{args.a}" if published else "")
    name = f"PTQ4ViT W{args.w}" + (f"A{args.a}" if published else "")

    torch.manual_seed(0)
    create = timm.create_model
    timm.create_model = lambda n, pretrained=False, **k: create(n, checkpoint_path=str(RAW))   # no download
    cfg = importlib.import_module("configs.PTQ4ViT")
    cfg.w_bit = {n: args.w for n in cfg.conv_fc_name_list}
    cfg.a_bit = {n: args.a for n in cfg.conv_fc_name_list}
    cfg.A_bit = {n: args.a for n in cfg.matmul_name_list}
    cfg.B_bit = {n: args.a for n in cfg.matmul_name_list}
    net = get_net("deit_small_patch16_224")
    tfm = create_transform(**resolve_data_config({}, model=net))
    wrapped = net_wrap.wrap_modules_in_net(net, cfg)
    if not published:
        for n in [n for n in wrapped if "matmul" in n]:
            wrapped.pop(n).mode = "raw"
        for m in wrapped.values():
            m.quant_input = lambda x: x
    loader = torch.utils.data.DataLoader(Calib(args.dataset, args.calib, tfm), batch_size=args.calib, num_workers=4)

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    HessianQuantCalibrator(net, wrapped, loader, sequential=False, batch_size=4).batching_quant_calib()
    torch.cuda.synchronize()
    seconds = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2 ** 20

    sd = net.state_dict()
    entries, ref = {}, {}
    for n, m in net.named_modules():
        if isinstance(m, (MinMaxQuantLinear, MinMaxQuantConv2d)):
            w = m.weight.data
            sim = m.quant_weight_bias()[0].detach()
            interval = m.w_interval.reshape(m.n_V, 1).detach()                  # one scale per group of rows
            rows = w.reshape(w.shape[0], -1)
            scale_rows = interval.repeat_interleave(w.shape[0] // m.n_V, dim=0)
            codes = (rows / scale_rows).round().clamp(-m.w_qmax, m.w_qmax - 1) + m.w_qmax
            entries[n + ".weight"] = {"q": codes.to(torch.uint8).reshape(w.shape).cpu(), "scale": interval.cpu(),
                                      "zero": torch.full((m.n_V, 1), float(m.w_qmax))}
            ref[n + ".weight"] = sim.cpu()
    packable = {}
    if published:   # the activation intervals the point needs at inference
        packable.update({k: v.float().reshape(-1).cpu() for k, v in sd.items() if "interval" in k})
        for n, m in net.named_modules():
            for attr in ("a_interval", "A_interval", "B_interval", "n_G_A"):
                v = getattr(m, attr, None)
                if isinstance(v, torch.Tensor):
                    packable[f"{n}.{attr}"] = v.float().reshape(-1).cpu()
    for k in deit_small().state_dict():
        packable[k] = entries[k] if k in entries else sd[k].float().half().cpu()
    out = REPO / "outputs/artifacts" / f"{tag}.safetensors"
    size = save_packed_rows(packable, out, args.w, method="ptq4vit")
    back = load_packed_rows(out)
    bad = [k for k in ref if not torch.equal(back[k], ref[k])]
    if bad:
        raise SystemExit(f"round trip failed on {len(bad)} layers, e.g. {bad[:3]}")
    print(f"{out.name}: {size / 1e6:.3f} MB, {len(ref)} quantized layers identical to the method's weights", flush=True)

    record = {"run": "as published" if published else "weights-only", "w_bit": args.w, "a_bit": args.a, "conv_a": 32,
              "calib_size": args.calib, "file": str(out.relative_to(REPO)), "calibration_s": seconds,
              "peak_gpu_mib_after_calibration": peak}
    with torch.no_grad():
        if published:
            score_in_process(net, tfm, name, REPO / "outputs/published_summary.json", REPO / "outputs/published_per_image.npz", record)
        else:
            score_in_process(net, tfm, f"{name}, in the method's forward", REPO / "outputs/validation_summary.json", None, record)


if __name__ == "__main__":
    main()
