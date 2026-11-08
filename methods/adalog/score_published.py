"""AdaLog as published (weights and activations quantized), scored on our three test sets.

    cd <AdaLog dir> && $ADALOG_PY <repo>/methods/adalog/score_published.py --config ./configs/4bit.py \
        --dataset <imagenet> --calib-list <repo>/splits/calib_seed0.txt --name "AdaLog W4A4"

The activations are quantized inside AdaLog's own forward pass, so a packed weight file cannot reproduce this
point and `scripts/quantize/evaluate_file.py` cannot score it. This script therefore builds and calibrates the model exactly as
the authors' `test_quant.py --calibrate` does (same seed, same wrapping, same calibrator, same reparameterization),
then scores it with our data on ImageNet-1k val, ImageNet-V2 and ImageNet-R, writing the same per-image results
and summary as `scripts/quantize/evaluate_file.py`. It also records the wall-clock time and peak GPU memory of the calibration.
The checkpoint it saves is the one `scripts/quantize/pack_published.py` turns into the file that sizes the point.

With `--load` it skips calibration and restores a checkpoint the way `test_quant.py --load-calibrate-checkpoint`
does; `--check-first` then reports the accuracy over the first images of val, to compare with a partial log.
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_data_config, create_transform

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [os.getcwd(), str(REPO)]          # the authors' modules (cwd) and our deitc

from utils.calibrator import QuantCalibrator                                   # noqa: E402
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net   # noqa: E402
import utils.datasets as mydatasets                                            # noqa: E402
from deitc.data import WNID_TO_INDEX, imagenet_v2, by_wnid                     # noqa: E402
from deitc.evaluate import DATA, bootstrap_ci, paired_diff_ci                  # noqa: E402

OUT = REPO / "outputs"
DATA = {k: v if v.is_absolute() else (REPO / v).resolve() for k, v in DATA.items()}   # we run from the AdaLog dir


def seed_all(seed):                               # test_quant.py, same order of calls
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def restore(model, path):                         # test_quant.load_model, mode='calibrate'
    for _, m in model.named_modules():
        if hasattr(m, "mode"):
            m.calibrated = True; m.mode = "quant_forward"
        for attr in ("a_quantizer", "w_quantizer", "A_quantizer", "B_quantizer"):
            if hasattr(m, attr):
                getattr(m, attr).inited = True
    print(model.load_state_dict(torch.load(path), strict=False))
    return model


@torch.no_grad()
def predict(model, ds, device, limit=None):
    loader = torch.utils.data.DataLoader(ds, batch_size=200, num_workers=8, pin_memory=False)  # see deitc.evaluate.test_loader
    logits, labels, seen = [], [], 0
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu()); labels.append(y.clone())
        seen += len(y)
        if limit and seen >= limit:
            break
    return torch.cat(logits), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib-list", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--load", default=None)
    ap.add_argument("--check-first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    sys.path.append(os.path.dirname(os.path.abspath(args.config)))
    cfg = importlib.import_module(Path(args.config).stem).Config()
    print({k: v for k, v in vars(cfg).items()})
    device = torch.device("cuda")
    seed_all(args.seed)
    name = "deit_small_patch16_224"
    model = timm.create_model(name, checkpoint_path=f"./checkpoints/vit_raw/{name}.bin").to(device).eval()
    g = mydatasets.ViTImageNetLoaderGenerator(args.dataset, 200, 8, kwargs={"model": model})
    tfm = create_transform(**resolve_data_config({}, model=model))

    record = {"config": args.config, "w_bit": cfg.w_bit, "a_bit": cfg.a_bit, "s_bit": cfg.s_bit,
              "calib_size": cfg.calib_size}
    model = wrap_modules_in_net(model, cfg, reparam=args.load is None).to(device).eval()
    if args.load:
        model = restore(model, args.load).to(device).eval()
        record["checkpoint"] = args.load
    else:
        calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed,
                                      calib_list=args.calib_list)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        QuantCalibrator(model, calib_loader).batching_quant_calib()
        model = wrap_reparamed_modules_in_net(model).to(device)
        for _, m in model.named_modules():        # test_quant.finish_training
            if hasattr(m, "mode") and hasattr(m, "reparam_bias"):
                m.reparam_bias()
        torch.cuda.synchronize()
        record["calib_seconds"] = time.time() - t0
        record["calib_peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
        ckdir = Path("checkpoints/quant_result/protocol"); ckdir.mkdir(parents=True, exist_ok=True)
        ck = ckdir / f"deit_small_w{cfg.w_bit}_a{cfg.a_bit}_s{cfg.s_bit}_calibsize_{cfg.calib_size}.pth"
        torch.save(model.state_dict(), ck)
        record["checkpoint"] = str(Path.cwd() / ck)
        print(f"calibrated in {record['calib_seconds']:.0f} s, peak {record['calib_peak_gpu_mb']:.0f} MiB -> {ck}")
    model.eval()

    if args.check_first:
        lg, y = predict(model, by_wnid(DATA["ImageNet-1k val"], tfm), device, args.check_first)
        n = min(args.check_first, len(y))
        print(f"first {n} val images: top-1 {(lg[:n].argmax(1) == y[:n]).float().mean() * 100:.3f} %")
        return

    base = np.load(OUT / "baselines_per_image.npz")
    res, correct = {}, {}
    for dname, root in DATA.items():
        ds = imagenet_v2(root, tfm) if dname == "ImageNet-V2" else by_wnid(root, tfm)
        lg, labels = predict(model, ds, device)
        if dname == "ImageNet-R":   # scored over R's 200 classes, as in evaluate.score
            classes = sorted({WNID_TO_INDEX[p.name] for p in root.iterdir() if p.is_dir()})
            mask = torch.full((1000,), float("-inf")); mask[classes] = 0
            lg = lg + mask
        c = (lg.argmax(1) == labels).numpy().astype(float)
        diff, (dlo, dhi) = paired_diff_ci(c, base[f"{dname}|DeiT-S FP32"].astype(float))
        res[dname] = {"top1": c.mean() * 100, "ci95": list(bootstrap_ci(c)), "vs_fp32": {"diff": diff, "ci95": [dlo, dhi]}}
        correct[dname] = c.astype(np.uint8)
        print(f"  {dname:<16} top-1 {res[dname]['top1']:6.2f} %  vs FP32 {diff:+.2f} [{dlo:+.2f}, {dhi:+.2f}]")

    summary_path = OUT / "published_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary[args.name] = {"run": "as published", **record, "accuracy": res}
    json.dump(summary, open(summary_path, "w"), indent=2, default=float)
    per_image = OUT / "published_per_image.npz"
    old = dict(np.load(per_image)) if per_image.exists() else {}
    old.update({f"{d}|{args.name}": c for d, c in correct.items()})
    np.savez_compressed(per_image, **old)


if __name__ == "__main__":
    main()
