"""Which part of FIMA-Q is fragile: its weights, its activations, or one activation point at some depth.

    $ADALOG_PY methods/fimaq/fragility.py --config ./configs/3bit/best.py --ckpt PATH --tag W3A3 --tests 1 2 3

Run from the repository root in the method's environment. The script moves into FIMA-Q's folder (`FIMAQ_DIR`, by default
`../third_party/FIMA-Q-protocol`), so `--config` is relative to it and the checkpoint is one of its calibration
checkpoints. Tests 1 to 4 switch quantizers off in a finished checkpoint: a UniformQuantizer at 32 bits
returns its input, so nothing is searched again and the scales stay those found for all parts together. Test 5 changes one
point's width and searches that point's scale again, by mean squared error over percentile ranges on the first 128
calibration images, with everything else held as it is. That search is ours and simpler than the method's, so test 5 also
runs A3 through it, which measures what the simpler search costs against the checkpoint's own scale. Expectations are in
`ABLATIONS.md`. Results go to `outputs/fragility_{tag}_{tests}.json`, and the per-image results to a matching npz.
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import timm
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FIMAQ_DIR", REPO.parent / "third_party" / "FIMA-Q-protocol")).resolve()
DATA_DIR = os.environ.get("IMAGENET_DIR", str(REPO.parent / "data" / "imagenet"))
os.chdir(WORK)   # the authors' code reads its configs and the raw DeiT weights from paths relative to its folder
sys.path.insert(0, str(WORK))
import utils.datasets as mydatasets  # noqa: E402
from utils.wrap_net import wrap_modules_in_net  # noqa: E402

sys.path.insert(0, str(REPO))
from deitc.evaluate import score_in_process  # noqa: E402

IMAGE = "image"
HEAD = "head input"
IN_BLOCK = {"qkv input": ("attn.qkv", "a_quantizer"), "Q": ("attn.matmul1", "A_quantizer"),
            "K": ("attn.matmul1", "B_quantizer"), "attention scores": ("attn.matmul2", "A_quantizer"),
            "V": ("attn.matmul2", "B_quantizer"), "proj input": ("attn.proj", "a_quantizer"),
            "fc1 input": ("mlp.fc1", "a_quantizer"), "fc2 input": ("mlp.fc2", "a_quantizer")}
OUTSIDE = {IMAGE: ("patch_embed.proj", "a_quantizer"), HEAD: ("head", "a_quantizer")}
DEPTHS = (0, 6, 11)
WIDTHS = (3, 4, 6, 8)


def load(config, ckpt, data):
    """The model as the authors' test_quant.py restores an optimized checkpoint."""
    sys.path.append(os.path.dirname(os.path.abspath(config)))
    cfg = importlib.import_module(os.path.splitext(os.path.basename(config))[0]).Config()
    model = timm.create_model("deit_small_patch16_224", checkpoint_path="./checkpoints/vit_raw/deit_small_patch16_224.bin")
    model.cuda().eval()
    gen = mydatasets.ViTImageNetLoaderGenerator(data, 200, 8, kwargs={"model": model})
    model = wrap_modules_in_net(model, cfg, reparam=False).cuda().eval()
    state = torch.load(ckpt)
    for name, module in model.named_modules():
        if hasattr(module, "mode"):
            module.calibrated = True
            module.mode = "quant_forward"
        for attr in ("a_quantizer", "w_quantizer", "A_quantizer", "B_quantizer"):
            if hasattr(module, attr):
                getattr(module, attr).inited = True
                getattr(module, attr).scale.data = state[f"{name}.{attr}.scale"].clone()
    print(model.load_state_dict(state, strict=False), flush=True)
    return model.cuda().eval(), gen


class Switches:
    """Every quantizer of the model, by name, with the widths the checkpoint gave them."""

    def __init__(self, model):
        self.model = model
        self.weights, self.activations = {}, {}
        for name, module in model.named_modules():
            if hasattr(module, "w_quantizer"):
                self.weights[f"{name}.w_quantizer"] = module.w_quantizer
            for attr in ("a_quantizer", "A_quantizer", "B_quantizer"):
                if hasattr(module, attr):
                    self.activations[f"{name}.{attr}"] = getattr(module, attr)
        self.saved = {k: self.save(q) for k, q in {**self.weights, **self.activations}.items()}

    @staticmethod
    def save(q):
        return {"bits": q.n_bits, "levels": q.n_levels, "scale": q.scale.data.clone(),
                "zero": q.zero_point.data.clone() if hasattr(q, "zero_point") else None}

    def restore(self):
        for k, q in {**self.weights, **self.activations}.items():
            s = self.saved[k]
            q.n_bits, q.n_levels = s["bits"], s["levels"]
            q.scale.data = s["scale"].clone()
            if s["zero"] is not None:
                q.zero_point.data = s["zero"].clone()

    def off(self, names):
        for k in names:
            {**self.weights, **self.activations}[k].n_bits = 32


def point(name, block=None):
    """The quantizers of an activation point: in one block, in every block, or the one outside the blocks."""
    if name in OUTSIDE:
        mod, attr = OUTSIDE[name]
        return [f"{mod}.{attr}"]
    mod, attr = IN_BLOCK[name]
    return [f"blocks.{b}.{mod}.{attr}" for b in (range(12) if block is None else [block])]


def capture(model, sw, names, calib):
    """The tensor each named quantizer sees on the calibration images, with those quantizers off."""
    seen, hooks = {k: [] for k in names}, []
    sw.off(names)
    for k in names:
        mod_name, _, attr = k.rpartition(".")
        if mod_name == OUTSIDE[IMAGE][0]:   # the patch embedding skips its quantizer when it is off, so read the module
            hooks.append(model.get_submodule(mod_name).register_forward_pre_hook(
                lambda m, a, key=k: seen[key].append(a[0].detach().cpu())))
        else:
            hooks.append(sw.activations[k].register_forward_pre_hook(
                lambda m, a, key=k: seen[key].append(a[0].detach().cpu())))
    with torch.no_grad():
        for x, _ in calib:
            model(x.cuda())
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in seen.items()}


def search_scale(q, x, bits, per_group=200_000, candidates=32):
    """Set q to `bits` with the percentile range that gives the lowest mean squared error, per scale group."""
    # An asymmetric quantizer's calibrated scale repeats its values at the granularity of its zero point (one value for an
    # MLP input, one per head for a matmul), whatever the size of the scale tensor, so that is the granularity to fit.
    full = q.scale.data.shape
    shape = q.zero_point.data.shape if hasattr(q, "zero_point") and not q.sym else full
    aligned = shape + (1,) * (x.dim() - len(shape))
    groups = [d for d in range(x.dim()) if aligned[d] > 1]
    rest = [d for d in range(x.dim()) if d not in groups]
    g = x.permute(*groups, *rest).reshape(int(torch.tensor([aligned[d] for d in groups]).prod()) if groups else 1, -1).cuda()
    if g.shape[1] > per_group:
        g = g[:, torch.randperm(g.shape[1], device=g.device)[:per_group]]
    ordered = g.sort(dim=1).values
    n = ordered.shape[1] - 1
    levels = 2 ** bits
    best, best_scale, best_zero = None, None, None
    for r in torch.linspace(0.99, 0.99999, candidates).tolist():
        hi, lo = ordered[:, int(r * n)], ordered[:, int((1 - r) * n)]
        if q.sym:
            scale = (torch.max(hi.abs(), lo.abs()) / (levels / 2 - 0.5)).clamp_min(1e-8)
            zero = torch.zeros_like(scale)
            sim = (g / scale[:, None]).round().clamp(-levels / 2, levels / 2 - 1) * scale[:, None]
        else:
            scale = ((hi - lo) / (levels - 1)).clamp_min(1e-8)
            zero = (-lo / scale).round()
            sim = ((g / scale[:, None]).round() + zero[:, None]).clamp(0, levels - 1)
            sim = (sim - zero[:, None]) * scale[:, None]
        err = ((g - sim) ** 2).mean(1)
        if best is None:
            best, best_scale, best_zero = err, scale, zero
        else:
            better = err < best
            best, best_scale, best_zero = (torch.where(better, err, best), torch.where(better, scale, best_scale),
                                           torch.where(better, zero, best_zero))
    q.n_bits, q.n_levels = bits, 2 ** (bits - 1)
    q.scale.data = best_scale.reshape(shape).expand(full).clone().to(q.scale.data.dtype)
    if hasattr(q, "zero_point") and not q.sym:
        q.zero_point.data = best_zero.reshape(q.zero_point.data.shape).to(q.zero_point.data.dtype)


def configurations(sw, tests, only=None):
    """(label, what to switch off, or None and a width change). Test 1 also holds the checks of the switches."""
    every_a, every_w = list(sw.activations), list(sw.weights)
    points = list(IN_BLOCK) + [HEAD]
    if 1 in tests:
        yield "base", [], None
        yield "everything off", every_a + every_w, None
        yield "weights only", every_a, None
        yield "activations only", every_w, None
    if 2 in tests:
        for p in points:
            yield f"remove | {p}", point(p), None
    if 3 in tests:
        for p in points:
            keep = set(point(p))
            yield f"keep | {p}", every_w + [k for k in every_a if k not in keep], None
    if 4 in tests:
        for p in IN_BLOCK:
            for b in DEPTHS:
                yield f"remove | {p} | block {b}", point(p, b), None
                keep = set(point(p, b))
                yield f"keep | {p} | block {b}", every_w + [k for k in every_a if k not in keep], None
    if 5 in tests:
        for p in (only or points + [IMAGE]):
            for bits in WIDTHS:
                if p == IMAGE and bits >= 8:   # the patch embedding does not quantize its input at 8 bits or more
                    continue
                yield f"width | {p} | A{bits}", [], (p, bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="the run, such as W3A3")
    ap.add_argument("--tests", type=int, nargs="+", required=True)
    ap.add_argument("--points", nargs="+", default=None, help="test 5 only: the activation points to take through the widths")
    ap.add_argument("--published", default=None, help="the name of this run in published_summary.json, to check against")
    args = ap.parse_args()

    tests = "".join(map(str, args.tests))
    out, per_image = REPO / "outputs" / f"fragility_{args.tag}_{tests}.json", REPO / "outputs" / f"fragility_{args.tag}_{tests}.npz"
    done = json.load(open(out)) if out.exists() else {}
    model, gen = load(args.config, args.ckpt, DATA_DIR)
    sw = Switches(model)
    calib = None
    for label, off, change in configurations(sw, set(args.tests), args.points):
        name = f"{args.tag} | {label}"
        if name in done:
            continue
        start = time.time()
        sw.restore()
        sw.off(off)
        if change:
            calib = calib or gen.calib_loader(num=128, batch_size=32, seed=3407, calib_list=str(REPO / "splits" / "calib_seed0.txt"))
            p, bits = change
            names = point(p)
            seen = capture(model, sw, names, calib)
            for k in names:
                search_scale(sw.activations[k], seen[k], bits)
            del seen
            torch.cuda.empty_cache()
        searched = time.time() - start
        acc = score_in_process(model, gen.val_transform, name, out, per_image,
                               {"tag": args.tag, "label": label, "search_s": searched}, "cuda")
        val = acc["ImageNet-1k val"]
        print(f"{name}: val {val['top1']:.2f} in {time.time() - start:.0f} s", flush=True)
        if label == "base" and args.published:
            want = json.load(open(REPO / "outputs" / "published_summary.json"))[args.published]["accuracy"]["ImageNet-1k val"]["top1"]
            if abs(val["top1"] - want) > 0.03:
                raise SystemExit(f"the checkpoint gives {val['top1']:.2f}, the published run gave {want:.2f}")
        if label == "everything off" and abs(val["vs_fp32"]["diff"]) > 0.1:
            raise SystemExit(f"with every quantizer off the model should be FP32, and differs by {val['vs_fp32']['diff']:+.2f}")
    sw.restore()


if __name__ == "__main__":
    main()
