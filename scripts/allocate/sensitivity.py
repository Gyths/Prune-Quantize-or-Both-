"""Per-layer sensitivity of DeiT-S to quantizing and to pruning, in `outputs/sensitivity_summary.json`.

    python scripts/allocate/sensitivity.py quant    # one layer at a time at low bits: round-to-nearest and each method's packed file
    python scripts/allocate/sensitivity.py prune    # one block's MLP or attention at a time: CORP, GRAIL and VBP
    python scripts/allocate/sensitivity.py all

Only one unit changes and the rest of the model stays FP32, so the drop belongs to that unit. Quantization has 50 units
(patch embedding, 48 block linears, head). Round-to-nearest uses `validate_low_bits.rtn_rows`; a published method
contributes the weights of that layer as stored in its packed file, so its heatmap shows what the method did to the
layer after seeing the whole model. Pruning has two units per block, the MLP hidden channels and, for CORP, the Q and
K dimensions; `proj` is pruned by no method. VBP uses its authors' code with the quantile taken inside the block,
since its global quantile has no meaning for one block.

Each configuration reports the KL divergence from FP32, the logit MSE and the top-1 change, on two image sets: the
first 2 000 calibration images, the only ones an allocation may look at, and every tenth val image (5 per class), to
check that the calibration ranking holds on held-out data. The script resumes: configurations already in the summary
are skipped.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from deitc.evaluate import DATA, WNID_TO_INDEX, by_wnid
from deitc.model import load_pretrained
from deitc.packing import dequantize_rows, load_packed_rows
from deitc.pruning import CalibImages, _attn_systems, _keep, _prune_attn, _prune_mlp, corp_collect
from deitc.pruning import THIRD, loader, run_grail, test_transform
from deitc.quant import adalog_layer, rtn_rows

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")
PATH = OUT / "sensitivity_summary.json"
RTN_BITS = (2, 3, 4, 6, 8)
FILES = {f"{m} W{b}": OUT / "artifacts" / f"{m.lower()}_w{b}.safetensors" for m, bits in
         (("AdaLog", (3, 4, 6)), ("FIMAQ", (2, 3, 4, 6)), ("APHQ", (3, 4))) for b in bits}
SPARSITIES = (0.1, 0.3, 0.5, 0.7)
GRAIL_N, CORP_N, VBP_N = 4000, 4000, 4000


class Sets:
    """The two image sets, kept in CPU memory, with the logits of FP32 on them."""
    def __init__(self, model, calib=2000, val_stride=10):
        tfm = test_transform()
        names = Path("splits/calib_seed0.txt").read_text().split()[:calib]
        c = CalibImages(calib, tfm)
        v = torch.utils.data.Subset(by_wnid(DATA["ImageNet-1k val"], tfm), range(0, 50000, val_stride))
        self.sets = {}
        for name, ds, labels in (("calib", c, torch.tensor([WNID_TO_INDEX[n.split("/")[0]] for n in names])), ("val", v, None)):
            xs, ys = [], []
            for x, y in torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8):
                xs.append(x); ys.append(y.clone())
            self.sets[name] = (torch.cat(xs), labels if labels is not None else torch.cat(ys))
        self.ref = {k: self.logits(model, k) for k in self.sets}

    @torch.no_grad()
    def logits(self, model, name):
        x = self.sets[name][0]
        return torch.cat([model(x[i:i + 256].cuda()).float().cpu() for i in range(0, len(x), 256)])

    def measure(self, model):
        out = {}
        for k, (_, y) in self.sets.items():
            lg, ref = self.logits(model, k), self.ref[k]
            logp, refp = lg.log_softmax(1), ref.log_softmax(1)
            out[k] = {"kl": (refp.exp() * (refp - logp)).sum(1).mean().item(), "logit_mse": (lg - ref).pow(2).mean().item(),
                      "top1": (lg.argmax(1) == y).float().mean().item() * 100,
                      "top1_change": ((lg.argmax(1) == y).float().mean() - (ref.argmax(1) == y).float().mean()).item() * 100}
        return out


class Book:
    def __init__(self, sets):
        self.sets = sets
        self.data = json.load(open(PATH)) if PATH.exists() else {"reference": {
            k: (sets.ref[k].argmax(1) == sets.sets[k][1]).float().mean().item() * 100 for k in sets.sets}, "points": {}}

    def done(self, method, setting, unit):
        return unit in self.data["points"].get(method, {}).get(setting, {})

    def add(self, method, setting, unit, model, **extra):
        self.data["points"].setdefault(method, {}).setdefault(setting, {})[unit] = {**extra, **self.sets.measure(model)}
        json.dump(self.data, open(PATH, "w"), indent=1)
        r = self.data["points"][method][setting][unit]
        print(f"{method:<16} {setting:<6} {unit:<28} KL {r['calib']['kl']:.4f}  top-1 {r['calib']['top1_change']:+.2f} / {r['val']['top1_change']:+.2f}", flush=True)


@torch.no_grad()
def quant(model, book):
    layers = [k for k, v in model.state_dict().items() if adalog_layer(k, v)]
    assert len(layers) == 50, len(layers)
    params = dict(model.named_parameters())
    def swap(method, setting, k, w):
        if book.done(method, setting, k[:-7]):
            return
        old = params[k].data.clone()
        params[k].data.copy_(w)
        book.add(method, setting, k[:-7], model)
        params[k].data.copy_(old)
    for b in RTN_BITS:
        for k in layers:
            e = rtn_rows(params[k].data.float().cpu(), b)
            swap("RTN", f"W{b}", k, dequantize_rows(e, e["q"].shape).cuda())
    for name, f in FILES.items():
        m, b = name.split()
        sd = load_packed_rows(f)
        for k in layers:
            swap(m, b, k, sd[k].cuda())


@torch.no_grad()
def prune(model, book):
    n_blocks = len(model.blocks)
    # CORP, one unit at a time, on statistics from the unpruned model
    data = loader(CORP_N, test_transform(), 64)
    stats = corp_collect(model, data)
    for s in SPARSITIES:
        setting = f"{round(s * 100)} %"
        for i in range(n_blocks):
            if not book.done("CORP", setting, f"blocks.{i}.mlp"):
                old, model.blocks[i] = model.blocks[i], copy.deepcopy(model.blocks[i])
                _prune_mlp(model.blocks[i], stats[("mlp", i)], s)
                book.add("CORP", setting, f"blocks.{i}.mlp", model, params_removed=sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in model.blocks[i].parameters()))
                model.blocks[i] = old
        if all(book.done("CORP", setting, f"blocks.{i}.attn") for i in range(n_blocks)):
            continue
        keeps = {}
        for i, blk in enumerate(model.blocks):
            energy = stats[("attn", i)]["energy"]
            keeps[i] = []
            for h in range(blk.attn.num_heads):
                keep = _keep(energy[h], s)
                drop = torch.ones(energy.shape[1], dtype=torch.bool, device=energy.device); drop[keep] = False
                keeps[i].append((keep, drop.nonzero().squeeze(1)))
        systems = _attn_systems(model, data, keeps, "cuda")
        for i in range(n_blocks):
            if not book.done("CORP", setting, f"blocks.{i}.attn"):
                old, model.blocks[i] = model.blocks[i], copy.deepcopy(model.blocks[i])
                _prune_attn(model.blocks[i], i, keeps, systems)
                book.add("CORP", setting, f"blocks.{i}.attn", model, params_removed=sum(p.numel() for p in old.parameters()) - sum(p.numel() for p in model.blocks[i].parameters()))
                model.blocks[i] = old
        del systems
        torch.cuda.empty_cache()
    # GRAIL: the authors' pipeline with its module map holding one block
    for label, method in (("GRAIL Wanda", "wanda"), ("GRAIL magnitude", "mag-l2")):
        for s in SPARSITIES:
            setting = f"{round(s * 100)} %"
            for i in range(n_blocks):
                if not book.done(label, setting, f"blocks.{i}.mlp"):
                    with torch.enable_grad():
                        pruned = run_grail(copy.deepcopy(model), method, s, GRAIL_N, True, blocks=[i])
                    book.add(label, setting, f"blocks.{i}.mlp", pruned.eval())
                    del pruned
    # VBP: the authors' score and prune, with the quantile taken inside one block
    import sys
    sys.path.insert(0, str(THIRD / "VBP/src"))
    from pruner import indices, prune as vbp_prune, score as vbp_score, threshold
    from torchvision import transforms
    tfm = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    args = SimpleNamespace(model="DeiT", device="cuda", num_batches=0, percentage=0.5)
    torch.manual_seed(0)
    variances, means = vbp_score(args, model, loader(VBP_N, tfm, 32))
    for s in SPARSITIES:
        setting = f"{round(s * 100)} %"
        args.percentage = s
        for i in range(n_blocks):
            if book.done("VBP", setting, f"blocks.{i}.mlp"):
                continue
            k = f"0.{i}.hidden"
            old, model.blocks[i] = model.blocks[i], copy.deepcopy(model.blocks[i])
            one = {k: variances[k]}
            vbp_prune(args, model, indices(args, one, threshold(args, one)), means)
            book.add("VBP", setting, f"blocks.{i}.mlp", model)
            model.blocks[i] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["quant", "prune", "all"])
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    model = load_pretrained("DeiT-S").cuda().eval()
    book = Book(Sets(model))
    if args.what in ("quant", "all"):
        quant(model, book)
    if args.what in ("prune", "all"):
        prune(model, book)


if __name__ == "__main__":
    main()
