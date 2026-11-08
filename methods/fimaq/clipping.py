"""How many activations fall outside the quantizer's range at each point, on val, V2 and R.

    $ADALOG_PY methods/fimaq/clipping.py --config ./configs/3bit/best.py --ckpt PATH --tag W3A3

Run from the repository root in the method's environment, like `fragility.py`, whose loader it uses.

The scales are fitted on ImageNet training images, so a point whose clipping grows on R is one that distribution shift
pushes out of range. This reads a finished checkpoint as it stands, with nothing switched off, on about 2 000 images per
set. Results go to `outputs/clipping_{tag}.json`.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fragility import DATA_DIR, IN_BLOCK, OUTSIDE, REPO, Switches, load  # noqa: E402
sys.path.insert(0, str(REPO))
from deitc.evaluate import DATA, by_wnid, imagenet_v2, test_loader  # noqa: E402


def outside(q, x):
    if q.n_bits >= 32:
        return 0, x.numel()
    s = q.scale
    if q.sym:
        lo, hi = -q.n_levels * s, (q.n_levels - 1) * s
    else:
        z = q.zero_point.round()
        lo, hi = -z * s, (2 * q.n_levels - 1 - z) * s
    return ((x < lo) | (x > hi)).sum().item(), x.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    model, gen = load(args.config, args.ckpt, DATA_DIR)
    sw = Switches(model)
    where = {f"blocks.{b}.{m}.{a}": p for p, (m, a) in IN_BLOCK.items() for b in range(12)}
    where.update({f"{m}.{a}": p for p, (m, a) in OUTSIDE.items() if p != "image"})
    out = {}
    for dname, root in DATA.items():
        ds = imagenet_v2(root, gen.val_transform) if dname == "ImageNet-V2" else by_wnid(root, gen.val_transform)
        step = max(1, len(ds) // 2000)
        loader = test_loader(torch.utils.data.Subset(ds, range(0, len(ds), step)), 100)
        counts = {p: [0, 0] for p in where.values()}
        hooks = []
        for k, p in where.items():
            def hook(q, a, point=p):
                n, total = outside(q, a[0])
                counts[point][0] += n
                counts[point][1] += total
            hooks.append(sw.activations[k].register_forward_pre_hook(hook))
        with torch.no_grad():
            for x, _ in loader:
                model(x.cuda())
        for h in hooks:
            h.remove()
        out[dname] = {p: c[0] / max(1, c[1]) for p, c in counts.items()}
        print(dname, {p: round(v * 100, 2) for p, v in out[dname].items()}, flush=True)
    json.dump(out, open(REPO / "outputs" / f"clipping_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
