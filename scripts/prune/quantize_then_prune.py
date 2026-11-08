"""The check of compression order: FIMA-Q W3 first, then CORP joint, against the same steps the other way round.

    python scripts/prune/quantize_then_prune.py 20 30

CORP's statistics come from the quantized model. Its correction leaves float weights in the layers it changes, so those
layers are rounded again at 3 bits, each row on a grid over its own range. Layers it leaves alone keep FIMA-Q's codes. Our
existing prune-then-quantize points get a fresh FIMA-Q search on the pruned weights, which this order does not, so a win for
this order is the stronger result and a loss is weak evidence. Results go to `outputs/order_summary.json`.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import sys
from pathlib import Path

import numpy as np
import torch

from deitc.evaluate import score
from deitc.model import from_state_dict
from deitc.packing import load_entries, load_packed_rows, save_packed_rows
from deitc.pruning import corp, corp_collect, loader, test_transform
from deitc.quant import round_rows
from deitc.results import add_point, fp32_reference

OUT = Path("outputs")
BITS = 3


def main():
    torch.manual_seed(0); np.random.seed(0)
    source = OUT / "artifacts" / "fimaq_w3.safetensors"
    entries, raw, _ = load_entries(source)
    sd = load_packed_rows(source)
    model = from_state_dict(sd).cuda().eval()
    data = loader(4000, test_transform(), 64)
    stats = corp_collect(model, data)
    for pct in map(int, sys.argv[1:]):
        pruned = corp(model, stats, data, pct / 100, pct / 100)
        new = {k: v.detach().cpu() for k, v in pruned.state_dict().items()}
        packable = {}
        changed = 0
        for k, v in new.items():
            if k in entries and entries[k]["q"].shape == v.shape and torch.equal(v, sd[k].to(v.dtype)):
                packable[k] = entries[k]
            elif k in entries:
                packable[k] = round_rows(v.float())
                changed += 1
            else:
                packable[k] = v.half()
        path = OUT / "artifacts" / f"fimaq_w3_then_corp_s{pct}.safetensors"
        save_packed_rows(packable, path, {k: BITS for k, v in packable.items() if isinstance(v, dict)}, method="fimaq then corp")
        res, _ = score(load_packed_rows(path), fp32_reference())
        name = f"FIMA-Q W3 then CORP joint {pct} %"
        add_point(OUT / "order_summary.json", name,
                  {"file": str(path), "size_mb": path.stat().st_size / 1e6, "layers_rounded_again": changed, "accuracy": res})
        print(name, f"{path.stat().st_size / 1e6:.2f} MB", {d.split('-')[-1]: round(r["top1"], 2) for d, r in res.items()}, flush=True)
        del pruned
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
