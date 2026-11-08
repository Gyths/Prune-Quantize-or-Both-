"""MACs and BOPs of every point on the chart, next to its file size, written to `outputs/costs_summary.json`.

    python scripts/measure/measure_costs.py

MACs come from `deitc.costs.count_macs` and must land on the published 4.6 G for DeiT-S and 1.3 G for DeiT-Ti,
which checks the counter. BOPs follow the convention in `deitc/costs.py`. Each point names the bit assignment of
the run it comes from, so a method's as-published and weights-only runs get different BOPs for the same MACs.

Each point also says whether its activations are quantized. Where they are (the as-published runs, W8A8), BOPs
estimate the arithmetic on hardware that computes natively at those widths. Where they are not (the weights-only
runs and the dense baselines), the activations count as 32 bits, so the BOPs are those of a consumer GPU that
keeps activations in float and serve as a comparison, with FP32 and FP16 as the references.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import json
import re
from pathlib import Path

from safetensors.torch import load_file

from deitc import costs
from deitc.model import deit_small, deit_tiny, from_state_dict

OUT = Path("outputs")


def points():
    base = json.load(open(OUT / "baselines_summary.json"))["size_mb"]
    pub = json.load(open(OUT / "published_summary.json"))
    for name, size in base.items():
        model, fmt = name.split()
        bits = {"FP32": costs.bits_float(32), "FP16": costs.bits_float(16),
                "INT8": costs.bits_rtn(8), "INT4": costs.bits_rtn(4)}[fmt]
        yield name, model, size, bits, "dense baseline", "FP16" if fmt == "FP16" else "full precision"
    for name, r in pub.items():
        if r.get("run") == "as published":
            # the AdaLog family wraps the same places; configs without s_bit quantize the softmax output at a_bit
            f = Path(r["file"]) if r.get("file") else None
            size = f.stat().st_size / 1e6 if f and f.exists() else None
            yield name, "DeiT-S", size, costs.bits_adalog_published(
                r["w_bit"], r["a_bit"], r.get("s_bit", r["a_bit"]), conv_a=r.get("conv_a", 8), head_a=r["a_bit"]), "as published", "quantized"
        else:
            yield (name, "DeiT-S", r["size_mb"], costs.bits_weights_only(int(re.search(r"W(\d+)", name).group(1))), "weights-only",
                   "full precision")
    for model in ("DeiT-S", "DeiT-Ti"):   # the format only; its accuracy and file come with the W8A8 series
        yield f"{model} W8A8 torchao", model, None, costs.bits_w8a8_linears(), "format only", "quantized (linears)"


def main():
    macs = {"DeiT-S": costs.count_macs(deit_small()), "DeiT-Ti": costs.count_macs(deit_tiny())}
    for m, ref in (("DeiT-S", 4.6), ("DeiT-Ti", 1.3)):
        g = sum(macs[m].values()) / 1e9
        assert abs(g - ref) < 0.05, f"{m}: {g:.3f} GMACs, published {ref}"
        print(f"{m}: {g:.3f} GMACs (published {ref})")
    out = {"convention": "BOPs = MACs x bits of both operands; activations at full precision count as 32 bits, "
                         "so points with float activations give consumer-GPU BOPs, kept as a comparison",
           "macs_per_layer": macs, "points": {}}
    for name, model, size, bits, run, acts in points():
        out["points"][name] = {"model": model, "run": run, "activations": acts, "size_mb": size,
                               "gmacs": sum(macs[model].values()) / 1e9, "gbops": costs.bops(macs[model], bits) / 1e9}
        p = out["points"][name]
        print(f"  {name:<20} {run:<15} {acts:<20} {p['size_mb'] or float('nan'):7.2f} MB  {p['gmacs']:.3f} GMACs"
              f"  {p['gbops']:8.1f} GBOPs")
    # Pruned models, each counted from its own layers. They run in FP16 like the dense FP16 baseline.
    for tag, r in json.load(open(OUT / "pruned_summary.json")).items():
        sd = {k: v.float() for k, v in load_file(r["file"]).items()}
        m = costs.count_macs(from_state_dict(sd))
        out["points"][f"Pruned {tag}"] = {"model": "DeiT-S", "run": "pruned", "activations": "FP16", "size_mb": r["size_mb"],
                                          "gmacs": sum(m.values()) / 1e9, "gbops": costs.bops(m, costs.bits_float(16)) / 1e9,
                                          "compress_seconds": r["calib_seconds"]}
        print(f"  {tag:<26} {r['size_mb']:7.2f} MB  {out['points'][f'Pruned {tag}']['gmacs']:.3f} GMACs"
              f"  {out['points'][f'Pruned {tag}']['gbops']:8.1f} GBOPs")
    json.dump(out, open(OUT / "costs_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
