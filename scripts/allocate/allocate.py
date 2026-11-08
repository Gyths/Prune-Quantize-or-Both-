"""Our method: allocate bits and pruning per block under a target file size, from the sensitivity test.

    python scripts/allocate/allocate.py plan q7.4 --curve q --target-mb 7.4      # write the allocation, nothing is run
    python scripts/allocate/allocate.py build q7.4 p35 both7.4                   # prune what needs pruning (FP16 and FP32 files)
    python scripts/allocate/allocate.py stitch q7.4                              # a first look: layers taken from FIMA-Q's uniform files
    python scripts/allocate/allocate.py quantize q7.4                            # FIMA-Q with the plan's per-layer bits, packed and scored
    python scripts/allocate/allocate.py quantize both7.4 --base both7.4          # the same on the pruned model `build` wrote
    python scripts/allocate/allocate.py add NAME FILE --curve q [--seconds S]    # score a packed file and add it to the summary

`quantize` runs `methods/fimaq/run.sh` in the method's environment (`ADALOG_PY`), so FIMA-Q's folder must exist
(`scripts/setup_third_party.sh`). It takes 40 minutes to 2 hours per model on an RTX 4070.

Each layer's bits and each block's pruning cost a number of bytes and, on the first 2 000 calibration images, a KL from
DeiT-S FP32 measured with only that unit changed. The allocation minimizes the sum of those costs under the target size
with one Lagrange multiplier, then spends what is left on the best upgrades that still fit. The sum is only a guide,
since the test showed that per-layer effects do not add, so every choice is scored as a whole model. Nothing here looks
at the test sets. Curves: `q` bits only, `p` pruning only with the rest in FP16, `both` the two on equal terms.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

OUT = Path("outputs")
PLANS = OUT / "allocations"
SUMMARY = OUT / "allocated_summary.json"
PER_IMAGE = OUT / "allocated_per_image.npz"
BITS = (2, 3, 4, 6)
SPARSITY = (0.0, 0.1, 0.3, 0.5, 0.7)
DIM, HEADS, HIDDEN, HEAD_DIM, DEPTH = 384, 6, 1536, 64, 12
PATCH_W, HEAD_W = (384, 3 * 16 * 16), (1000, 384)
# bytes of what no choice changes: class token, position embedding, LayerNorms, proj, patch and head biases, in FP16
FIXED = (1 + 197) * DIM * 2 + (2 * DEPTH + 1) * 2 * DIM * 2 + DEPTH * DIM * 2 + DIM * 2 + 1000 * 2


def hidden_of(s): return max(1, round(HIDDEN * (1 - s)))
def qk_of(s): return max(1, round(HEAD_DIM * (1 - s)))


def layer_bytes(rows, cols, bits):
    """A weight matrix in the file: packed codes and one FP32 scale and INT16 zero point per row; FP16 if bits is None."""
    return rows * cols * 2 if bits is None else (rows * cols * bits + 7) // 8 + rows * 6


def sens(signal):
    """{(unit, option): cost}, from the summary. Costs never fall as a choice gets more aggressive."""
    d = json.load(open(OUT / "sensitivity_summary.json"))["points"]
    def value(r): return r["calib"]["kl"] if signal == "kl" else max(0.0, -r["calib"]["top1_change"])
    cost = {}
    layers = [k for k in d["FIMAQ"]["W3"]]
    for layer in layers:
        best = 0.0
        for b in reversed(BITS):
            best = max(best, value(d["FIMAQ"][f"W{b}"][layer]))
            cost[layer, b] = best
    for i in range(DEPTH):
        for unit in ("mlp", "attn"):
            cost[f"blocks.{i}.{unit}", 0.0] = best = 0.0
            for s in SPARSITY[1:]:
                best = max(best, value(d["CORP"][f"{round(s * 100)} %"][f"blocks.{i}.{unit}"]))
                cost[f"blocks.{i}.{unit}", s] = best
    return cost


def options(curve, fixed):
    """{unit: [(bytes, cost_key_list, choice)]} for every independent unit of the model."""
    bits = (None,) if curve == "p" else BITS
    prune = (0.0,) if curve == "q" else SPARSITY
    units = {}
    def allowed(layer, b): return fixed.get(layer, b) == b
    if curve != "p":
        for name, (r, c) in (("patch_embed.proj", PATCH_W), ("head", HEAD_W)):
            units[name] = [(layer_bytes(r, c, b), [(name, b)], {name: b}) for b in bits if allowed(name, b)]
    else:
        units["fixed"] = [(layer_bytes(*PATCH_W, None) + layer_bytes(*HEAD_W, None), [], {})]
    for i in range(DEPTH):
        fc1, fc2, qkv, proj = (f"blocks.{i}.{u}" for u in ("mlp.fc1", "mlp.fc2", "attn.qkv", "attn.proj"))
        mlp, attn = f"blocks.{i}.mlp", f"blocks.{i}.attn"
        rows = []
        for s, b1, b2 in product(prune, bits, bits):
            if allowed(fc1, b1) and allowed(fc2, b2) and fixed.get(mlp, s) == s:
                h = hidden_of(s)
                keys = [(mlp, s)] + ([(fc1, b1), (fc2, b2)] if b1 else [])
                rows.append((layer_bytes(h, DIM, b1) + layer_bytes(DIM, h, b2) + 2 * h, keys,
                             {fc1: b1, fc2: b2, mlp: s}))
        units[mlp] = rows
        rows = []
        for s, b in product(prune, bits):
            if allowed(qkv, b) and fixed.get(attn, s) == s:
                r = HEADS * (2 * qk_of(s) + HEAD_DIM)
                rows.append((layer_bytes(r, DIM, b) + 2 * r, [(attn, s)] + ([(qkv, b)] if b else []), {qkv: b, attn: s}))
        units[attn] = rows
        units[proj] = [(layer_bytes(DIM, DIM, b), [(proj, b)] if b else [], {proj: b}) for b in bits if allowed(proj, b)]
    return units


def solve(units, cost, budget):
    """One Lagrange multiplier by bisection, then the best upgrades that still fit."""
    def pick(lam):
        chosen, total = {}, 0
        for u, rows in units.items():
            r = min(rows, key=lambda r: sum(cost[k] for k in r[1]) + lam * r[0])
            chosen[u] = r; total += r[0]
        return chosen, total
    lo, hi = 0.0, 1e-3
    while pick(hi)[1] > budget and hi < 1e6:
        hi *= 4
    for _ in range(60):
        mid = (lo + hi) / 2
        if pick(mid)[1] > budget: lo = mid
        else: hi = mid
    chosen, total = pick(hi)
    if total > budget:
        raise SystemExit(f"the target cannot be met: smallest is {(total + FIXED) / 1e6:.2f} MB")
    while True:
        best, gain_best = None, 0.0
        for u, rows in units.items():
            cur = chosen[u]
            cur_cost = sum(cost[k] for k in cur[1])
            for r in rows:
                extra = r[0] - cur[0]
                if extra > 0 and total + extra <= budget:
                    gain = (cur_cost - sum(cost[k] for k in r[1])) / extra
                    if gain > gain_best:
                        best, gain_best = (u, r), gain
        if best is None:
            return chosen, total
        total += best[1][0] - chosen[best[0]][0]
        chosen[best[0]] = best[1]


def plan(args):
    PLANS.mkdir(exist_ok=True)
    fixed = json.loads(args.fix) if args.fix else {}
    fixed = {k: (float(v) if ".mlp" == k[-4:] or k.endswith(".attn") else v) for k, v in fixed.items()}
    cost, units = sens(args.signal), options(args.curve, fixed)
    header = args.overhead
    chosen, total = solve(units, cost, int(args.target_mb * 1e6) - FIXED - header)
    choice = {}
    for r in chosen.values():
        choice.update(r[2])
    layer_bits = {f"{k}.weight": v for k, v in choice.items() if not k.endswith((".mlp", ".attn")) and v is not None}
    mlp = {int(k.split(".")[1]): v for k, v in choice.items() if k.endswith(".mlp")}
    attn = {int(k.split(".")[1]): v for k, v in choice.items() if k.endswith(".attn")}
    record = {"name": args.name, "curve": args.curve, "signal": args.signal, "target_mb": args.target_mb, "fix": fixed,
              "predicted_mb": (total + FIXED + header) / 1e6, "predicted_cost": sum(cost[k] for r in chosen.values() for k in r[1]),
              "layer_bits": layer_bits, "mlp_sparsity": mlp, "attn_sparsity": attn}
    json.dump(record, open(PLANS / f"{args.name}.json", "w"), indent=1)
    (PLANS / f"{args.name}.bits.json").write_text(json.dumps(layer_bits))
    hist = {b: sum(v == b for v in layer_bits.values()) for b in BITS}
    print(f"{args.name}: predicted {record['predicted_mb']:.3f} MB, cost {record['predicted_cost']:.4f}; bits {hist}; "
          f"mlp {[mlp[i] for i in range(DEPTH)] if args.curve != 'q' else '-'}; attn {[attn[i] for i in range(DEPTH)] if args.curve != 'q' else '-'}")
    return record


def add(name, file, curve, seconds=None, mlp_act="gelu", extra=None):
    from deitc.evaluate import score
    from deitc.packing import load_packed_rows
    from deitc.results import add_point, fp32_reference
    sd = load_file(file) if str(file).endswith("_fp16.safetensors") else load_packed_rows(file)
    res, correct = score(sd, fp32_reference(), mlp_act=mlp_act)
    size = Path(file).stat().st_size
    add_point(SUMMARY, name, {"curve": curve, "file": str(file), "size_mb": size / 1e6, "seconds": seconds, **(extra or {}),
                              "accuracy": res}, PER_IMAGE, correct)
    print(f"{name}: {size / 1e6:.3f} MB  " + "  ".join(f"{d.split('-')[-1]} {r['top1']:.2f}" for d, r in res.items()), flush=True)


def build(args):
    """Prune every named plan that has pruning, with one pass of statistics shared between them."""
    import time
    from deitc.model import load_pretrained
    from deitc.packing import save_float
    from deitc.pruning import corp, corp_collect
    from deitc.pruning import loader, test_transform
    torch.manual_seed(0); np.random.seed(0)
    model = load_pretrained("DeiT-S").cuda()
    data = loader(4000, test_transform(), 64)
    start = time.time()
    stats = corp_collect(model, data)
    collect = time.time() - start
    for name in args.names:
        p = json.load(open(PLANS / f"{name}.json"))
        start = time.time()
        pruned = corp(model, stats, data, {int(k): v for k, v in p["mlp_sparsity"].items()},
                      {int(k): v for k, v in p["attn_sparsity"].items()})
        secs = collect + time.time() - start
        sd = {k: v.detach().cpu() for k, v in pruned.state_dict().items()}
        fp32, fp16 = OUT / "artifacts" / f"{name}_fp32.safetensors", OUT / "artifacts" / f"{name}_fp16.safetensors"
        save_float(sd, fp32, torch.float32); save_float(sd, fp16, torch.float16)
        print(f"{name}: pruned in {secs:.0f} s, FP16 {fp16.stat().st_size / 1e6:.3f} MB", flush=True)
        if p["curve"] == "p":
            add(name, fp16, "p", secs)
        del pruned
        torch.cuda.empty_cache()


def stitch(args):
    """Take each layer from the FIMA-Q file of its width. The files were searched at one width for the whole model,
    so this is a first look, not the method's result."""
    from deitc.packing import load_entries, save_packed_rows
    for name in args.names:
        p = json.load(open(PLANS / f"{name}.json"))
        files = {b: load_entries(OUT / "artifacts" / f"fimaq_w{b}.safetensors") for b in BITS}
        qsd = dict(files[3][1])
        for k, b in p["layer_bits"].items():
            qsd[k] = files[b][0][k]
        f = OUT / "artifacts" / f"{name}_stitched.safetensors"
        save_packed_rows(qsd, f, p["layer_bits"], method="fimaq-stitched")
        add(f"{name} stitched", f, p["curve"], extra={"stitched": True, "plan": p["name"], "signal": p["signal"], "fix": p["fix"]})


def quantize(args):
    """FIMA-Q weights-only with the plan's bits per layer (on the pruned model if `--base`), then pack, check and score."""
    import os
    import re
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    work = Path(os.environ.get("FIMAQ_DIR", repo.parent / "third_party" / "FIMA-Q-protocol"))
    name, bits = args.name, PLANS / f"{args.name}.bits.json"
    cmd = ["bash", "methods/fimaq/run.sh", "3bit/w3a32.py", name, "--layer-bits", str(bits.resolve())]
    if args.base:
        cmd += ["--pruned-weights", str((OUT / "artifacts" / f"{args.base}_fp32.safetensors").resolve())]
    subprocess.run(cmd, check=True, env={**os.environ, "SCORE_OUT": "allocated_check.json"})
    log = (work / f"run_{name.replace(' ', '_')}.log").read_text(errors="replace")
    ckpt = work / re.findall(r"Saving checkpoint to (.*optimsize.*)", log)[-1].strip()
    f = OUT / "artifacts" / f"{name}_fimaq.safetensors"
    subprocess.run([sys.executable, "scripts/quantize/pack_published.py", "--method", "fimaq", "--layer-bits", str(bits),
                    "--ckpt", str(ckpt), "--out", str(f)], check=True)
    r = json.load(open(OUT / "allocated_check.json"))[f"{name}, in the method's forward"]
    curve = json.load(open(PLANS / f"{name}.json"))["curve"]
    add(name, f, curve, r["calibration_s"] + r["block_reconstruction_s"], extra={"plan": name, "base": args.base or ""})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    a = sub.add_parser("plan"); a.add_argument("name"); a.add_argument("--curve", choices=["q", "p", "both"], required=True)
    a.add_argument("--target-mb", type=float, required=True); a.add_argument("--signal", default="kl", choices=["kl", "top1"])
    a.add_argument("--fix", help='json of choices that are not free, e.g. {"blocks.0.attn": 0, "blocks.0.mlp.fc1": 3}')
    a.add_argument("--overhead", type=int, default=34300, help="file bytes outside the model, measured on FIMA-Q's files")
    b = sub.add_parser("build"); b.add_argument("names", nargs="+")
    c = sub.add_parser("stitch"); c.add_argument("names", nargs="+")
    e = sub.add_parser("quantize"); e.add_argument("name"); e.add_argument("--base", help="the pruned plan whose FP32 file to start from")
    d = sub.add_parser("add"); d.add_argument("name"); d.add_argument("file"); d.add_argument("--curve", required=True)
    d.add_argument("--seconds", type=float); d.add_argument("--mlp-act", default="gelu"); d.add_argument("--extra")
    args = ap.parse_args()
    if args.what == "plan": plan(args)
    elif args.what == "build": build(args)
    elif args.what == "stitch": stitch(args)
    elif args.what == "quantize": quantize(args)
    else: add(args.name, args.file, args.curve, args.seconds, args.mlp_act, json.loads(args.extra) if args.extra else None)


if __name__ == "__main__":
    main()
