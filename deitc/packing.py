"""Weight-only quantization of the block linears, packed into a real safetensors file.

The file is the measure of size on every curve: the 48 block linears (qkv, proj, fc1, fc2) hold integer
codes, and everything else (biases, LayerNorms, embeddings, head) is FP16.

- INT8: symmetric, one FP16 scale per output row.
- INT4 (and below): asymmetric, groups of `group` consecutive input weights per row, FP16 scale and FP16 minimum
  per group; two 4-bit codes per byte, and other widths as one continuous bit stream.

The quantizer here is plain round-to-nearest. Published methods plug in their own codes and scales and reuse
`save_packed` / `load_packed`, so that every point on the chart is sized the same way.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "deitc-pack-v1"


def is_block_linear(name, tensor):
    return name.startswith("blocks.") and name.endswith(".weight") and tensor.ndim == 2


def rtn_rows_int8(w):
    # Scales are rounded to FP16 before computing the codes, so that the stored file reproduces them exactly.
    scale = (w.abs().amax(1, keepdim=True) / 127).clamp_min(1e-8).half().float()
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return {"q": q, "scale": scale.half()}


def rtn_groups(w, bits, group):
    out_f, in_f = w.shape
    assert in_f % group == 0, f"{in_f} inputs are not divisible into groups of {group}"
    wg = w.reshape(out_f, in_f // group, group)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    lo16 = lo.half().float()
    scale = ((hi - lo16) / (2 ** bits - 1)).clamp_min(1e-8).half().float()
    q = torch.round((wg - lo16) / scale).clamp(0, 2 ** bits - 1).to(torch.uint8)
    return {"q": q.reshape(out_f, in_f), "scale": scale.squeeze(-1).half(), "min": lo16.squeeze(-1).half()}


def dequantize(entry, bits, group):
    if bits == 8:
        return entry["q"].float() * entry["scale"].float()
    q = entry["q"].float()
    out_f, in_f = q.shape
    w = q.reshape(out_f, in_f // group, group) * entry["scale"].float()[..., None] + entry["min"].float()[..., None]
    return w.reshape(out_f, in_f)


def pack4(q):
    flat = q.reshape(-1)
    return (flat[0::2] | (flat[1::2] << 4)).contiguous()


def unpack4(packed, shape):
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    out[0::2], out[1::2] = packed & 0x0F, packed >> 4
    return out.reshape(shape)


def pack_bits(q, bits):
    # Any width below 8: each code's low `bits` bits, concatenated into one bit stream.
    b = np.unpackbits(q.reshape(-1, 1).numpy(), axis=1)[:, 8 - bits:]
    return torch.from_numpy(np.packbits(b.reshape(-1)))


def unpack_bits(packed, shape, bits):
    n = int(np.prod(shape))
    b = np.unpackbits(packed.numpy())[: n * bits].reshape(n, bits)
    full = np.zeros((n, 8), dtype=np.uint8); full[:, 8 - bits:] = b
    return torch.from_numpy(np.packbits(full, axis=1).reshape(shape))


def pack_codes(q, bits):
    return pack4(q) if bits == 4 else pack_bits(q, bits)


def unpack_codes(packed, shape, bits):
    return unpack4(packed, shape) if bits == 4 else unpack_bits(packed, shape, bits)


def quantize_state_dict(sd, bits, group=64):
    """{name: tensor} -> {name: FP16 tensor or dict of codes}. Uses round-to-nearest."""
    out = {}
    for k, v in sd.items():
        v = v.detach().float().cpu()
        if is_block_linear(k, v):
            out[k] = rtn_rows_int8(v) if bits == 8 else rtn_groups(v, bits, group)
        else:
            out[k] = v.half()
    return out


def save_packed(qsd, path, bits, group=64):
    if not 2 <= bits <= 8:
        raise NotImplementedError("packing covers 2 to 8 bits")
    tensors, shapes = {}, {}
    for k, v in qsd.items():
        if isinstance(v, dict):
            shapes[k] = list(v["q"].shape)
            tensors[k + ".q"] = v["q"].contiguous() if bits == 8 else pack_codes(v["q"], bits)
            for part in ("scale", "min"):
                if part in v:
                    tensors[f"{k}.{part}"] = v[part].contiguous()
        else:
            tensors[k] = v.contiguous()
    meta = {"format": FORMAT, "bits": str(bits), "group": str(group), "shapes": json.dumps(shapes)}
    save_file(tensors, str(path), metadata=meta)
    return Path(path).stat().st_size


def load_packed(path):
    """Read a packed file back into an FP32 state_dict (weights dequantized)."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        bits, group, shapes = int(meta["bits"]), int(meta["group"]), json.loads(meta["shapes"])
        raw = {k: f.get_tensor(k) for k in f.keys()}
    sd = {}
    for k, shape in shapes.items():
        q = raw.pop(k + ".q")
        entry = {"q": q if bits == 8 else unpack_codes(q, shape, bits), "scale": raw.pop(k + ".scale")}
        if k + ".min" in raw:
            entry["min"] = raw.pop(k + ".min")
        sd[k] = dequantize(entry, bits, group)
    sd.update({k: v.float() for k, v in raw.items()})
    return sd


def dequantized_state_dict(qsd, bits, group=64):
    return {k: dequantize(v, bits, group) if isinstance(v, dict) else v.float() for k, v in qsd.items()}


def save_float(sd, path, dtype):
    save_file({k: v.detach().to(dtype).contiguous().cpu() for k, v in sd.items()}, str(path))
    return Path(path).stat().st_size


# ---------- Per-row format for published methods ----------
# One asymmetric grid per output row (conv weights flattened to rows), stored as the method keeps it: FP32 scale,
# integer zero point. w = (q - zero) * scale, the same operations the methods use in their fake quantization.

ROWS_FORMAT = "deitc-pack-rows-v1"


def dequantize_rows(entry, shape):
    q = entry["q"].float().reshape(shape[0], -1)
    return ((q - entry["zero"].float()) * entry["scale"].float()).reshape(shape)


def save_packed_rows(qsd, path, bits, method=""):
    if not 2 <= bits <= 8:
        raise NotImplementedError("packing covers 2 to 8 bits")
    tensors, shapes = {}, {}
    for k, v in qsd.items():
        if isinstance(v, dict):
            shapes[k] = list(v["q"].shape)
            tensors[k + ".q"] = v["q"].contiguous() if bits == 8 else pack_codes(v["q"], bits)
            tensors[k + ".scale"] = v["scale"].float().contiguous()
            tensors[k + ".zero"] = v["zero"].to(torch.int16).contiguous()
        else:
            tensors[k] = v.contiguous()
    meta = {"format": ROWS_FORMAT, "bits": str(bits), "method": method, "shapes": json.dumps(shapes)}
    save_file(tensors, str(path), metadata=meta)
    return Path(path).stat().st_size


def load_packed_rows(path):
    """Read a per-row packed file back into an FP32 state_dict (weights dequantized)."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        bits, shapes = int(meta["bits"]), json.loads(meta["shapes"])
        raw = {k: f.get_tensor(k) for k in f.keys()}
    sd = {}
    for k, shape in shapes.items():
        q = raw.pop(k + ".q")
        entry = {"q": q.reshape(shape) if bits == 8 else unpack_codes(q, shape, bits),
                 "scale": raw.pop(k + ".scale"), "zero": raw.pop(k + ".zero")}
        sd[k] = dequantize_rows(entry, shape)
    sd.update({k: v.float() for k, v in raw.items()})
    return sd
