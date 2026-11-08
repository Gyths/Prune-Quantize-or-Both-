"""Rounding helpers shared by the scripts: per-row round-to-nearest grids, AdaLog's grid for a layer, and 8-bit
per-token rounding of activations. Moved here unchanged from `scripts/quantize/validate_low_bits.py`, `scripts/prune/quantize_then_prune.py` and
`scripts/quantize/a8_score.py` on 2026-11-06.
"""
import torch


def rtn_rows(w, bits):
    rows = w.reshape(w.shape[0], -1)
    lo, hi = rows.amin(1, keepdim=True), rows.amax(1, keepdim=True)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    zero = torch.round(-lo / scale)
    q = (torch.round(rows / scale) + zero).clamp(0, 2 ** bits - 1)
    return {"q": q.to(torch.uint8).reshape(w.shape), "scale": scale, "zero": zero}


def adalog_layer(k, v):
    return k.endswith(".weight") and v.ndim in (2, 4)   # patch embedding (conv), block linears, head


def round_rows(w, bits=3):
    """Per-row asymmetric codes at bits for a weight matrix, in the packed format."""
    flat = w.reshape(w.shape[0], -1)
    lo, hi = flat.amin(1, keepdim=True).clamp(max=0), flat.amax(1, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    zero = (-lo / scale).round()
    q = ((flat / scale).round() + zero).clamp(0, 2 ** bits - 1).to(torch.uint8).reshape(w.shape)
    return {"q": q, "scale": scale, "zero": zero}


def round_input(_, args):
    x = args[0]
    lo, hi = x.amin(-1, keepdim=True).clamp(max=0), x.amax(-1, keepdim=True).clamp(min=0)
    scale = ((hi - lo) / 255).clamp_min(1e-8)
    zero = (-lo / scale).round()
    return (((x / scale).round() + zero).clamp(0, 255) - zero) * scale,
