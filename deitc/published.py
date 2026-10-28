"""Turn the checkpoint a published method saves into a packed file, without changing what the method computed.

The methods save fake-quantized checkpoints: the original FP32 weights plus the grid each layer chose. Here the
grid is applied once to get the integer codes, which `packing.save_packed_rows` stores at their real bit width.
`fake_weights` repeats the method's own rounding, so the file can be checked against it bit for bit.
Tensors the method does not quantize (biases, LayerNorms, embeddings) are stored in FP16, as in every file of the
protocol.
"""
import torch

from .model import deit_small


def _adalog_layers(ckpt):
    # Every layer AdaLog quantized keeps `<layer>.w_quantizer.scale` and `.zero_point` (asymmetric, per row).
    return [k[: -len(".w_quantizer.scale")] for k in ckpt if k.endswith(".w_quantizer.scale")]


def _adalog_grid(ckpt, layer, bits):
    w = ckpt[layer + ".weight"].float()
    rows = w.reshape(w.shape[0], -1)
    scale = ckpt[layer + ".w_quantizer.scale"].float().reshape(w.shape[0], 1)
    zero = torch.round(ckpt[layer + ".w_quantizer.zero_point"].float()).reshape(w.shape[0], 1)
    # quantizers/uniform.py: x_int = round(x / scale); q = clamp(x_int + round(zp), 0, 2^b - 1)
    q = (torch.round(rows / scale) + zero).clamp(0, 2 ** bits - 1)
    return w.shape, q, scale, zero


def adalog_to_packable(ckpt, bits):
    """{name: FP16 tensor or {q, scale, zero}} for `save_packed_rows`, keyed like our DeiT."""
    layers = set(_adalog_layers(ckpt))
    out = {}
    for k in deit_small().state_dict():
        layer = k[: -len(".weight")]
        if k.endswith(".weight") and layer in layers:
            shape, q, scale, zero = _adalog_grid(ckpt, layer, bits)
            out[k] = {"q": q.to(torch.uint8).reshape(shape), "scale": scale, "zero": zero}
        else:
            out[k] = ckpt[k].float().half()
    return out


def adalog_fake_weights(ckpt, bits):
    """The quantized weights exactly as AdaLog's forward pass uses them: (q - round(zp)) * scale."""
    out = {}
    for layer in _adalog_layers(ckpt):
        shape, q, scale, zero = _adalog_grid(ckpt, layer, bits)
        out[layer + ".weight"] = ((q - zero) * scale).reshape(shape)
    return out
