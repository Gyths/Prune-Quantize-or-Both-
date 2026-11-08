"""The costs of a point besides its file size: arithmetic (MACs, BOPs), latency and memory at run time.

MACs are counted by `torch.utils.flop_counter.FlopCounterMode` on our DeiT, one image of 224 x 224, and split by
layer. Every multiply-accumulate in DeiT happens in one of eight places per block or end: the patch embedding, the
four block linears (qkv, proj, fc1, fc2), the two products inside attention (q times k, and the attention weights
times v, which cost the same unless Q and K are pruned) and the head. Softmax, GELU and LayerNorm are not counted, as is usual. A pruned
model is counted from its own smaller layers, so pruning shows up here and bit width does not.

BOPs weight each layer's MACs by the bits of its two operands: MACs x b_w x b_a for a layer with weights, and
MACs x b_left x b_right for the two products inside attention. The convention for what is not quantized is fixed
for every point: an operand takes the bits it is stored or quantized at, and an activation left at full precision
counts as 32 bits, because that is how our pipeline computes it. So the FP16 layers of a packed file count as 16
bits on the weight side and 32 on the activation side.
"""
import statistics

import torch
from torch.utils.flop_counter import FlopCounterMode

KINDS = ("patch_embed", "qkv", "proj", "fc1", "fc2", "qk", "av", "head")


@torch.no_grad()
def count_macs(model, img=224):
    """{layer: MACs} for one image, with layer names like 'blocks.3.attn.qk' and a 'kind' per layer."""
    model = model.to("meta").eval()
    x = torch.empty(1, 3, img, img, device="meta")
    fc = FlopCounterMode(display=False)
    with fc:
        model(x)
    counts = {k.split(".", 1)[1] if "." in k else k: sum(v.values()) // 2 for k, v in fc.get_flop_counts().items()}
    macs = {"patch_embed.proj": counts["patch_embed.proj"], "head": counts["head"]}
    for i in range(len(model.blocks)):
        b = f"blocks.{i}"
        for lin in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            macs[f"{b}.{lin}"] = counts[f"{b}.{lin}"]
        inner = counts[f"{b}.attn"] - counts[f"{b}.attn.qkv"] - counts[f"{b}.attn.proj"]
        attn = model.blocks[i].attn                                # q times k costs B*H*N*N*qk_dim, weights times v B*H*N*N*d
        macs[f"{b}.attn.qk"] = inner * attn.qk_dim // (attn.qk_dim + attn.head_dim)
        macs[f"{b}.attn.av"] = inner - macs[f"{b}.attn.qk"]
    assert counts["Global"] == sum(macs.values()), "MACs outside the eight known places"
    return macs


def kind(layer):
    last = layer.rsplit(".", 1)[-1]
    return {"proj": "patch_embed" if layer.startswith("patch_embed") else "proj"}.get(last, last)


def bops(macs, bits):
    """Sum of MACs x bits_a x bits_b, with `bits(layer) -> (b_weight_or_left, b_activation_or_right)`."""
    return sum(m * bits(layer)[0] * bits(layer)[1] for layer, m in macs.items())


# ---------- Bit assignments for the points on the chart ----------

def bits_float(b):
    """Every operand at b bits (FP32 = 32, FP16 = 16)."""
    return lambda layer: (b, b)


def bits_rtn(w):
    """Our round-to-nearest files: block linears at w bits, the rest stored in FP16; activations at 32."""
    def f(layer):
        k = kind(layer)
        if k in ("qkv", "proj", "fc1", "fc2"):
            return (w, 32)
        if k in ("qk", "av"):
            return (32, 32)
        return (16, 32)
    return f


def bits_weights_only(w):
    """A published method run weights-only: patch embedding, block linears and head at w bits; activations 32."""
    return lambda layer: (32, 32) if kind(layer) in ("qk", "av") else (w, 32)


def bits_adalog_published(w, a, s, conv_a=8, head_a=None):
    """AdaLog-family configs (utils/wrap_net.py): linears W x A, the patch embedding's input at `qconv_a_bit`,
    the head's input at `qhead_a_bit`, q times k at A x A, and the post-softmax weights (s bits) times v (A)."""
    head_a = a if head_a is None else head_a
    def f(layer):
        k = kind(layer)
        return {"patch_embed": (w, conv_a), "head": (w, head_a), "qk": (a, a), "av": (s, a)}.get(k, (w, a))
    return f


def bits_w8a8_linears():
    """torchao W8A8 on the 48 block linears; patch embedding, head and attention products in float."""
    return lambda layer: (8, 8) if kind(layer) in ("qkv", "proj", "fc1", "fc2") else (32, 32)


# ---------- Latency and memory ----------

@torch.no_grad()
def time_model(model, batch, dtype=torch.float32, warmup=30, iters=100, device="cuda"):
    """Median and quartiles of one forward pass (ms) by CUDA events, and the peak memory allocated (MiB).

    The input is created on the GPU once, so the time is the model's alone, without data loading or copies."""
    x = torch.randn(batch, 3, 224, 224, device=device, dtype=dtype)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); model(x); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    q = statistics.quantiles(times, n=4)
    return {"median_ms": statistics.median(times), "p25_ms": q[0], "p75_ms": q[2],
            "images_per_s": batch * 1000 / statistics.median(times),
            "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20}
