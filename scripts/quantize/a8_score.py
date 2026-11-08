"""Score packed models with their block linears' inputs rounded to 8 bits on the fly, as dynamic INT8 kernels do.

    python scripts/quantize/a8_score.py "FIMA-Q W3 A8"=outputs/artifacts/fimaq_w3.safetensors ...

Each token gets its own 8-bit asymmetric grid over its own range. Nothing is stored for the activations, so the file size
is that of the packed weights. This runs in floating point, so it says nothing about speed. It needs no kernel, so pruned
widths score as easily as dense ones. It covers the same activations as torchao and GemLite: the inputs to the 48 block
linears, and not Q, K, V or the attention scores. GemLite's A8 on the dense FIMA-Q files is the check: 72.95, 78.53, 79.50
and 79.74 on val at W2, W3, W4 and W6. Results go to `outputs/a8_summary.json`.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
from pathlib import Path

import timm
import torch
from timm.data import create_transform, resolve_data_config

from deitc.evaluate import score_in_process
from deitc.model import TIMM_NAMES, from_state_dict
from deitc.packing import load_packed_rows
from deitc.quant import round_input

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
OUT = Path("outputs")


def main():
    tfm = create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES["DeiT-S"], pretrained=False)))
    for pair in sys.argv[1:]:
        name, path = pair.rsplit("=", 1)
        model = from_state_dict(load_packed_rows(path)).cuda().eval()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and fqn.startswith("blocks."):
                mod.register_forward_pre_hook(round_input)
        score_in_process(model, tfm, name, OUT / "a8_summary.json", OUT / "a8_per_image.npz",
                         {"file": path, "size_mb": Path(path).stat().st_size / 1e6, "activations": "dynamic 8-bit, simulated"})
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
