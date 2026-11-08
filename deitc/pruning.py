"""Pruning without retraining: the shared calibration images and our implementation of CORP.

CORP (Zhang and Yang, 2026, arXiv:2602.05243) has no public code that we could find, so this is a reimplementation
from the paper's Algorithms 1 to 5, checked against the DeiT-S numbers it reports. It removes MLP hidden channels and
query and key dimensions, and folds a closed-form correction into the weights that remain. The paper does not print
its ridge value, so `RIDGE` is our choice: the relative value GRAIL's code uses, scaled by the mean diagonal.
"""
import copy
from pathlib import Path

import torch
from PIL import Image
import sys
from types import SimpleNamespace
import timm
from timm.data import create_transform, resolve_data_config
from torchvision import transforms
from deitc.model import TIMM_NAMES

RIDGE = 1e-3


class CalibImages(torch.utils.data.Dataset):
    """The first `n` lines of the calibration list, in order."""
    def __init__(self, n, transform, root="../data/imagenet/train", listing="splits/calib_seed0.txt"):
        lines = Path(listing).read_text().split()
        if n > len(lines):
            raise ValueError(f"asked for {n} calibration images, the list has {len(lines)}")
        self.files, self.transform, self.root = lines[:n], transform, Path(root)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.transform(Image.open(self.root / self.files[i]).convert("RGB")), 0


def _keep(scores, sparsity):
    """Indices of the channels to keep, in their original order."""
    k = max(1, round(scores.numel() * (1 - sparsity)))
    return scores.topk(k).indices.sort().values


def _ridge(matrix):
    return RIDGE * matrix.diagonal().mean()


@torch.no_grad()
def corp_collect(model, loader, device="cuda", mlp=True, attn=True):
    """First pass, on the unpruned model: MLP activation statistics and the logit energy of each Q/K dimension.
    A scope that prunes only one of the two collects only its own statistics, so its time is its own."""
    model.to(device).eval()
    stats, hooks = {}, []

    def mlp_hook(i):
        def hook(_, __, out):
            x = out.reshape(-1, out.shape[-1]).double()
            s = stats.setdefault(("mlp", i), {"n": 0, "sum": 0, "xx": 0})
            s["n"] += x.shape[0]; s["sum"] = s["sum"] + x.sum(0); s["xx"] = s["xx"] + x.T @ x
        return hook

    def attn_hook(i, attn):
        def hook(_, __, out):
            B, N, _ = out.shape
            q, k, _ = out.reshape(B, N, 3, attn.num_heads, attn.head_dim).double().unbind(2)   # (B, N, H, d)
            energy = (q ** 2).sum(1) * (k ** 2).sum(1)                                         # (B, H, d), Alg. 4
            s = stats.setdefault(("attn", i), {"n": 0, "energy": 0})
            s["n"] += B; s["energy"] = s["energy"] + energy.sum(0)
        return hook

    for i, blk in enumerate(model.blocks):
        if mlp:
            hooks.append(blk.mlp.act.register_forward_hook(mlp_hook(i)))
        if attn:
            hooks.append(blk.attn.qkv.register_forward_hook(attn_hook(i, blk.attn)))
    for x, _ in loader:
        model(x.to(device))
    for h in hooks:
        h.remove()
    return stats


def _prune_mlp(blk, s, sparsity):
    """Algorithms 2 and 3. Ranked by activation energy times squared column norm, the paper's default strategy."""
    fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
    W, b = fc2.weight.data.double(), fc2.bias.data.double()
    mean = s["sum"] / s["n"]
    second = s["xx"] / s["n"]
    keep = _keep(second.diagonal() * (W ** 2).sum(0), sparsity)
    drop = torch.ones(W.shape[1], dtype=torch.bool, device=W.device); drop[keep] = False
    drop = drop.nonzero().squeeze(1)
    cov = second - torch.outer(mean, mean)
    ss, ps = cov[keep][:, keep], cov[drop][:, keep]
    B = torch.linalg.solve(ss + _ridge(ss) * torch.eye(len(keep), device=W.device, dtype=W.dtype), ps.T).T
    c = mean[drop] - B @ mean[keep]
    new_w = (W[:, keep] + W[:, drop] @ B).float()
    new_b = (b + W[:, drop] @ c).float()
    new_fc1 = torch.nn.Linear(fc1.in_features, len(keep)).to(W.device)
    new_fc1.weight.data, new_fc1.bias.data = fc1.weight.data[keep], fc1.bias.data[keep]
    new_fc2 = torch.nn.Linear(len(keep), fc2.out_features).to(W.device)
    new_fc2.weight.data, new_fc2.bias.data = new_w, new_b
    blk.mlp.fc1, blk.mlp.fc2 = new_fc1, new_fc2


@torch.no_grad()
def _attn_systems(model, loader, keeps, device):
    """Second pass, for Algorithm 5: the normal equations of QP KP^T = QS M KS^T, one system per head of each block.
    The Kronecker sum over images is built with one matrix product per batch instead of one product per image."""
    systems, hooks = {}, []

    def hook_for(i, attn):
        def hook(_, __, out):
            B, N, _ = out.shape
            q, k, _ = out.reshape(B, N, 3, attn.num_heads, attn.head_dim).double().unbind(2)
            for h in range(attn.num_heads):
                keep, drop = keeps[i][h]
                qs, qp = q[:, :, h][:, :, keep], q[:, :, h][:, :, drop]
                ks, kp = k[:, :, h][:, :, keep], k[:, :, h][:, :, drop]
                a, c = qs.transpose(1, 2) @ qs, ks.transpose(1, 2) @ ks               # (B, d', d')
                rhs = (qs.transpose(1, 2) @ qp) @ (kp.transpose(1, 2) @ ks)
                sys = systems.setdefault((i, h), {"G": 0, "h": 0})
                d = len(keep)
                t = a.reshape(B, d * d).T @ c.reshape(B, d * d)                       # indexed (i, k), (j, l)
                sys["G"] = sys["G"] + t.reshape(d, d, d, d).permute(0, 2, 1, 3).reshape(d * d, d * d)
                sys["h"] = sys["h"] + rhs.sum(0).reshape(-1)
        return hook

    for i, blk in enumerate(model.blocks):
        if i in keeps:
            hooks.append(blk.attn.qkv.register_forward_hook(hook_for(i, blk.attn)))
    for x, _ in loader:
        model(x.to(device))
    for h in hooks:
        h.remove()
    return systems


def _prune_attn(blk, i, keeps, systems):
    """Fold the logit correction into Q and K: QS (I + M) KS^T = (QS U S^1/2)(KS V S^1/2)^T. V is left as it was."""
    attn = blk.attn
    H, d = attn.num_heads, attn.head_dim
    W, b = attn.qkv.weight.data.double(), attn.qkv.bias.data.double()
    d_keep = len(keeps[i][0][0])
    w_q, w_k, b_q, b_k = [], [], [], []
    for h in range(H):
        keep = keeps[i][h][0]
        sys = systems[(i, h)]
        n = d_keep * d_keep
        M = torch.linalg.solve(sys["G"] + _ridge(sys["G"]) * torch.eye(n, device=W.device, dtype=W.dtype),
                               sys["h"]).reshape(d_keep, d_keep)
        U, S, Vh = torch.linalg.svd(torch.eye(d_keep, device=W.device, dtype=W.dtype) + M)
        # Layers hold rows and Q = X W^T + b, so the factors act on the left, on the kept rows.
        left_q, left_k = S.sqrt().diag() @ U.T, S.sqrt().diag() @ Vh
        qrows, krows = h * d + keep, H * d + h * d + keep
        w_q.append(left_q @ W[qrows]); b_q.append(left_q @ b[qrows])
        w_k.append(left_k @ W[krows]); b_k.append(left_k @ b[krows])
    new = torch.nn.Linear(W.shape[1], H * (2 * d_keep + d)).to(W.device)
    new.weight.data = torch.cat(w_q + w_k + [W[2 * H * d:]]).float()
    new.bias.data = torch.cat(b_q + b_k + [b[2 * H * d:]]).float()
    attn.qkv, attn.qk_dim = new, d_keep


@torch.no_grad()
def corp(model, stats, loader, s_mlp, s_attn, device="cuda"):
    """Algorithm 1. Every statistic comes from the unpruned model, so `model` is left as it was. Each sparsity is one
    value for all blocks, as in the paper, or {block: value} for a per-block allocation."""
    model.to(device).eval()
    pruned = copy.deepcopy(model)
    n = len(model.blocks)
    s_mlp = s_mlp if isinstance(s_mlp, dict) else {i: s_mlp for i in range(n)}
    s_attn = s_attn if isinstance(s_attn, dict) else {i: s_attn for i in range(n)}
    keeps = {}
    for i, blk in enumerate(model.blocks):
        if s_attn.get(i, 0) > 0:
            energy = stats[("attn", i)]["energy"]
            keeps[i] = []
            for h in range(blk.attn.num_heads):
                keep = _keep(energy[h], s_attn[i])
                drop = torch.ones(energy.shape[1], dtype=torch.bool, device=energy.device); drop[keep] = False
                keeps[i].append((keep, drop.nonzero().squeeze(1)))
    systems = _attn_systems(model, loader, keeps, device) if keeps else None
    for i, blk in enumerate(pruned.blocks):
        if s_mlp.get(i, 0) > 0:
            _prune_mlp(blk, stats[("mlp", i)], s_mlp[i])
        if i in keeps:
            _prune_attn(blk, i, keeps, systems)
    return pruned


# Adapters for the three pruning methods, moved here from `scripts/prune/prune_points.py` on 2026-11-06 so that every script
# calls them the same way. VBP and GRAIL run the authors' code from `third_party/`.
THIRD = Path(__file__).resolve().parents[2] / "third_party"


def test_transform():
    return create_transform(**resolve_data_config({}, model=timm.create_model(TIMM_NAMES["DeiT-S"], pretrained=False)))


def loader(n, transform, batch_size):
    return torch.utils.data.DataLoader(CalibImages(n, transform), batch_size=batch_size, num_workers=8)


def run_vbp(model, sparsity, n):
    sys.path.insert(0, str(THIRD / "VBP/src"))
    from pruner import indices, prune, score as vbp_score, threshold
    tfm = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    args = SimpleNamespace(model="DeiT", device="cuda", num_batches=0, percentage=sparsity)
    variances, means = vbp_score(args, model, loader(n, tfm, 32))    # batch size from the authors' README
    return prune(args, model, indices(args, variances, threshold(args, variances)), means)


def run_grail(model, method, sparsity, n, compensate=True, blocks=None):
    sys.path.insert(0, str(THIRD / "GRAIL/grail-vision"))
    # GRAIL's modules import these at the top for its folding and CLIP paths, which we do not run.
    sys.modules.setdefault("hkmeans", SimpleNamespace(HKMeans=None))
    sys.modules.setdefault("open_clip", SimpleNamespace(create_model_and_transforms=None))
    import compression.base_vit as base_vit
    from compensation.prune_compensation import Vit_PruneCompensation
    from compression.mag_prune import ViT_MagnitudePruning
    from compression.wanda_prune import ViT_WandaPruning
    base_vit.get_axis_to_perm_ViT_Simple = lambda m: {
        f"group_{i}": [f"blocks.{i}.mlp.fc1", f"blocks.{i}.mlp.fc2"] for i in (range(len(m.blocks)) if blocks is None else blocks)}
    pruner = {"mag-l1": lambda: ViT_MagnitudePruning(model, compression_ratio=sparsity, p=1),
              "mag-l2": lambda: ViT_MagnitudePruning(model, compression_ratio=sparsity, p=2),
              "wanda": lambda: ViT_WandaPruning(model, compression_ratio=sparsity)}[method]()
    pruner.run_calibration(loader(n, test_transform(), 128), "cuda", num_batches=10 ** 9)
    pruned = pruner.apply().cuda()
    if compensate:
        comp = Vit_PruneCompensation(pruned, ridge_lambda=1e-3)
        comp.load_compression_state(pruner.get_compression_state())
        comp.load_gram_stats(pruner.get_gram_stats())
        pruned = comp.apply()
    return pruned


def run_corp(model, sparsity, n, scope):
    data = loader(n, test_transform(), 64)
    stats = corp_collect(model, data, mlp=scope != "attn", attn=scope != "mlp")
    return corp(model, stats, data, sparsity if scope in ("both", "mlp") else 0, sparsity if scope in ("both", "attn") else 0)
