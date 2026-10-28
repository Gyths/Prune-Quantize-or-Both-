"""DeiT rebuilt in plain PyTorch, as in notebook 01, with the official weights loaded strictly."""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    # A 16x16 conv with stride 16 is the same as flattening each patch and applying one linear layer.
    def __init__(self, img_size=224, patch_size=16, in_chans=3, dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (B, 3, 224, 224)
        x = self.proj(x)                       # (B, 384, 14, 14)
        return x.flatten(2).transpose(1, 2)    # (B, 196, 384)


class Attention(nn.Module):
    def __init__(self, dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * num_heads * self.head_dim)   # rows: [q heads | k heads | v heads]
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.last_attn = None                  # filled when record_attn=True

    def forward(self, x, record_attn=False):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                # each (B, H, N, 64)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)            # (B, H, N, N)
        if record_attn:
            self.last_attn = attn.detach()
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim=384, hidden=1536):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()                   # exact (erf) GELU, as in the checkpoint
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim=384, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, record_attn=False):
        x = x + self.attn(self.norm1(x), record_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class DeiT(nn.Module):
    def __init__(self, dim=384, depth=12, num_heads=6, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(dim=dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, return_hidden=False, record_attn=False):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        hidden = []
        for blk in self.blocks:
            x = blk(x, record_attn)
            hidden.append(x)
        logits = self.head(self.norm(x)[:, 0])  # classify from the class token
        return (logits, hidden) if return_hidden else logits


def deit_small(): return DeiT(dim=384, depth=12, num_heads=6)
def deit_tiny():  return DeiT(dim=192, depth=12, num_heads=3)

TIMM_NAMES = {"DeiT-S": "deit_small_patch16_224.fb_in1k", "DeiT-Ti": "deit_tiny_patch16_224.fb_in1k"}
BUILDERS = {"DeiT-S": deit_small, "DeiT-Ti": deit_tiny}


def load_pretrained(name):
    """Our rebuild with the official weights copied in (strict=True, as in notebook 01)."""
    import timm
    ref = timm.create_model(TIMM_NAMES[name], pretrained=True)
    model = BUILDERS[name]()
    model.load_state_dict(ref.state_dict(), strict=True)
    return model.eval()
