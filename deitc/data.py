"""The three test sets and the calibration pool, with labels as ImageNet-1k class indices."""
from pathlib import Path

import torch
from PIL import Image
from timm.data import ImageNetInfo

WNIDS = ImageNetInfo().label_names()            # index -> synset id, in the standard ImageNet-1k order
WNID_TO_INDEX = {w: i for i, w in enumerate(WNIDS)}


class ImageList(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        f, y = self.items[i]
        return self.transform(Image.open(f).convert("RGB")), y


def imagenet_v2(root, transform):
    # Folders are class indices "0"…"999"; ImageFolder would sort them as strings.
    return ImageList([(f, int(f.parent.name)) for f in sorted(Path(root).glob("*/*.jpeg"))], transform)


def by_wnid(root, transform):
    # Folders named by synset id (ImageNet-1k val/train as prepared here, ImageNet-R); labels are ImageNet-1k indices.
    files = [f for f in sorted(Path(root).glob("*/*")) if f.parent.name in WNID_TO_INDEX and f.is_file()]
    return ImageList([(f, WNID_TO_INDEX[f.parent.name]) for f in files], transform)
