"""Download ImageNet-1k from Hugging Face (ILSVRC/imagenet-1k, gated) into folders by synset id.

    hf auth login                      # once, after accepting the terms on the dataset page
    python prepare_imagenet.py --out ../data/imagenet --train-shards 2

Writes <out>/val/<wnid>/*.JPEG (all 50 000 images) and <out>/train/<wnid>/*.JPEG from the first
`--train-shards` training files (about 4 300 images each). The training images are only a pool to draw
calibration sets from; nothing is trained here. The layout is the ImageFolder one that our notebooks and the
published methods' code read.
"""
import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from timm.data import ImageNetInfo

REPO = "ILSVRC/imagenet-1k"
WNIDS = ImageNetInfo().label_names()


def export(parquet_path, out_dir, prefix):
    n = 0
    table = pq.ParquetFile(parquet_path)
    for batch in table.iter_batches(batch_size=512, columns=["image", "label"]):
        for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
            name = Path(img.get("path") or f"{prefix}_{n:06d}.JPEG").name
            d = out_dir / WNIDS[label]
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(img["bytes"])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/imagenet")
    ap.add_argument("--train-shards", type=int, default=2)
    ap.add_argument("--cache", default=None, help="Hugging Face cache dir (default: the usual one)")
    args = ap.parse_args()
    out = Path(args.out)

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    val = [f for f in files if f.startswith("data/validation-")]
    train = [f for f in files if f.startswith("data/train-")][: args.train_shards]
    for split, shards in [("val", val), ("train", train)]:
        total = 0
        for i, f in enumerate(shards):
            local = hf_hub_download(REPO, f, repo_type="dataset", cache_dir=args.cache)
            total += export(local, out / split, f"{split}{i:03d}")
            print(f"{split}: {i + 1}/{len(shards)} files, {total} images")
    print("done:", out.resolve())


if __name__ == "__main__":
    main()
