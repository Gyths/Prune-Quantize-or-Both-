"""Fix the calibration images that every method uses, so all of them see the same inputs.

    python make_calib_list.py --train ../data/imagenet/train --seed 0

Draws one image per class from the training pool and shuffles the 1 000 of them with a fixed seed. The order
is kept, so a method that calibrates with N images takes the first N: with one image per class, any prefix
has N different classes. Writes `splits/calib_seed<seed>.txt`, one `<wnid>/<file>` per line, relative to
the train folder. Nothing in it comes from val, V2 or R.
"""
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/imagenet/train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = Path(args.train)
    picks = []
    for d in sorted(p for p in train.iterdir() if p.is_dir()):
        files = sorted(f.name for f in d.iterdir() if f.is_file())
        picks.append(f"{d.name}/{rng.choice(files)}")
    rng.shuffle(picks)

    out = Path(args.out) / f"calib_seed{args.seed}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picks) + "\n")
    print(f"{len(picks)} images from {len(picks)} classes -> {out}")


if __name__ == "__main__":
    main()
