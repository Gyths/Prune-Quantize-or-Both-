"""Fix the calibration images that every method uses, so all of them see the same inputs.

    python scripts/data/make_calib_list.py --train ../data/imagenet/train --seed 0 --rounds 5

Draws one image per class from the training pool and shuffles the 1 000 of them with a fixed seed. The order
is kept, so a method that calibrates with N images takes the first N: with one image per class, any prefix
has N different classes. Methods that need more than 1 000 images (block reconstruction uses 1 024) read into a
second round, drawn and shuffled the same way from the images not yet picked and appended after the first, so
the first 1 000 lines do not change with `--rounds`. Writes `splits/calib_seed<seed>.txt`, one `<wnid>/<file>`
per line, relative to the train folder. Nothing in it comes from val, V2 or R.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for deitc
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/imagenet/train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = Path(args.train)
    left = {d.name: sorted(f.name for f in d.iterdir() if f.is_file())
            for d in sorted(p for p in train.iterdir() if p.is_dir())}
    picks = []
    for r in range(args.rounds):
        this_round = []
        for wnid, files in left.items():
            if files:
                f = rng.choice(files)
                files.remove(f)
                this_round.append(f"{wnid}/{f}")
        rng.shuffle(this_round)
        picks += this_round

    out = Path(args.out) / f"calib_seed{args.seed}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picks) + "\n")
    print(f"{len(picks)} images in {args.rounds} rounds from {len(left)} classes -> {out}")


if __name__ == "__main__":
    main()
