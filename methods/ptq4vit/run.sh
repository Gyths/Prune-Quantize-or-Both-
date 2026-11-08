#!/bin/bash
# PTQ4ViT under our protocol: W6A6 and W8A8 as published, W6 and W8 weights-only. Run from the repository root with
# ADALOG_PY set to the environment of the AdaLog family (torch 2.2.2, timm 0.9.2) and PY to the project's.
set -e
PY=${PY:-python}; ADALOG_PY=${ADALOG_PY:-python}
REPO=$(pwd); DATA=${IMAGENET_DIR:-$REPO/../data/imagenet}; WORK=${PTQ4VIT_DIR:-$REPO/../third_party/PTQ4ViT}
[ -d "$WORK" ] || { echo "$WORK is missing: run bash scripts/setup_third_party.sh first"; exit 1; }
for spec in "8 8" "6 6" "8 32" "6 32"; do
  set -- $spec
  (cd "$WORK" && "$ADALOG_PY" "$REPO/methods/ptq4vit/protocol.py" --w $1 --a $2 --dataset "$DATA")
  if [ "$2" = 32 ]; then "$PY" scripts/quantize/evaluate_file.py outputs/artifacts/ptq4vit_w$1.safetensors --name "PTQ4ViT W$1"; fi
done
