#!/bin/bash
# RepQ-ViT under our protocol: W4A4 and W6A6 as published, W4 and W6 weights-only. Run from the repository root with
# ADALOG_PY set to the environment of the AdaLog family (torch 2.2.2, timm 0.9.2) and PY to the project's.
set -e
PY=${PY:-python}; ADALOG_PY=${ADALOG_PY:-python}
REPO=$(pwd); DATA=${IMAGENET_DIR:-$REPO/../data/imagenet}; WORK=${REPQ_DIR:-$REPO/../third_party/RepQ-ViT}
[ -d "$WORK" ] || { echo "$WORK is missing: run bash scripts/setup_third_party.sh first"; exit 1; }
for spec in "4 4" "6 6" "4 32" "6 32"; do
  set -- $spec
  (cd "$WORK/classification" && "$ADALOG_PY" "$REPO/methods/repq/protocol.py" --w $1 --a $2 --dataset "$DATA")
  if [ "$2" = 32 ]; then "$PY" scripts/quantize/evaluate_file.py outputs/artifacts/repq_w$1.safetensors --name "RepQ-ViT W$1"; fi
done
