#!/bin/bash
# APHQ-ViT (Wu et al., 2025) on DeiT-S under our protocol: the authors' code at commit 84c9fde with protocol.patch,
# calibrated on the first 128 and reconstructed on the first 1 024 images of splits/calib_seed0.txt.
#
#   bash methods/aphq/run.sh 4bit/best.py "APHQ-ViT W4A4" --reconstruct-mlp   # as published
#   bash methods/aphq/run.sh 4bit/w4a32.py "APHQ-ViT W4 MR" --reconstruct-mlp  # weights-only
#
# As published means the authors' full pipeline (MLP reconstruction with ReLU, calibration, block reconstruction).
# The weights-only run keeps the MLP reconstruction too, so the model it packs has ReLU in its MLPs and is scored with
# `scripts/quantize/evaluate_file.py --mlp-act relu`. The reconstruction does not depend on the bit width, so a run can
# reuse a saved one with `--load-reconstruct-checkpoint`. The patch adds
# the calibration list, the weights-only guards and a last step that scores the model in memory on our three test
# sets (as-published runs into outputs/published_summary.json; weights-only runs into validation_summary.json as a
# check of the packed file, which is what places them on the chart). Run from the repository root with
# ADALOG_PY set to the environment of the AdaLog family (torch 2.2.2, timm 0.9.2).
set -e
CONFIG=$1; NAME=$2; shift 2
REPO=$(pwd)
DATA=${IMAGENET_DIR:-$REPO/../data/imagenet}
WORK=${APHQ_DIR:-$REPO/../third_party/APHQ-ViT-protocol}
ADALOG_PY=${ADALOG_PY:-python}

[ -d "$WORK" ] || { echo "$WORK is missing: run bash scripts/setup_third_party.sh first"; exit 1; }
mkdir -p "$WORK/checkpoint/vit_raw"
[ -f "$WORK/checkpoint/vit_raw/deit_small_patch16_224.bin" ] || \
  "$ADALOG_PY" -c "import timm, torch; torch.save(timm.create_model('deit_small_patch16_224', pretrained=True).state_dict(), '$WORK/checkpoint/vit_raw/deit_small_patch16_224.bin')"

case "$CONFIG" in *a32.py) OUT=validation_summary.json; PER="" ; SUFFIX=", in the method's forward";;
                  *) OUT=published_summary.json; PER="--score-per-image $REPO/outputs/published_per_image.npz"; SUFFIX="";; esac
LOG="$WORK/run_$(echo "$NAME" | tr ' ' '_').log"
(cd "$WORK" && "$ADALOG_PY" test_quant.py --model deit_small --config "./configs/$CONFIG" --dataset "$DATA" \
   --calib-list "$REPO/splits/calib_seed0.txt" --val-batch-size 200 --calibrate --optimize "$@" \
   --repo "$REPO" --score-name "$NAME$SUFFIX" --score-out "$REPO/outputs/$OUT" $PER > "$LOG" 2>&1)
echo "$NAME: $(grep -E 'protocol stages' "$LOG" | tail -1)"
