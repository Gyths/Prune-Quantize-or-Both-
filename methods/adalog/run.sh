#!/bin/bash
# AdaLog (Wu et al., 2024) on DeiT-S under our protocol: the authors' code at commit 75a57d2 with
# weights_only.patch applied, calibrated on the first 32 images of splits/calib_seed0.txt, then packed into a
# real file, checked bit for bit against AdaLog's own weights, and scored on ImageNet-1k val, V2 and R.
#
#   ADALOG_PY=/path/to/env/bin/python PY=/path/to/env/bin/python bash methods/adalog/run.sh
#
# ADALOG_PY runs the authors' code (they use timm 0.9.2; we used torch 2.2.2). PY runs ours (README).
# Run from the repository root.
set -e
REPO=$(pwd)
DATA=${IMAGENET_DIR:-$REPO/../data/imagenet}
WORK=${ADALOG_DIR:-$REPO/../third_party/AdaLog-protocol}
ADALOG_PY=${ADALOG_PY:-python}
PY=${PY:-python}

if [ ! -d "$WORK" ]; then
  git clone https://github.com/GoatWu/AdaLog.git "$WORK"
  git -C "$WORK" checkout 75a57d2
  git -C "$WORK" apply "$REPO/methods/adalog/weights_only.patch"
fi
# AdaLog loads checkpoints/vit_raw/<model>.bin when it exists; these are the same timm weights as in notebook 01.
mkdir -p "$WORK/checkpoints/vit_raw"
"$PY" -c "import timm, torch; torch.save(timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True).state_dict(), '$WORK/checkpoints/vit_raw/deit_small_patch16_224.bin')"

for b in 3 4 6; do
  (cd "$WORK" && "$ADALOG_PY" test_quant.py --model deit_small --config ./configs/w${b}a32.py --dataset "$DATA" \
     --calib-list "$REPO/splits/calib_seed0.txt" --val-batch-size 200 --calibrate > "run_w${b}.log" 2>&1)
  ckpt=$(ls -t "$WORK"/checkpoints/quant_result/*/deit_small_w${b}_a32_s32_calibsize_32.pth | head -1)
  "$PY" pack_published.py --method adalog --bits $b --ckpt "$ckpt" --out outputs/artifacts/adalog_w${b}.safetensors
  "$PY" evaluate_file.py outputs/artifacts/adalog_w${b}.safetensors --name "AdaLog W$b"
done
