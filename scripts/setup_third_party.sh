#!/bin/bash
# Clone the published methods' code next to this repository, at the commits we used, and apply our patches.
#
#   bash scripts/setup_third_party.sh              # into ../third_party
#   THIRD_PARTY=/some/dir bash scripts/setup_third_party.sh
#
# None of this code is included in the repository, because each project has its own licence (VBP is AGPL-3.0).
# The folder names are the ones `methods/*/run.sh` and `deitc/pruning.py` look for. A folder that already exists is
# left as it is. Run from the repository root.
set -e
REPO=$(pwd)
THIRD=${THIRD_PARTY:-$REPO/../third_party}
mkdir -p "$THIRD"

get() {  # folder, url, commit, patch (or empty)
  local dir="$THIRD/$1"
  if [ -d "$dir" ]; then echo "$1: already there, left as it is"; return; fi
  git clone --quiet "$2" "$dir"
  git -C "$dir" checkout --quiet "$3"
  [ -n "$4" ] && git -C "$dir" apply "$REPO/$4"
  echo "$1: $3${4:+ with $4}"
}

get AdaLog-protocol   https://github.com/GoatWu/AdaLog.git                           75a57d2 methods/adalog/weights_only.patch
get APHQ-ViT-protocol https://github.com/GoatWu/APHQ-ViT.git                         84c9fde methods/aphq/protocol.patch
get FIMA-Q-protocol   https://github.com/ShiheWang/FIMA-Q.git                        64c4b00 methods/fimaq/protocol.patch
get RepQ-ViT          https://github.com/zkkli/RepQ-ViT.git                          a160561 ""
get PTQ4ViT           https://github.com/hahnyuan/PTQ4ViT.git                        b42c790 ""
get GRAIL             https://github.com/TWWinde/GRAIL_Compensation.git              ba5e48b ""
get VBP               https://github.com/boschresearch/variance-based-pruning.git    b6834b4 ""
# GemLite's low-bit kernels, used only by `scripts/measure/gemlite_test.py` (run it with PYTHONPATH=../third_party/gemlite).
get gemlite           https://github.com/dropbox/gemlite.git                         f649e80 ""
