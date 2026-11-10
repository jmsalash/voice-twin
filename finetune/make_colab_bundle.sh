#!/usr/bin/env bash
# Packs what colab_train.ipynb needs into one zip: the training code and the encoded data (work/encoded).
# Upload the zip to Google Drive, in a folder called omni-cv.
#   finetune/make_colab_bundle.sh [OUT.zip]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${FT_WORK:-$HERE/work}
OUT=$(realpath -m "${1:-$WORK/omni_cv_train.zip}")
[ -f "$WORK/encoded/train.pt" ] || { echo "no $WORK/encoded/train.pt: run 01 to 03 first" >&2; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/finetune/work/encoded" "$TMP/model"
cp "$HERE"/{04_train.py,common.py,omni_train.py} "$TMP/finetune/"
cp "$HERE/../model/omni_lm.py" "$TMP/model/"
cp "$WORK"/encoded/*.pt "$TMP/finetune/work/encoded/"
rm -f "$OUT"
(cd "$TMP" && zip -qr0 "$OUT" .)
echo "$OUT ($(du -h "$OUT" | cut -f1))"
