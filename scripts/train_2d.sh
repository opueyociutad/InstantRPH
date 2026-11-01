#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# 2D LAE training on DIV2K (paper settings) -> output/2d/runs/lae_2d_tmx<TMX>_<DIST>mm.
#   bash scripts/train_2d.sh
#   TMX=4 DIST=120.0 bash scripts/train_2d.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TMX="${TMX:-8}"
DIST="${DIST:-100.0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-6}"

python train_2d.py \
    --datasets-root data/dataset_2d \
    --tmx "${TMX}" --target-distance-mm "${DIST}" --layer-id 0 \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 1e-2 \
    --tensorboard --val-every 1 --log-every 0 \
    --output-dir "output/2d/runs/lae_2d_tmx${TMX}_${DIST}mm"
