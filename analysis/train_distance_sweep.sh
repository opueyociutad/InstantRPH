#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Train the 2D LAE at several propagation distances, then plot PSNR and PSNR gain vs. distance.
#   bash analysis/train_distance_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DISTANCES_MM="${DISTANCES_MM:-40.0 60.0 80.0 100.0 120.0 160.0 200.0}"
SWEEP_DIR="${SWEEP_DIR:-output/2d/analysis/distance_sweep}"
TMX="${TMX:-8}"

for DIST in ${DISTANCES_MM}; do
    python train_2d.py \
        --tmx "${TMX}" --target-distance-mm "${DIST}" --layer-id 0 \
        --epochs "${EPOCHS:-10}" --batch-size 6 --lr 1e-2 --weight-decay 0 \
        --tensorboard --val-every 1 --log-every 0 --output-dir "${SWEEP_DIR}/dist_${DIST}mm"
done
python analysis/plot_distance_sweep.py --sweep-dir "${SWEEP_DIR}"
