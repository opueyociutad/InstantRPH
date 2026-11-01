#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Short 2D LAE run (100 mm, tmx 8) with a snapshot after every optimizer step, then the
# evolution animations of the LAE maps, two reconstructions and the validation PSNR (needs ffmpeg).
#   bash analysis/train_lae_evolution_viz.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-output/2d/analysis/lae_evolution_viz}"
MAX_STEPS="${MAX_STEPS:-30}"

python train_2d.py \
    --tmx 8 --target-distance-mm 100.0 --layer-id 0 --batch-size 6 --lr 1e-2 --weight-decay 0 \
    --epochs 1 --max-steps "${MAX_STEPS}" --tensorboard --val-every 1 --log-every 0 \
    --viz-every-steps 1 --viz-image-indices 0,2 --output-dir "${OUTPUT_DIR}"
python analysis/visualize_lae_evolution.py --run-dir "${OUTPUT_DIR}" \
    --scale-tone-min 0.25 --scale-tone-max 0.75 --bias-tone-min 0.05 --bias-tone-max 0.45
