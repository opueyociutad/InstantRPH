#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Evaluate the ARH baseline (weights in data/model_ARH/) on the same scenes, focus planes and metric as
# scripts/eval_3d.sh, writing its phase patterns to output/arh/comparisons/<run>. For an apples-to-apples
# comparison, run both launchers with the same RUN (and SPLIT / DATA_ROOT / MAX_BATCHES), the LAE at ARH's 8 frames:
#   RUN=data/models_3d/lae/lae_3d_tmx24_rgb bash scripts/eval_3d_arh.sh
#   RUN=data/models_3d/lae/lae_3d_tmx24_rgb TMX=8 bash scripts/eval_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-data/models_3d/lae/lae_3d_tmx8_rgb}"
python eval_3d_arh.py --run-dir "${RUN}" --output-dir "output/arh/comparisons/$(basename "${RUN}")" \
    --eval-split "${SPLIT:-auto}" ${DATA_ROOT:+--data-root "${DATA_ROOT}"} \
    --batch-size 1 --max-batches "${MAX_BATCHES:-5}" --save-phase-patterns
