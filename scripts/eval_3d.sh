#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Evaluate a merged 3D LAE on its validation scenes (or on other data: DATA_ROOT=... SPLIT=all), writing
# the GT / AD / OURS comparisons and phase patterns to output/3d/comparisons/<run>. Defaults to the
# pretrained LAE; RUN=output/3d/runs/<run> for your own. TMX=N evaluates only the first N frames (e.g. TMX=8 to
# compare with scripts/eval_3d_arh.sh).
#   RUN=data/models_3d/lae/lae_3d_tmx8_rgb bash scripts/eval_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-data/models_3d/lae/lae_3d_tmx8_rgb}"
python eval_3d.py --run-dir "${RUN}" --output-dir "output/3d/comparisons/$(basename "${RUN}")" \
    --eval-split "${SPLIT:-auto}" ${DATA_ROOT:+--data-root "${DATA_ROOT}"} ${TMX:+--tmx "${TMX}"} \
    --batch-size 1 --max-batches "${MAX_BATCHES:-5}" --save-phase-patterns
