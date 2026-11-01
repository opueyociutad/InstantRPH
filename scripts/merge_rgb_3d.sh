#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Merge the R, G and B runs of scripts/train_3d.sh into one RGB run (<PREFIX>_rgb).
#   TMX=8 bash scripts/merge_rgb_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX="${PREFIX:-output/3d/runs/lae_3d_tmx${TMX:-8}}"
python merge_rgb_3d.py --r "${PREFIX}_r" --g "${PREFIX}_g" --b "${PREFIX}_b" --output-run-dir "${PREFIX}_rgb"
