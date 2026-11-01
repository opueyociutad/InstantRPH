#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# PSNR / SSIM of a 2D LAE on the DIV2K validation set, with the optical intermediates of image 0,
# written to output/2d/eval/<run>. Defaults to the pretrained LAE; RUN=output/2d/runs/<run> for your own.
#   RUN=data/models_2d/lae/lae_2d_tmx8_100.0mm bash scripts/eval_2d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-data/models_2d/lae/lae_2d_tmx8_100.0mm}"
python eval_2d.py --checkpoint "${RUN}/best.pt" --output-dir "output/2d/eval/$(basename "${RUN}")" --datasets-root data/dataset_2d \
    --batch-size 6 --save-single-sample-plots --save-input-phase-pattern
