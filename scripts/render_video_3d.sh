#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Render an MPI video (clip_XXXX/ layer folders) with a merged 3D LAE along a focus trajectory,
# to output/video/<run>_<trajectory>.
#   DATA_VIDEO_DIR=data/dataset_video RUN=data/models_3d/lae/lae_3d_tmx8_rgb TRAJECTORY=data/video_config/with_sweeps.yaml bash scripts/render_video_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-data/models_3d/lae/lae_3d_tmx8_rgb}"
TRAJECTORY="${TRAJECTORY:-data/video_config/constant.yaml}"
python render_video_3d.py \
    --data-video-dir "${DATA_VIDEO_DIR:-data/dataset_video}" --model-run "${RUN}" --trajectory-yaml "${TRAJECTORY}" \
    --output-dir "${OUTPUT_DIR:-output/video/$(basename "${RUN}")_$(basename "${TRAJECTORY}" .yaml)}" \
    --batch-size "${BATCH_SIZE:-5}" --device "${DEVICE:-cuda}"
