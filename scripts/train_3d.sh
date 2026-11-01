#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# End-to-end 3D LAE training with accelerate, one wavelength (or all three) per run.
#   WAVELENGTH=r bash scripts/train_3d.sh
# scripts/train_3d_rgb.sh trains r, g and b and merges them into one RGB run.
set -euo pipefail
cd "$(dirname "$0")/.."

WAVELENGTH="${WAVELENGTH:-rgb}"            # rgb, r, g or b
TMX="${TMX:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"        # GPUs
MIXED_PRECISION="${MIXED_PRECISION:-no}"   # no, fp16 or bf16
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-5}"
DATA_ROOT="${DATA_ROOT:-data/dataset_3d}"
OUTPUT_DIR="${OUTPUT_DIR:-output/3d/runs/lae_3d_tmx${TMX}_${WAVELENGTH}}"
RESUME="${RESUME:-}"                       # e.g. ${OUTPUT_DIR}/last.pt

export NCCL_P2P_DISABLE=1
LAUNCH=(accelerate launch --num_processes "${NUM_PROCESSES}" --mixed_precision "${MIXED_PRECISION}")
[ "${NUM_PROCESSES}" -gt 1 ] && LAUNCH+=(--multi_gpu)

"${LAUNCH[@]}" train_3d.py \
    --use-accelerate --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}" \
    --train-wavelength "${WAVELENGTH}" --tmx "${TMX}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" --lr 1e-2 --num-workers 4 --val-fraction 0.05 \
    --focus-start-mm 97.75 --focus-end-mm 100.0 --enable-augmentation \
    --tensorboard --tb-max-images 2 \
    ${RESUME:+--resume-checkpoint "${RESUME}"}
