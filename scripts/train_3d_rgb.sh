#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Train the R, G and B runs of scripts/train_3d.sh one after the other, then merge them into one
# RGB run (output/3d/runs/lae_3d_tmx<TMX>_rgb). Variables of scripts/train_3d.sh (TMX, NUM_PROCESSES, ...)
# are passed on; the wavelengths can also run in parallel: WAVELENGTH=r bash scripts/train_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
for WAVELENGTH in r g b; do
    WAVELENGTH="${WAVELENGTH}" bash scripts/train_3d.sh
done
bash scripts/merge_rgb_3d.sh
