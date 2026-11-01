#!/usr/bin/env bash
# Technical Paper: Instant Random Phase Holography with Parallax and Defocus Cues
# Authors: Oscar Pueyo-Ciutad, Victor Chu, Congli Wang, Ethan Tseng, Florian Schiffers, Albert Redo-Sanchez, Diego Gutierrez, Felix Heide
# Conference: SIGGRAPH Asia 2026
# Project Page: https://opueyociutad.github.io/InstantRPH_Web/
#
# Hologram generation time of InstantRPH (merged 3D LAE) and, if its weights are in data/model_ARH/,
# of the ARH baseline. Logs go to output/3d/benchmark/ and output/arh/benchmark/.
#   RUN=data/models_3d/lae/lae_3d_tmx24_rgb TMX=24 bash scripts/benchmark_3d.sh
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:-data/models_3d/lae/lae_3d_tmx24_rgb}"
TMX="${TMX:-24}"
ARH_TMX="${ARH_TMX:-8}"
mkdir -p output/3d/benchmark output/arh/benchmark
python benchmark_3d.py --method instantrph --run-dir "${RUN}" --tmx "${TMX}" --optimize --use-autocast \
    | tee "output/3d/benchmark/$(basename "${RUN}")_tmx${TMX}.log"
if [ -f data/model_ARH/red_1.pth ]; then
    python benchmark_3d.py --method arh --arh-model-path data/model_ARH --tmx "${ARH_TMX}" ${MPI_DIR:+--mpi-dir "${MPI_DIR}"} \
        | tee "output/arh/benchmark/tmx${ARH_TMX}.log"
fi
