# Analysis and figure scripts

Supplementary scripts to inspect and visualize 2D LAE runs. They are not needed to reproduce the
core pipeline (see the top-level README). Run them from the repository root; the `.sh`
launchers do this automatically. Results go to `output/2d/analysis/`
([output/README.md](../output/README.md)).

- **`visualize_lae_evolution.py`** (+ `train_lae_evolution_viz.sh`, `fonts/`): MP4 animations of
  the LAE scale/bias maps, reconstructed holograms and validation PSNR over training, from the
  snapshots that `train_2d.py --viz-every-steps` saves.
- **`plot_distance_sweep.py`** (+ `train_distance_sweep.sh`): trains the LAE at a range of
  propagation distances and plots PSNR / PSNR gain vs. distance.
- **`visualize_lae_params.py`**: renders the learned LAE scale/bias maps of a trained checkpoint
  as PNGs.
