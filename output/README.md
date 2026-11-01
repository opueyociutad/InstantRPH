# Output

Everything the scripts write goes here. Their inputs come from [`data/`](../data/README.md).
Folders are created on first use.

```
output/
├── 2d/
│   ├── runs/<run>/                train_2d.py runs
│   ├── eval/<run>/                eval_2d.py / eval_2d_images.py results
│   └── analysis/
│       ├── distance_sweep/        PSNR vs. propagation distance
│       ├── lae_evolution_viz/     LAE / PSNR evolution during training
│       └── lae_params/<run>/      learned scale / bias maps
├── 3d/
│   ├── runs/<run>/                train_3d.py runs (one per wavelength) and merged RGB runs
│   ├── comparisons/<run>/         eval_3d.py: GT / AD / OURS previews, phase patterns, exports
│   └── benchmark/                 InstantRPH hologram-generation timings
├── arh/
│   ├── comparisons/<run>/         eval_3d_arh.py: ARH phase patterns, exports (scenes of <run>)
│   └── benchmark/                 ARH baseline timings
└── video/<run>_<trajectory>/      render_video_3d.py frames
```


## Contents

**Training run** (`2d/runs/<run>/`, `3d/runs/<run>/`):

```
<run>/
├── config.json          every flag of the run (also embedded in the checkpoints)
├── best.pt / last.pt    checkpoints (best = lowest validation loss)
├── tensorboard/         with --tensorboard
├── sweep_metrics.json   2D: best val PSNR, baseline PSNR, PSNR gain, best epoch
└── viz_snapshots/       2D, with --viz-every-steps > 0 (input of the evolution animations)
```

To use a run as a pretrained model, copy its folder to `data/models_2d/lae/` or
`data/models_3d/lae/`.

**2D evaluation** (`2d/eval/<run>/`):

```
<run>/
├── images/                  <idx>_input / _pred_ours / _pred_ad_baseline / _target .png
├── metrics.csv              sample_idx, psnr_ad_baseline, ssim_ad_baseline, psnr_ours, ssim_ours
├── single_sample_plots/     with --save-single-sample-plots: optical intermediates of one image
└── input_phase_patterns/    with --save-input-phase-pattern: the random phases as HSV PNGs
```

**3D comparisons** (`3d/comparisons/<run>/`):

```
<run>/
├── validation_rgb_output.png               per scene: GT, AD, AD+Q, OURS, OURS+Q at up to five focus planes
├── validation_rgb_output.pt                first batch's predicted and target focal stacks
├── validation_target_scene_rgb_means.json  mean RGB of each scene's first target plane
├── validation_phase_patterns/              with --save-phase-patterns: SLM phases in checkpoint_model/ (ours) and raw_phase_identity_uncorrected/ (AD baseline)
└── export/                                 with --export-dir: per-scene color-balanced focal stacks and phase patterns
```
