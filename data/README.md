# Data

Everything the scripts read lives here: datasets, the fixed parts of the models (random phase
tensors, propagation kernels), pretrained LAEs, the ARH baseline weights and the video focus
trajectories. Scripts write nothing here; their results go to [`output/`](../output/README.md).

```
data/
├── dataset_2d/                  2D images (DIV2K)
│   ├── DIV2K_train_HR/          800 training images   (0001.png … 0800.png)
│   └── DIV2K_valid_HR/          100 validation images (0801.png … 0900.png)
├── dataset_3d/                  3D MPI / focal-stack scene pairs (not released yet)
├── dataset_video/               MPI video: clip_XXXX/ layer folders of RGBA frames
├── models_2d/
│   ├── phase/l1_tmx24.pt        2D random phase tensor   [1, 24, RGB, 800, 1280]
│   └── lae/<run>/               pretrained 2D LAEs       (config.json + best.pt)
├── models_3d/
│   ├── kernels/                 compact inter-layer RSD kernels, one folder per layer spacing
│   ├── phase/l21_tmx24.pt       3D random phase tensor   [21, 24, RGB, 800, 1280]
│   └── lae/<run>/               pretrained 3D LAEs       (merged RGB runs: config.json + best.pt)
├── model_ARH/                   ARH baseline HoloNet weights, {red,green,blue}_{frame}.pth
└── video_config/                focus trajectories for render_video_3d.py (*.yaml)
```

Only this README, `models_3d/kernels/`, `video_config/`, `model_ARH/README.md` and the
`.gitkeep` placeholders are tracked by git. You download or generate the rest as described below.

## Folders

| Folder | Contents | How to get it | Read by (flag) |
|---|---|---|---|
| `dataset_2d/` | DIV2K train / validation images. Any folder of `.png/.jpg/.bmp/.tif` images works; they are resized to the SLM resolution on load. | [DIV2K](#div2k) | `train_2d.py`, `eval_2d.py` (`--datasets-root`, `--train-subdir`, `--val-subdir`) |
| `dataset_3d/` | One MPI / focal-stack pair per scene, in any layout of [instantrph/data_3d.py](../instantrph/data_3d.py). | not released yet | `train_3d.py`, `eval_3d.py`, `eval_3d_arh.py`, `benchmark_3d.py` (`--data-root`) |
| `dataset_video/` | One `clip_XXXX/` folder per MPI layer, holding that layer's RGBA frames. | your own MPI video | `render_video_3d.py` (`--data-video-dir`) |
| `models_2d/phase/` | `l1_tmx24.pt`: i.i.d. uniform phases in `[0, 2π)`, `[LAYERS, TMX, RGB, H, W]`. | [Phase tensors](#phase-tensors) | `train_2d.py` (`--phase-file`); eval scripts use the run's file |
| `models_2d/lae/` | Pretrained 2D LAEs, one run folder each, e.g. `lae_2d_tmx8_100.0mm/`. | copy a run from `output/2d/runs/` | `eval_2d.py` (`--checkpoint`), `eval_2d_images.py`, `analysis/visualize_lae_params.py` (`--run-dir`) |
| `models_3d/kernels/` | `<spacing>mm_ksize_2/fullkernel.pt`, see its [README](models_3d/kernels/README.md). | tracked | `instantrph/renderer_3d.py` (picks the folder matching `--layer-spacing-mm`) |
| `models_3d/phase/` | `l21_tmx24.pt`: one phase layer per MPI layer (21). | [Phase tensors](#phase-tensors) | `train_3d.py` (`--phase-file`); eval scripts use the run's file |
| `models_3d/lae/` | Pretrained 3D LAEs (merged RGB runs), e.g. `lae_3d_tmx8_rgb/`. | copy a run from `output/3d/runs/` | `eval_3d.py`, `eval_3d_arh.py`, `benchmark_3d.py` (`--run-dir`), `render_video_3d.py` (`--model-run`) |
| `model_ARH/` | ARH checkpoints, see its [README](model_ARH/README.md). | ARH release | `eval_3d_arh.py`, `benchmark_3d.py --method arh` (`--arh-model-path`) |
| `video_config/` | `constant.yaml`, `with_sweeps.yaml`, `cinematic.yaml`. | tracked | `render_video_3d.py` (`--trajectory-yaml`) |

Every path above is the script's default, resolved from the repository root. Pass the flag to
use another location.

## DIV2K

```bash
cd data/dataset_2d
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_train_HR.zip && unzip DIV2K_valid_HR.zip
cd ../..
```

## Phase tensors

```bash
python generate_phase_pattern.py                                            # data/models_3d/phase/l21_tmx24.pt (6.2 GB)
python generate_phase_pattern.py --layers 1 --out-dir data/models_2d/phase  # data/models_2d/phase/l1_tmx24.pt  (295 MB)
```

The default seed (0) reproduces the paper's tensors bit for bit, and the 2D tensor equals the
first layer of the 3D one. Runs keep only the first `tmx` frames and the top-left crop they
need, so these tensors cover every setting up to `tmx = 24` and 800 × 1280 pixels. For larger
images, generate a larger tensor (`--rows/--cols`) and pass it with `--phase-file`.

## Pretrained LAEs

A pretrained LAE is a training run folder with at least `config.json` and `best.pt` (or
`last.pt`). The evaluation scripts read the optics from `config.json` (tmx, distance, phase
layer, layer spacing). To use one of your own runs as a pretrained model, copy its folder from
`output/2d/runs/` or `output/3d/runs/` into `models_2d/lae/` or `models_3d/lae/`. The launchers
in [scripts/](../scripts/) default to the folder names that training produces:
`lae_2d_tmx8_100.0mm`, `lae_3d_tmx8_rgb` and `lae_3d_tmx24_rgb`.

If the phase file or the 3D data root stored in a run's `config.json` does not exist on this
machine (e.g. a model trained elsewhere), the scripts fall back to `models_2d/phase/`,
`models_3d/phase/` and `dataset_3d/`.
