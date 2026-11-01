# ARH pretrained weights

Place the ARH baseline's pretrained HoloNet checkpoints here to evaluate it with `eval_3d_arh.py` (same scenes and metric as `eval_3d.py`) and to time it with `benchmark_3d.py --method arh` (see
`instantrph/arh.py`). One checkpoint per color channel and time-multiplexed frame is expected:

```
{color}_{frame}.pth
```

where `color` is `red`, `green` or `blue` and `frame` runs from 1 to `--tmx` (e.g. `red_1.pth`
… `red_8.pth`, `green_1.pth` … `blue_8.pth` for `--tmx 8`). The released weights have 8 frames
for a 100 mm SLM distance.

These weights are not included in this repository. Obtain them from the ARH (Artifact-Resilient
Real-Time Holography) baseline release and place them in this folder, which is the default
`--arh-model-path`. `scripts/benchmark_3d.sh` times ARH whenever `red_1.pth` is here and logs
the timings to `output/arh/benchmark/`; `scripts/eval_3d_arh.sh` evaluates it and writes to
`output/arh/comparisons/<run>/`.
