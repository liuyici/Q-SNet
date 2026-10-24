# Q-SNet: SEED-IV

See the [main README](../README.md) for reported results and the
[shared configuration](../configs/paper.json) for training settings.

```bash
python seed-iv/main.py --file-path /data/SEED-IV/features --device cuda --results-file runs/seed-iv/results.json --checkpoint-dir runs/seed-iv/checkpoints
```

Run from the repository root. `Q_SNet.py` calls the same entry point.
Defaults: four feature frames, post-residual LayerNorm, a 64-dimensional
classifier bottleneck, β=0.01 and fixed-schedule final evaluation.
See [reproducibility](../docs/reproducibility.md) for data formats and outputs.
