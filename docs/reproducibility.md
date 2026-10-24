# Reproducibility

## Configuration and input

`configs/paper.json` is the shared configuration for all three dataset entry points.
Its version, SHA-256 and resolved CLI settings are recorded for every run.

| Component | Configuration |
| --- | --- |
| Input sequence | Four consecutive precomputed feature frames |
| Features per frame | 62 channels × 5 bands = 310 |
| Quaternion packing | Two padded channels per band → 320 stored components |
| Attention | 8 heads, depth 6, post-residual LayerNorm |
| Q-LIF | 320 → 256 → 128 |
| Classifier | 128 → 64 → logits |

SEED and SEED-IV accept MAT features; `--feature de_LDS` selects one feature
family. SEED-V accepts per-trial NPZ features in `[N,310]` or `[N,62,5]` layout,
or prewindowed data with the configured sequence length. Windows never cross
trial boundaries. Inputs are supplied precomputed features; raw-EEG LQFT/Q-DE
extraction and channel ordering belong to the upstream preprocessing stage.

The residual block is `LN(x + Dropout(Attention(x)))`, matching the manuscript.
The 310 in Table 1 counts effective features; quaternion layers store 320
components after padding. The Q-LIF time constant `tau=1` gives decay `exp(-1)`.

## Training objective

Both classification losses use mean cross-entropy. The forward objective is

$$L=L_{cls}+\beta L_{dom},\qquad\beta=0.01.$$

GRL is identity in the forward pass and multiplies the feature gradient by
`-lambda`. For feature extractor θ, emotion classifier φ and discriminator ψ:

$$g_\theta=\nabla_\theta L_{cls}-\beta\lambda\nabla_\theta L_{dom},\quad
g_\phi=\nabla_\phi L_{cls},\quad g_\psi=\beta\nabla_\psi L_{dom}.$$

`--adversarial-weight` sets β. `--grl-lambda` sets the constant or maximum λ;
the default schedule is `lambda(p)=lambda_max*(2/(1+exp(-10*p))-1)`, where p
is completed pretraining updates divided by total updates. Use
`--grl-schedule constant` for a fixed coefficient. The scalar loss contains no
second negative sign or λ multiplier. Fine-tuning uses source classification
and pseudo-label conditional MMD.

## Running and outputs

```bash
python seed-9376/main.py --file-path /data/SEED --session 1 --results-file runs/seed/results.json --checkpoint-dir runs/seed/checkpoints
python seed-iv/main.py --file-path /data/SEED_IV/eeg_feature_smooth --session 3 --results-file runs/seed-iv/results.json --checkpoint-dir runs/seed-iv/checkpoints
python seed-v-89.98/main.py --file-path /data/SEED_V/features --session 3 --results-file runs/seed-v/results.json --checkpoint-dir runs/seed-v/checkpoints
```

`--target 0` evaluates every subject in LOSO order. Positive values select one
subject. Each run saves a `.config.json`, a training `.log`, the results JSON,
final checkpoints, and `subject_XX_predictions.npz` with labels and probabilities.
Mean/std are in fractions, with population standard deviation (`ddof=0`).
Target emotion labels are read only for the final evaluation.

## Results and figures

`paper_reported_results.json` records the manuscript's results.
`figures/hyperparameter_sweep.json` preserves the supplied sensitivity data,
whose SEED-IV values differ from the main table. The plot uses those values
unchanged; dashed boxes mark `configs/paper.json` settings, including β=0.01.
The numerical version difference requires the corresponding original run records.

```bash
python -m pip install -r requirements-plot.txt
python figures/plot_hyperparameters.py
```
