# Manuscript and implementation consistency

This correction is based on the maintainer's `cas-dc-template.tex` manuscript
provided on 2026-10-24 and public commit
`bb6268f6c33b740eb21c6c5cf67dc1e02575b19e`.

## Reported results and their provenance

The main README now transcribes the manuscript's SEED-IV results:

| Metric (%) | Manuscript | Previous README |
| --- | ---: | ---: |
| Accuracy | 80.25 ± 10.89 | 78.25 ± 10.89 |
| Macro F1 | 79.31 ± 11.82 | 76.31 ± 11.82 |
| Macro AUC | 87.60 ± 7.59 | 85.60 ± 7.59 |

These are **manuscript-reported values**, not measurements produced by this
correction. The repository does not provide the original fold predictions,
checkpoints and full run configuration needed to establish which historical
run produced either set of values. The old directory name is not experimental
evidence. It has been replaced with the score-independent name `seed-iv/`.
The original release remains available in Git history; historical grid-search
CSVs and analysis images are retained as development artifacts, not evidence
for the fixed-schedule protocol or the corrected manuscript numbers.

`paper_reported_results.json` records the manuscript values and this status.
`final_results.json`, produced by an actual run, contains measured scores in
fractions, per-subject results, population standard deviations (`ddof=0`),
valid subject counts for each metric, the configuration and the source commit.
Multiply measured means and standard deviations by 100 to express percentages.
Each final evaluation also saves `subject_XX_predictions.npz` beside the
checkpoint, containing `y_true`, `y_pred`, `probabilities` and class indices,
so its accuracy/F1/AUC can be audited without selecting a new checkpoint.
Undefined AUCs are `null` and excluded with an explicit count. No reported
accuracy is used as a training target, stopping criterion or output adjustment.
Do not describe the corrected configuration as having reproduced 80.25% until
its complete LOSO experiment has actually been run and the results archived.

## Input, padding and Table 1

The default configuration uses **one** selected five-band feature family, not
the concatenation of `de_LDS` and `de_movingAve`. The SEED-IV loader now honors
`--feature`; `de_LDS` is the default. Its feature width is 62 × 5 = 310.

| Stage | Default shape or width | Meaning |
| --- | --- | --- |
| Supplied features | `[batch, steps, 62, 5]` | 310 effective real features per step |
| Channel padding | `[batch, steps, 64, 5]` | Two zero channels complete the final group |
| Quaternion packing | `[batch, steps, 320]` | 80 quaternions stored as four component blocks |
| QRA | 320 stored components, 8 heads, depth 6 | 10 whole quaternions per head |
| Q-LIF head | 320 stored → 256 → 128 | 310 measured input features plus 10 padding components |
| Temporal pooling | `[batch, 128]` | Mean over temporal steps |
| Classifier | 128 → 64 → number of classes | 64-dimensional bottleneck from Table 1 |

The 310 in Table 1 is an **effective input-feature count**, not a literal
quaternion storage width. The executable head operates on 320 components.
Calling `QuaternionLinear(310, 256)` would silently floor the number of
quaternion inputs in the bundled layer, so it is not a valid correction.
The code rejects 620-dimensional concatenated inputs and the old 640-component
configuration. Channel-major flattened `[batch, steps, 310]` inputs are reshaped
to `[batch, steps, 62, 5]` before the same packing operation; they do not follow
a different quaternion grouping rule.

**Suggested Table 1 clarification:** “Spiking head: 310 effective input features
(320 components after channel-group padding) → 256 → 128; classifier:
128 → 64 → logits.” The current literal `310 → 256 → 128` table requires this
qualification; the code cannot make 310 divisible by four. Eight attention
heads remain valid because 320 is divisible by 4 × 8. The separate 640-width
development setting is no longer a default or an accepted paper configuration.

This repository's entry points consume precomputed feature files. They do not
implement or certify the manuscript's complete raw-EEG → region-aware LQFT →
Q-DE extraction pipeline. In particular, ordinary supplied `de_LDS` features
must not be described as LQFT-derived Q-DE merely because they are packed into
quaternions. The channel order/grouping and feature provenance must be supplied
and recorded upstream. The default sequence construction remains 12 successive
feature frames for SEED and 9 for SEED-IV; those frames are not evidence of the
manuscript's four raw-EEG extraction windows. SEED-V consumes the stored feature
layout. Its 45-trial archives are selected by the requested session (trials
0–14, 15–29 or 30–44); the old loader always selected 15–29 despite its default
`--session 3` argument. These upstream preprocessing details must be reconciled with original
experimental artifacts before claiming full manuscript reproduction.

The two Q-LIF layers use `decay = exp(-1 / tau)`; the default time constant
`tau=1` gives decay `exp(-1)`. Distinguish the time constant from the decay
factor when describing the hyperparameter table. Changing the input family or
adding the classifier bottleneck changes the trained model: old checkpoints
are not compatible and experiments must be retrained.

## Loss, GRL and optimization directions

Let `F_theta` denote the feature extractor, `C_phi` the emotion classifier and
`D_psi` the subject-domain discriminator. Source emotion labels are `y_s`.
Domain labels `d` identify subjects and are available for both source and
target samples; they are not target emotion labels.

Both losses are nonnegative mean cross-entropies:

\[
L_{cls}=-\frac1{n_s}\sum_{i=1}^{n_s}\log C_\phi(F_\theta(x_i^s))_{y_i^s},
\qquad
L_{dom}=-\frac1{n_s+n_t}\sum_{i=1}^{n_s+n_t}
\log D_\psi(F_\theta(x_i))_{d_i}.
\]

The GRL is identity in the forward pass and has a prescribed backward rule:

\[
G_\lambda(z)=z,\qquad
\frac{\partial G_\lambda}{\partial z}=-\lambda I.
\]

The implemented scalar loss is

\[
L_{forward}=L_{cls}+\beta\,
L_{dom}\big(D_\psi(G_\lambda(F_\theta(x))),d\big),\qquad \beta=0.01.
\]

The optimizer receives these gradients (before ordinary weight decay and
momentum):

\[
g_\theta=\nabla_\theta L_{cls}-\beta\lambda\nabla_\theta L_{dom},\quad
g_\phi=\nabla_\phi L_{cls},\quad
g_\psi=\beta\nabla_\psi L_{dom}.
\]

Thus the feature extractor minimizes classification loss while maximizing the
domain loss, and the discriminator minimizes the domain loss. For fixed
`lambda > 0`, the feature/classifier saddle objective can be written as
`min_(theta,phi) max_psi [L_cls - beta*lambda*L_dom]`, **with a discriminator
step-size rescaling by `1/lambda`** to match the code's `beta*grad(L_dom)` update.
The explicit three gradients above are the definition of the implementation;
a plain scalar objective without GRL does not generate them simultaneously.
At `lambda=0` the discriminator still learns, but the domain gradient into the
feature extractor is zero.

`--adversarial-weight` is **beta**, the weight on the scalar domain CE.
`--grl-lambda` is the GRL coefficient (or its maximum), applied only to the
backward gradient entering the extractor. With the default
`--grl-schedule sigmoid`, the existing schedule is made explicit:

\[
\lambda(p)=\lambda_{max}\left(\frac{2}{1+e^{-10p}}-1\right),\quad
\lambda_{max}=1,\quad
p=\frac{\text{completed training updates}}{\text{total pretraining updates}}.
\]

`--grl-schedule constant --grl-lambda VALUE` instead fixes the coefficient.
The effective feature-side adversarial strength is `beta * lambda(p)`.
Neither `-lambda * L_dom` nor a second `lambda` factor belongs in the forward
loss when GRL is already used. A negative forward weight would also train the
discriminator in the wrong direction. This is the required interpretation
and replacement for the manuscript's ambiguous adversarial-loss expression.
It is not mathematically sound to reproduce a double sign reversal just to
make the code resemble that expression.

Fine-tuning uses source emotion CE plus positively weighted conditional MMD
of source features and target features grouped by detached predicted labels.
There is no GRL or target ground-truth emotion loss in that stage.

## Fixed schedule and final evaluation

All three `main.py` entry points, and their `Q_SNet.py` compatibility commands,
use the same policy: 50 pretraining epochs, 200 fine-tuning epochs by default,
then save/freeze the final model and evaluate target emotion labels once.
The target training `TensorDataset` contains inputs only. There is no
target-metric checkpoint selection, early stopping, best-iteration reporting
or target-label hyperparameter search. Any tuning must be specified in
advance or use validation subjects drawn only from the source subjects.
Legacy target-label-dependent code is recoverable from Git history.

Example commands from the repository root (replace dataset paths):

```bash
python -m pip install -r requirements.txt
python seed-9376/main.py --file-path /data/SEED --session 1 --device cuda --results-file runs/seed/results.json --checkpoint-dir runs/seed/checkpoints
python seed-iv/main.py --file-path /data/SEED_IV/eeg_feature_smooth --session 3 --feature de_LDS --device cuda --results-file runs/seed-iv/results.json --checkpoint-dir runs/seed-iv/checkpoints
python seed-v-89.98/main.py --file-path /data/SEED_V/features --session 3 --device cuda --results-file runs/seed-v/results.json --checkpoint-dir runs/seed-v/checkpoints
```

Each command defaults to all subjects; `--target N` selects one one-based
subject index for a diagnostic run, which is not a complete LOSO result.
Use separate output directories for different datasets, sessions, seeds and
configurations. Checkpoints contain `model_state_dict` and configuration;
load the model parameters from that key. A run must be tied to an unchanged
checkout and archived input preprocessing metadata to be reproducible.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check packing and preservation of all 310 features, rejection of old
concatenated inputs, classifier dimensions, forward/backward execution,
the separate beta/lambda gradient effects, selected-feature loading,
target-label isolation, final-checkpoint evaluation and measured aggregation.
They use synthetic data. Passing them does not establish a dataset accuracy.
