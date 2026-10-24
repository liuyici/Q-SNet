# Q-SNet: SEED-IV

This dataset folder uses the maintained fixed-schedule LOSO-UDA entry point.
See the [main README](../README.md) and [reproducibility notes](../docs/reproducibility.md)
for the architecture, loss derivation, data requirements and remaining evidence gaps.

## Manuscript-reported results

| Metric (%) | Mean ± standard deviation |
| --- | ---: |
| Accuracy | 80.25 ± 10.89 |
| Macro F1 | 79.31 ± 11.82 |
| Macro AUC | 87.60 ± 7.59 |

These values come from the supplied manuscript, not from a new experiment.
See [machine-readable provenance](../paper_reported_results.json).

## Run

From the repository root, using Python 3.10+:

```bash
python -m pip install -r requirements.txt
python seed-iv/main.py --file-path /data/SEED-IV/features --session 3 --device cuda --results-file runs/seed-iv/results.json --checkpoint-dir runs/seed-iv/checkpoints
```

`Q_SNet.py` delegates to the same command. Target emotion labels are evaluated
only after fixed-schedule training and checkpoint saving; they never select
iterations, hyperparameters or pseudo-labels. Training target loaders contain
only inputs.

The default feature width is 62 × 5 = 310, packed into 320 stored quaternion
components. The Q-LIF layers output 256 then 128 features and the classifier
uses a 64-dimensional bottleneck. Pretraining uses positive domain CE weighted
by beta=0.01; the feature-side negative sign is supplied once by GRL.
Actual runs save per-subject measured scores, mean/std, configuration and source
revision. A changed architecture must be retrained and its accuracy remeasured.
