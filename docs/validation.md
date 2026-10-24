# Validation of the consistency correction

Local validation on 2026-10-24 used Python 3.11.14, PyTorch
2.10.0.dev20251124+cu128, NumPy, SciPy 1.16.3 and scikit-learn 1.7.2 on Windows.
Tests explicitly used the CPU. The GitHub Actions workflow separately runs the
test suite on Linux with Python 3.11 and the CPU PyTorch package.

Command: `python -B -m pytest -q`

Result: **36 passed, 4 skipped**. The skipped parameterizations are the
SEED-IV-only MAT-loader test applied to the other two datasets and the
SEED-V-only session-loader test applied to the other two datasets.

The suite exercises all three maintained entry points. It checks full-depth
model forward/backward execution, loss and GRL gradient signs/scales, feature
packing, selected-feature loading, final checkpoint creation, target-label
isolation, prediction export, and score aggregation. The label-isolation test
trains twice with identical inputs/seeds and changed target emotion labels;
every saved model tensor is exactly equal across the two runs.

No real SEED, SEED-IV or SEED-V LOSO experiment was run for this correction.
The tests use synthetic inputs and do not substantiate the manuscript's
reported accuracies. The revised input configuration and classifier bottleneck
require new training and archived predictions before numerical reproduction
can be claimed. See [the provenance and remaining preprocessing gaps](reproducibility.md).
