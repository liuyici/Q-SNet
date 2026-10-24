# Validation

Run `python -B -m pytest -q` after installing `requirements-dev.txt`.
Local result on 2026-10-24: **46 passed, 4 dataset-inapplicable cases skipped**.
Tests cover all three entry points: four-frame windowing, post-residual
normalization, the 64-dimensional bottleneck, GRL gradients, target-label
isolation, configuration overrides and saved run artifacts.

The test suite uses synthetic data; real-dataset LOSO results are separate.
GitHub Actions runs the same suite on Python 3.11 with CPU PyTorch.
