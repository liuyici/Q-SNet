import importlib
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("seed-9376", "seed-iv", "seed-v-89.98")


@pytest.fixture(params=DATASETS)
def implementation(request, monkeypatch):
    # Each historical dataset folder exposes the same top-level module names.
    names = ("main", "data_final", "Q_SNet", "core_qnn")
    for key in list(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in names):
            del sys.modules[key]
    monkeypatch.syspath_prepend(str(ROOT / request.param))
    torch.set_num_threads(1)
    module = importlib.import_module("main")
    module.dataset_folder = request.param
    yield module
    for key in list(sys.modules):
        if any(key == name or key.startswith(name + ".") for name in names):
            del sys.modules[key]
