import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def build_model(module, depth=1, dropout=0.0):
    args = module.parse_args([])
    return module.Q_SNetFinal(
        args.quaternion_dim, args.classes, args.hidden_dim, depth,
        args.heads, dropout, args.tau, args.threshold,
    )


def test_packing_preserves_all_measured_features(implementation):
    m = implementation
    x = torch.arange(1, 2 * 3 * 62 * 5 + 1, dtype=torch.float32).reshape(2, 3, 62, 5)
    packed = m.prepare_quaternion_sequence(x)
    assert packed.shape == (2, 3, 320)
    torch.testing.assert_close(packed, m.prepare_quaternion_sequence(x.flatten(2)))
    components = m.split_components(packed).reshape(2, 3, 5, 64)
    restored = components.permute(0, 1, 3, 2)
    torch.testing.assert_close(restored[:, :, :62], x)
    assert torch.count_nonzero(restored[:, :, 62:]) == 0
    assert (packed == 0).sum().item() == 2 * 3 * 10


def test_reject_legacy_concatenation_and_invalid_heads(implementation):
    m = implementation
    for shape in [(2, 9, 620), (2, 9, 62, 10), (2, 9, 308)]:
        with pytest.raises(ValueError):
            m.prepare_quaternion_sequence(torch.zeros(shape))
    with pytest.raises(ValueError):
        m.prepare_quaternion_sequence(torch.zeros(2, 9, 62, 5), 640)
    with pytest.raises(SystemExit):
        m.parse_args(["--quaternion-dim", "640"])
    with pytest.raises(SystemExit):
        m.parse_args(["--heads", "7"])


def test_default_model_shapes_and_backward(implementation):
    m = implementation
    model = build_model(m, depth=6)
    seen = {}
    handles = [
        model.qlif.first.register_forward_hook(lambda _, inputs, output: seen.update(first=output[0].shape)),
        model.qlif.second.register_forward_hook(lambda _, inputs, output: seen.update(second=output[0].shape)),
        model.bottleneck.register_forward_hook(lambda _, inputs, output: seen.update(bottleneck=output.shape)),
    ]
    inputs = torch.randn(2, 3, 62, 5, requires_grad=True)
    features, logits = model(inputs)
    assert seen == {"first": (2, 256), "second": (2, 128), "bottleneck": (2, 64)}
    assert features.shape == (2, 128)
    assert logits.shape == (2, m.parse_args([]).classes)
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("beta,coefficient", [(0.01, 0.7), (0.01, 0.0), (0.0, 0.7)])
def test_grl_reverses_only_feature_gradient_once(implementation, beta, coefficient):
    m = implementation
    torch.manual_seed(17)
    features = torch.randn(5, 4, dtype=torch.double, requires_grad=True)
    classifier = torch.nn.Linear(4, 3).double()
    discriminator = torch.nn.Linear(4, 2).double()
    source_labels = torch.tensor([0, 1, 2])
    domain_labels = torch.tensor([0, 0, 0, 1, 1])
    logits = classifier(features[:3])
    cls = F.cross_entropy(logits, source_labels)
    dom = F.cross_entropy(discriminator(features), domain_labels)
    cls_grad = torch.autograd.grad(cls, (features, classifier.weight), retain_graph=True)
    dom_grad = torch.autograd.grad(dom, (features, discriminator.weight), retain_graph=True)
    loss, cls_returned, dom_returned = m.pretraining_objective(
        logits, source_labels, features, domain_labels, discriminator, beta, coefficient,
    )
    torch.testing.assert_close(loss, cls + beta * dom)
    torch.testing.assert_close(cls_returned, cls)
    torch.testing.assert_close(dom_returned, dom)
    loss.backward()
    torch.testing.assert_close(features.grad, cls_grad[0] - beta * coefficient * dom_grad[0])
    torch.testing.assert_close(classifier.weight.grad, cls_grad[1])
    torch.testing.assert_close(discriminator.weight.grad, beta * dom_grad[1])


def test_grl_schedule_and_validation(implementation):
    m = implementation
    assert m.grl_coefficient(0, 0.6, "sigmoid") == 0
    assert 0 < m.grl_coefficient(0.5, 0.6) < m.grl_coefficient(1.0, 0.6) < 0.6
    assert m.grl_coefficient(0.5, 0.6, "constant") == 0.6
    for flag in ("--adversarial-weight", "--grl-lambda"):
        for value in ("-1", "nan", "inf"):
            with pytest.raises(SystemExit):
                m.parse_args([flag, value])


def test_target_labels_cannot_change_training(implementation, tmp_path, monkeypatch):
    m = implementation
    args = m.parse_args([
        "--device", "cpu", "--batch-size", "3", "--depth", "1",
        "--pretrain-epochs", "1", "--fine-tune-epochs", "1", "--dropout", "0.1",
    ])
    rng = np.random.default_rng(13)
    subjects = [rng.normal(size=(4, 2, 62, 5)).astype(np.float32) for _ in range(3)]
    labels = [np.arange(4) % args.classes for _ in subjects]
    target_index = 1
    original_evaluate = m.evaluate_once
    seen_labels = []

    def checked_evaluation(model, loader, device, classes, predictions_path=None):
        # Evaluation starts only after the final checkpoint exists and is frozen.
        assert (Path(args.checkpoint_dir) / "subject_02_final.pt").is_file()
        assert not any(p.requires_grad for p in model.parameters())
        seen_labels.append(torch.cat([batch[1] for batch in loader]).clone())
        return original_evaluate(model, loader, device, classes, predictions_path=predictions_path)

    monkeypatch.setattr(m, "evaluate_once", checked_evaluation)
    checkpoints = []
    for run in range(2):
        args.checkpoint_dir = str(tmp_path / str(run))
        scores = m.run_subject(args, subjects, labels, target_index, torch.device("cpu"))
        checkpoints.append(torch.load(scores["checkpoint"], weights_only=True)["model_state_dict"])
        labels = [value.copy() for value in labels]
        labels[target_index] = (labels[target_index] + 1) % args.classes
    assert len(seen_labels) == 2  # Once per run, never during either training phase.
    assert not torch.equal(seen_labels[0], seen_labels[1])
    for key in checkpoints[0]:
        torch.testing.assert_close(checkpoints[0][key], checkpoints[1][key], rtol=0, atol=0)
    source, target, _ = m.build_loso_loaders(subjects, labels, target_index, 3, 4)
    assert len(next(iter(source))) == 3
    assert len(next(iter(target))) == 1
    assert sum(batch[0].size(0) for batch in target) == 4  # Retain final partial batch.


def test_aggregate_uses_measured_scores_and_records_missing_auc(implementation):
    m = implementation
    result = m.aggregate_results([
        {"accuracy": 0.4, "macro_f1": 0.2, "macro_auc": None},
        {"accuracy": 0.8, "macro_f1": 0.6, "macro_auc": 0.7},
    ])
    assert result["mean_accuracy"] == pytest.approx(0.6)
    assert result["std_accuracy"] == pytest.approx(0.2)
    assert result["std_macro_f1"] == pytest.approx(0.2)
    assert result["mean_macro_auc"] == pytest.approx(0.7)
    assert result["count_macro_auc"] == 1
    assert result["metric_units"] == "fraction"
    assert result["standard_deviation_ddof"] == 0


def test_legacy_command_delegates_without_monitoring(implementation):
    m = implementation
    assert importlib.import_module("Q_SNet").main is m.main


def test_selected_feature_family_is_honored(implementation, monkeypatch, tmp_path):
    m = implementation
    if m.dataset_folder != "seed-iv":
        pytest.skip("SEED-IV-specific MAT feature selection")
    data = importlib.import_module("data_final")
    monkeypatch.setitem(data.SESSIONS, 3, ["synthetic_subject"])
    # Supply no de_LDS keys: requesting movingAve must not touch the other family.
    selected = {f"de_movingAve{i}": np.full((62, 9, 5), i, dtype=np.float32) for i in range(1, 25)}
    monkeypatch.setattr(data.scipy.io, "loadmat", lambda *a, **kw: selected)
    subjects, labels = data.load_subjects(tmp_path, 3, "de_movingAve")
    assert subjects[0].shape == (24, 9, 62, 5)
    np.testing.assert_array_equal(subjects[0][:, 0, 0, 0], np.arange(1, 25))
    np.testing.assert_array_equal(labels[0], data.LABELS[3])


def test_seed_v_session_selection(implementation, tmp_path, monkeypatch):
    if implementation.dataset_folder != "seed-v-89.98":
        pytest.skip("SEED-V-specific archive session selection")
    data = importlib.import_module("data_final")
    monkeypatch.setattr(data, "SUBJECTS", ["synthetic"])
    features = {i: np.full((1, 310), i, dtype=np.float32) for i in range(45)}
    labels = {i: np.array([i % 5]) for i in range(45)}
    np.savez(tmp_path / "synthetic.npz", data=pickle.dumps(features), label=pickle.dumps(labels))
    for session in (1, 2, 3):
        subjects, _ = data.load_subjects(tmp_path, session, "de_LDS")
        expected = np.arange((session - 1) * 15, session * 15)
        np.testing.assert_array_equal(subjects[0][:, 0], expected)


def test_cli_result_contains_measured_provenance(implementation, tmp_path, monkeypatch):
    m = implementation
    rng = np.random.default_rng(2)
    subjects = [rng.normal(size=(3, 1, 62, 5)).astype(np.float32) for _ in range(2)]
    labels = [np.arange(3) for _ in subjects]
    monkeypatch.setattr(m, "load_subjects", lambda *args: (subjects, labels))
    output = tmp_path / "results" / "run.json"
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--target", "1", "--device", "cpu", "--depth", "1",
        "--batch-size", "3", "--pretrain-epochs", "1", "--fine-tune-epochs", "1",
        "--checkpoint-dir", str(tmp_path / "weights"), "--results-file", str(output),
    ])
    m.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["config"]["target"] == 1
    assert result["architecture"]["effective_input_dim"] == 310
    assert result["architecture"]["stored_quaternion_dim"] == 320
    assert result["count_accuracy"] == 1
    assert result["mean_accuracy"] == result["subjects"][0]["accuracy"]
    assert result["std_accuracy"] == 0
    assert "source_revision" in result
    saved = np.load(result["subjects"][0]["predictions"])
    np.testing.assert_array_equal(saved["y_true"], labels[0])
    assert saved["probabilities"].shape == (3, m.parse_args([]).classes)
    assert (saved["y_true"] == saved["y_pred"]).mean() == result["mean_accuracy"]


def test_reported_results_are_explicitly_not_runtime_measurements():
    root = Path(__file__).resolve().parents[1]
    reported = json.loads((root / "paper_reported_results.json").read_text(encoding="utf-8"))
    assert reported["status"] == "manuscript_reported_not_independently_reproduced"
    assert reported["datasets"]["SEED-IV"]["accuracy"] == {"mean": 80.25, "std": 10.89}
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "80.25 ± 10.89" in readme
    assert "79.31 ± 11.82" in readme
    assert "87.60 ± 7.59" in readme
