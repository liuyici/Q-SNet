import argparse
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from core_qnn.quaternion_layers import QuaternionLinear
from data_final import build_loso_loaders, load_subjects


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def hamilton(a, b):
    ar, ai, aj, ak = a.unbind(dim=-1)
    br, bi, bj, bk = b.unbind(dim=-1)
    return torch.stack((
        ar * br - ai * bi - aj * bj - ak * bk,
        ar * bi + ai * br + aj * bk - ak * bj,
        ar * bj - ai * bk + aj * br + ak * bi,
        ar * bk + ai * bj - aj * bi + ak * br,
    ), dim=-1)


def conjugate(q):
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def split_components(x):
    return torch.stack(x.chunk(4, dim=-1), dim=-1)


def merge_components(q):
    return torch.cat(q.unbind(dim=-1), dim=-1)


# Table 1 counts real, measured features: 62 channels x 5 bands = 310.
# Two padded channels complete the last quaternion in each of the five bands.
INPUT_CHANNELS = 62
INPUT_BANDS = 5
INPUT_DIM = INPUT_CHANNELS * INPUT_BANDS
QUATERNION_DIM = 4 * math.ceil(INPUT_CHANNELS / 4) * INPUT_BANDS


def prepare_quaternion_sequence(x, quaternion_dim=QUATERNION_DIM):
    """Pack [B,T,62,5] (or channel-major [B,T,310]) into 320 components.

    Channels must already be in the intended grouping order. This function
    packs supplied features; it does not compute raw-EEG LQFT/Q-DE features.
    """
    if quaternion_dim != QUATERNION_DIM:
        raise ValueError("The 62-channel, five-band configuration requires 320 stored components")
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if x.ndim == 3:
        if x.size(-1) != INPUT_DIM:
            raise ValueError("Expected 310 channel-major features; do not concatenate feature families")
        x = x.reshape(*x.shape[:2], INPUT_CHANNELS, INPUT_BANDS)
    if x.ndim != 4 or tuple(x.shape[-2:]) != (INPUT_CHANNELS, INPUT_BANDS):
        raise ValueError("Expected [batch, steps, 62 channels, 5 bands]")
    if x.size(1) == 0:
        raise ValueError("At least one temporal step is required")
    batch, steps = x.shape[:2]
    x = F.pad(x, (0, 0, 0, 2))
    q = x.permute(0, 1, 3, 2).contiguous().reshape(batch, steps, -1, 4)
    return merge_components(q)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("GRL lambda must be finite and nonnegative")
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def grl_coefficient(progress, maximum=1.0, schedule="sigmoid"):
    """lambda(p); beta is a separate, positive weight on the domain CE."""
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("GRL lambda must be finite and nonnegative")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Training progress must lie in [0, 1]")
    if schedule == "constant":
        return maximum
    if schedule == "sigmoid":
        return maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
    raise ValueError(f"Unknown GRL schedule: {schedule}")


def pretraining_objective(source_logits, source_labels, domain_features,
                          domain_labels, discriminator, beta, grl_lambda):
    """Forward: L_cls + beta L_dom; feature domain gradient: -beta lambda grad(L_dom).

    Both cross-entropies use mean reduction. GRL reverses only the gradient
    into features; the discriminator minimizes beta * L_dom. Do not put an
    additional minus sign or lambda multiplier on the scalar domain loss.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Adversarial weight beta must be finite and nonnegative")
    cls_loss = F.cross_entropy(source_logits, source_labels, reduction="mean")
    reversed_features = GradientReverse.apply(domain_features, grl_lambda)
    domain_logits = discriminator(reversed_features)
    domain_loss = F.cross_entropy(domain_logits, domain_labels, reduction="mean")
    return cls_loss + beta * domain_loss, cls_loss, domain_loss


class QuaternionRotationAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if heads <= 0 or dim <= 0 or dim % (4 * heads) != 0:
            raise ValueError("Quaternion dimension must be divisible by four times the number of heads")
        self.dim = dim
        self.heads = heads
        self.units_per_head = dim // (4 * heads)
        self.queries = QuaternionLinear(dim, dim, bias=False)
        self.keys = QuaternionLinear(dim, dim, bias=False)
        self.values = QuaternionLinear(dim, dim, bias=False)
        self.projection = QuaternionLinear(dim, dim, bias=False)
        axis = F.normalize(torch.randn(heads, self.units_per_head, 3), dim=-1)
        angle = torch.empty(heads, self.units_per_head).uniform_(-math.pi, math.pi)
        rotation = torch.cat((torch.cos(angle / 2).unsqueeze(-1), axis * torch.sin(angle / 2).unsqueeze(-1)), dim=-1)
        self.rotation = nn.Parameter(rotation)
        self.dropout = nn.Dropout(dropout)

    def to_heads(self, x):
        batch, steps, _ = x.shape
        q = split_components(x)
        return q.reshape(batch, steps, self.heads, self.units_per_head, 4).permute(0, 2, 1, 3, 4)

    def from_heads(self, q):
        batch, _, steps, _, _ = q.shape
        q = q.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, steps, -1, 4)
        return merge_components(q)

    def forward(self, x):
        queries = self.to_heads(self.queries(x))
        keys = self.to_heads(self.keys(x))
        values = self.to_heads(self.values(x))
        rotation = F.normalize(self.rotation, dim=-1).unsqueeze(0).unsqueeze(2)
        values = hamilton(hamilton(rotation, values), conjugate(rotation))
        scores = hamilton(queries.unsqueeze(3), keys.unsqueeze(2)).sum(dim=-2)
        attention = F.softmax(scores / math.sqrt(self.units_per_head * 4), dim=3)
        attention = self.dropout(attention)
        output = hamilton(attention.unsqueeze(-2), values.unsqueeze(2)).sum(dim=3)
        return self.projection(self.from_heads(output))


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = QuaternionRotationAttention(dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.attention(self.norm(x)))


class QuaternionLIFLayer(nn.Module):
    def __init__(self, input_dim, output_dim, tau, threshold, surrogate_width):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or input_dim % 4 or output_dim % 4:
            raise ValueError("Q-LIF stored dimensions must be positive multiples of four")
        if tau <= 0 or threshold <= 0 or surrogate_width <= 0:
            raise ValueError("Q-LIF tau, threshold and surrogate width must be positive")
        self.projection = QuaternionLinear(input_dim, output_dim)
        self.output_dim = output_dim
        self.register_buffer("decay", torch.tensor(math.exp(-1.0 / tau)))
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("surrogate_width", torch.tensor(float(surrogate_width)))

    def forward(self, current, membrane):
        membrane = self.decay * membrane + self.projection(current)
        q = split_components(membrane)
        magnitude = torch.linalg.vector_norm(q, dim=-1)
        soft = torch.sigmoid((magnitude - self.threshold) / self.surrogate_width)
        hard = (magnitude >= self.threshold).to(membrane.dtype)
        gate = hard.detach() - soft.detach() + soft
        spike = merge_components(gate.unsqueeze(-1).expand_as(q))
        membrane = membrane * (1.0 - spike)
        return spike, membrane


class QLIFEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tau, threshold, dropout):
        super().__init__()
        self.first = QuaternionLIFLayer(input_dim, hidden_dim, tau, threshold, 0.4)
        self.second = QuaternionLIFLayer(hidden_dim, 128, tau, threshold, 0.4)
        self.dropout = nn.Dropout(dropout)
        self.feature = nn.Sequential(QuaternionLinear(128, 128), nn.ELU())
        self.shortcut = nn.Linear(128, 128)
        self.radius = 10.0

    def forward(self, x):
        first_state = x.new_zeros(x.size(0), self.first.output_dim)
        second_state = x.new_zeros(x.size(0), self.second.output_dim)
        outputs = []
        for current in x.unbind(dim=1):
            first_spike, first_state = self.first(current, first_state)
            second_spike, second_state = self.second(self.dropout(first_spike), second_state)
            outputs.append(second_spike)
        pooled = torch.stack(outputs, dim=1).mean(dim=1)
        feature = self.feature(pooled) + self.shortcut(pooled)
        return self.radius * F.normalize(feature, dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, input_dim, classes, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, input_dim))
        self.scale = scale
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return self.scale * F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))


class Q_SNetFinal(nn.Module):
    def __init__(self, quaternion_dim, classes, hidden_dim, depth, heads, dropout, tau, threshold):
        super().__init__()
        if quaternion_dim != QUATERNION_DIM:
            raise ValueError("310 measured features require 320 stored quaternion components")
        if depth < 1:
            raise ValueError("QRA depth must be positive")
        self.input_dim = INPUT_DIM
        self.quaternion_dim = quaternion_dim
        self.encoder = nn.Sequential(*[EncoderBlock(quaternion_dim, heads, dropout) for _ in range(depth)])
        self.qlif = QLIFEncoder(quaternion_dim, hidden_dim, tau, threshold, dropout)
        self.bottleneck = nn.Sequential(nn.Linear(128, 64), nn.ELU())
        self.classifier = CosineClassifier(64, classes)

    def forward(self, x):
        x = prepare_quaternion_sequence(x, self.quaternion_dim)
        feature = self.qlif(self.encoder(x))
        return feature, self.classifier(self.bottleneck(feature))


class DomainDiscriminator(nn.Module):
    def __init__(self, domains):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Dropout(0.3), nn.Linear(64, domains))

    def forward(self, x):
        return self.network(x)


def conditional_mmd(source, target, source_labels, target_logits, classes, confidence):
    probabilities = F.softmax(target_logits.detach(), dim=1)
    target_confidence, pseudo_labels = probabilities.max(dim=1)
    losses = []
    for class_index in range(classes):
        source_mask = source_labels == class_index
        target_mask = (pseudo_labels == class_index) & (target_confidence >= confidence)
        if source_mask.any() and target_mask.any():
            difference = source[source_mask].mean(dim=0) - target[target_mask].mean(dim=0)
            losses.append(difference.square().mean())
    if not losses:
        return source.sum() * 0.0
    return torch.stack(losses).mean()


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


class FinalProtocol:
    def __init__(self, model, classes, device, learning_rate, weight_decay, mmd_weight, confidence, domains, target_domain, adversarial_weight, grl_lambda=1.0, grl_schedule="sigmoid"):
        self.model = model.to(device)
        self.classes = classes
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mmd_weight = mmd_weight
        self.confidence = confidence
        self.domains = domains
        self.target_domain = target_domain
        self.adversarial_weight = adversarial_weight
        self.grl_lambda = grl_lambda
        self.grl_schedule = grl_schedule
        self.discriminator = DomainDiscriminator(domains).to(device)

    def pretrain(self, source_loader, target_loader, epochs):
        parameters = list(self.model.parameters()) + list(self.discriminator.parameters())
        optimizer = torch.optim.SGD(parameters, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        if not len(source_loader) or not len(target_loader):
            raise ValueError("Both source and unlabeled target loaders must have batches")
        steps = max(len(source_loader), len(target_loader))
        for epoch in range(epochs):
            self.model.train()
            self.discriminator.train()
            source_iterator = iter(source_loader)
            target_iterator = iter(target_loader)
            running = 0.0
            for step in range(steps):
                source_batch, source_iterator = next_batch(source_iterator, source_loader)
                target_batch, target_iterator = next_batch(target_iterator, target_loader)
                source_x, source_y, source_domain = source_batch
                target_x = target_batch[0]
                source_x = source_x.to(self.device)
                source_y = source_y.to(self.device)
                source_domain = source_domain.to(self.device)
                target_x = target_x.to(self.device)
                source_feature, source_logits = self.model(source_x)
                target_feature, _ = self.model(target_x)
                progress = (epoch * steps + step + 1) / (epochs * steps)
                strength = grl_coefficient(progress, self.grl_lambda, self.grl_schedule)
                domain_feature = torch.cat((source_feature, target_feature), dim=0)
                domain_labels = torch.cat((source_domain, torch.full((target_feature.size(0),), self.target_domain, dtype=torch.long, device=self.device)))
                loss, cls_loss, domain_loss = pretraining_objective(
                    source_logits, source_y, domain_feature, domain_labels,
                    self.discriminator, self.adversarial_weight, strength,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"pretrain epoch {epoch + 1:03d}/{epochs:03d} loss {running / steps:.6f}")

    def fine_tune(self, source_loader, target_loader, epochs):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        if not len(source_loader) or not len(target_loader):
            raise ValueError("Both source and unlabeled target loaders must have batches")
        steps = max(len(source_loader), len(target_loader))
        for epoch in range(epochs):
            self.model.train()
            source_iterator = iter(source_loader)
            target_iterator = iter(target_loader)
            running = 0.0
            for step in range(steps):
                source_batch, source_iterator = next_batch(source_iterator, source_loader)
                target_batch, target_iterator = next_batch(target_iterator, target_loader)
                source_x, source_y, _ = source_batch
                target_x = target_batch[0]
                source_x = source_x.to(self.device)
                source_y = source_y.to(self.device)
                target_x = target_x.to(self.device)
                source_feature, source_logits = self.model(source_x)
                target_feature, target_logits = self.model(target_x)
                alignment = conditional_mmd(
                    source_feature,
                    target_feature,
                    source_y,
                    target_logits,
                    self.classes,
                    self.confidence,
                )
                loss = F.cross_entropy(source_logits, source_y) + self.mmd_weight * alignment
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"fine-tune epoch {epoch + 1:03d}/{epochs:03d} loss {running / steps:.6f}")


def evaluate_once(model, loader, device, classes, predictions_path=None):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            _, logits = model(x.to(device))
            outputs.append(logits.cpu())
            labels.append(y.cpu())
    logits = torch.cat(outputs, dim=0)
    y_true = torch.cat(labels, dim=0).numpy()
    probabilities = F.softmax(logits, dim=1).numpy()
    y_pred = probabilities.argmax(axis=1)
    if predictions_path is not None:
        np.savez_compressed(predictions_path, y_true=y_true, y_pred=y_pred,
                            probabilities=probabilities, classes=np.arange(classes))
    try:
        if len(np.unique(y_true)) != classes:
            auc = None  # Macro one-vs-rest AUC requires both labels for every class.
        else:
            binary = label_binarize(y_true, classes=np.arange(classes))
            auc = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
            if not math.isfinite(auc):
                auc = None
    except ValueError:
        auc = None
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(classes), average="macro", zero_division=0)),
        "macro_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(classes)).tolist(),
    }


def run_subject(args, subjects, labels, target_index, device):
    source_loader, target_loader, evaluation_data = build_loso_loaders(
        subjects,
        labels,
        target_index,
        args.batch_size,
        args.evaluation_batch_size,
    )
    set_seed(args.seed)
    model = Q_SNetFinal(
        args.quaternion_dim,
        args.classes,
        args.hidden_dim,
        args.depth,
        args.heads,
        args.dropout,
        args.tau,
        args.threshold,
    )
    protocol = FinalProtocol(
        model,
        args.classes,
        device,
        args.learning_rate,
        args.weight_decay,
        args.mmd_weight,
        args.pseudo_label_confidence,
        len(subjects),
        target_index,
        args.adversarial_weight,
        args.grl_lambda,
        args.grl_schedule,
    )
    protocol.pretrain(source_loader, target_loader, args.pretrain_epochs)
    protocol.fine_tune(source_loader, target_loader, args.fine_tune_epochs)
    checkpoint_directory = Path(args.checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"subject_{target_index + 1:02d}_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                "target_subject": target_index + 1, "input_dim": INPUT_DIM,
                "quaternion_dim": QUATERNION_DIM}, checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evaluation_loader = evaluation_data.loader()
    predictions = checkpoint_directory / f"subject_{target_index + 1:02d}_predictions.npz"
    metrics = evaluate_once(model, evaluation_loader, device, args.classes, predictions_path=predictions)
    metrics["predictions"] = str(predictions)
    metrics["subject"] = target_index + 1
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, default="/home/lyc/research/research_5/DA/ExtractedQuantFeatures/")
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--feature", type=str, default="de_LDS")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--quaternion-dim", type=int, choices=[QUATERNION_DIM], default=QUATERNION_DIM,
                        help="320 stored components = (62 + 2 padded channels) x 5 bands")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--mmd-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.01,
                        help="beta: positive weight on mean domain cross-entropy")
    parser.add_argument("--grl-lambda", type=float, default=1.0,
                        help="lambda: constant GRL coefficient or maximum of its schedule")
    parser.add_argument("--grl-schedule", choices=["constant", "sigmoid"], default="sigmoid")
    parser.add_argument("--pseudo-label-confidence", type=float, default=0.0)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--fine-tune-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--evaluation-batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="final_checkpoints")
    parser.add_argument("--results-file", type=str, default="final_results.json")
    args = parser.parse_args(argv)
    for name in ("pretrain_epochs", "fine_tune_epochs", "batch_size", "evaluation_batch_size", "depth", "heads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target < 0:
        parser.error("--target must be 0 (all subjects) or a positive subject number")
    if not math.isfinite(args.adversarial_weight) or args.adversarial_weight < 0:
        parser.error("--adversarial-weight must be finite and nonnegative")
    if not math.isfinite(args.grl_lambda) or args.grl_lambda < 0:
        parser.error("--grl-lambda must be finite and nonnegative")
    if args.quaternion_dim % (4 * args.heads):
        parser.error("--heads must divide the 80 quaternion units")
    return args


def aggregate_results(results):
    """Summarize measured subject scores, in fractions, using population std."""
    summary = {"standard_deviation_ddof": 0, "metric_units": "fraction"}
    for metric in ("accuracy", "macro_f1", "macro_auc"):
        scores = [item[metric] for item in results if item[metric] is not None]
        summary[f"mean_{metric}"] = float(np.mean(scores)) if scores else None
        summary[f"std_{metric}"] = float(np.std(scores, ddof=0)) if scores else None
        summary[f"count_{metric}"] = len(scores)
    return summary


def source_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    args = parse_args()
    device = torch.device(args.device)
    subjects, labels = load_subjects(args.file_path, args.session, args.feature)
    if args.target > len(subjects):
        raise ValueError(f"Target subject {args.target} exceeds {len(subjects)} loaded subjects")
    target_indices = [args.target - 1] if args.target else list(range(len(subjects)))
    results = []
    for target_index in target_indices:
        print(f"target subject {target_index + 1}")
        results.append(run_subject(args, subjects, labels, target_index, device))
    output = {
        "protocol": "fixed training schedule with one final target-label evaluation",
        "subjects": results,
        "config": vars(args),
        "source_revision": source_revision(),
        "architecture": {"effective_input_dim": INPUT_DIM, "stored_quaternion_dim": QUATERNION_DIM,
                         "spiking_hidden_dim": args.hidden_dim, "spiking_output_dim": 128,
                         "classifier_bottleneck_dim": 64, "attention_heads": args.heads},
        **aggregate_results(results),
    }
    result_path = Path(args.results_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)
    result_path.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
