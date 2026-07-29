import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SUBJECTS = [f"{index}_123" for index in range(1, 17)]


def load_subjects(file_path, session, feature):
    root = Path(file_path)
    subjects = []
    labels = []
    for name in SUBJECTS:
        archive = np.load(root / f"{name}.npz", allow_pickle=True)
        data = pickle.loads(archive["data"].item())
        trial_labels = pickle.loads(archive["label"].item())
        subject_features = []
        subject_labels = []
        for trial_index in range(15, 30):
            subject_features.append(np.asarray(data[trial_index]))
            subject_labels.append(np.asarray(trial_labels[trial_index]).reshape(-1))
        subjects.append(np.concatenate(subject_features, axis=0).astype(np.float32))
        labels.append(np.concatenate(subject_labels, axis=0).astype(np.int64))
    return subjects, labels


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    deviation = x.std(axis=0, keepdims=True)
    return ((mean - x) / (deviation + 1e-8)).astype(np.float32)


@dataclass
class EvaluationData:
    inputs: torch.Tensor
    labels: torch.Tensor
    batch_size: int

    def loader(self):
        return DataLoader(TensorDataset(self.inputs, self.labels), batch_size=self.batch_size, shuffle=False)


def build_loso_loaders(subjects, labels, target_index, batch_size, evaluation_batch_size):
    source_inputs = []
    source_labels = []
    source_domains = []
    target_inputs = None
    evaluation_labels = None
    for index, (subject_inputs, subject_labels) in enumerate(zip(subjects, labels)):
        normalized = standardize(subject_inputs)
        if index == target_index:
            target_inputs = normalized
            evaluation_labels = subject_labels
        else:
            source_inputs.append(normalized)
            source_labels.append(subject_labels)
            source_domains.append(np.full(subject_labels.shape[0], index, dtype=np.int64))
    source_x = torch.from_numpy(np.concatenate(source_inputs, axis=0)).float()
    source_y = torch.from_numpy(np.concatenate(source_labels, axis=0)).long()
    source_d = torch.from_numpy(np.concatenate(source_domains, axis=0)).long()
    target_x = torch.from_numpy(target_inputs).float()
    evaluation_y = torch.from_numpy(evaluation_labels).long()
    source_loader = DataLoader(TensorDataset(source_x, source_y, source_d), batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(TensorDataset(target_x), batch_size=batch_size, shuffle=True, drop_last=True)
    evaluation_data = EvaluationData(target_x, evaluation_y, evaluation_batch_size)
    return source_loader, target_loader, evaluation_data
