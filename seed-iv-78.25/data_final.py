from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset


SESSIONS = {
    1: [
        "1_20160518", "2_20150915", "3_20150919", "4_20151111", "5_20160406",
        "6_20150507", "7_20150715", "8_20151103", "9_20151028", "10_20151014",
        "11_20150916", "12_20150725", "13_20151115", "14_20151205", "15_20150508",
    ],
    2: [
        "1_20161125", "2_20150920", "3_20151018", "4_20151118", "5_20160413",
        "6_20150511", "7_20150717", "8_20151110", "9_20151119", "10_20151021",
        "11_20150921", "12_20150804", "13_20151125", "14_20151208", "15_20150514",
    ],
    3: [
        "1_20161126", "2_20151012", "3_20151101", "4_20151123", "5_20160420",
        "6_20150512", "7_20150721", "8_20151117", "9_20151209", "10_20151023",
        "11_20151011", "12_20150807", "13_20161130", "14_20151215", "15_20150527",
    ],
}


LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def windows(features, size):
    return np.stack([features[index:index + size] for index in range(features.shape[0] - size + 1)])


def load_subjects(file_path, session, feature):
    root = Path(file_path) / str(session)
    subjects = []
    labels = []
    for name in SESSIONS[session]:
        data = scipy.io.loadmat(root / f"{name}.mat", mat_dtype=True)
        subject_features = []
        subject_labels = []
        for trial_index, trial_label in enumerate(LABELS[session]):
            first = data[f"de_LDS{trial_index + 1}"]
            second = data[f"de_movingAve{trial_index + 1}"]
            trial = np.swapaxes(np.concatenate((first, second), axis=2), 0, 1)
            trial = trial[-100:]
            trial_windows = windows(trial, 9)
            subject_features.append(trial_windows)
            subject_labels.append(np.full(trial_windows.shape[0], trial_label, dtype=np.int64))
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
