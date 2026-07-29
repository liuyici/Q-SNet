from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset


SESSIONS = {
    1: [
        "1_20131027", "2_20140404", "3_20140603", "4_20140621", "5_20140411",
        "6_20130712", "7_20131027", "8_20140511", "9_20140620", "10_20131130",
        "11_20140618", "12_20131127", "13_20140527", "14_20140601", "15_20130709",
    ],
    2: [
        "1_20131030", "2_20140413", "3_20140611", "4_20140702", "5_20140418",
        "6_20131016", "7_20131030", "8_20140514", "9_20140627", "10_20131204",
        "11_20140625", "12_20131201", "13_20140603", "14_20140615", "15_20131016",
    ],
    3: [
        "1_20131107", "2_20140419", "3_20140629", "4_20140705", "5_20140506",
        "6_20131113", "7_20131106", "8_20140521", "9_20140704", "10_20131211",
        "11_20140630", "12_20131207", "13_20140610", "14_20140627", "15_20131105",
    ],
}


def windows(features, size):
    return np.stack([features[index:index + size] for index in range(features.shape[0] - size + 1)])


def load_subjects(file_path, session, feature):
    root = Path(file_path)
    names = SESSIONS[session]
    trial_labels = scipy.io.loadmat(root / "label.mat", mat_dtype=True)["label"][0] + 1
    subjects = []
    labels = []
    for name in names:
        data = scipy.io.loadmat(root / f"{name}.mat", mat_dtype=True)
        subject_features = []
        subject_labels = []
        for trial_index in range(15):
            trial = np.swapaxes(data[f"{feature}{trial_index + 1}"], 0, 1)
            trial = trial[-185:]
            trial_windows = windows(trial, 12)
            subject_features.append(trial_windows)
            subject_labels.append(np.full(trial_windows.shape[0], trial_labels[trial_index], dtype=np.int64))
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
