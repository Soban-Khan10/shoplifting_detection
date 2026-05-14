
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class UCFCrimeDataset(Dataset):
    def __init__(self, anomaly_dir, normal_dir, n_segments=32):
        self.n_segments = n_segments
        self.feature_dim = 1024
        self.anomaly_files = sorted(Path(anomaly_dir).rglob("*.npy"))
        self.normal_files = sorted(Path(normal_dir).rglob("*.npy"))

        print(f"Anomaly feature files found: {len(self.anomaly_files)}")
        print(f"Normal feature files found:  {len(self.normal_files)}")

        if not self.anomaly_files:
            raise ValueError(f"No anomaly .npy files found under: {anomaly_dir}")
        if not self.normal_files:
            raise ValueError(f"No normal .npy files found under: {normal_dir}")

    def load_features(self, path):
        feat = np.load(path)

        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = np.squeeze(feat, axis=0)

        if feat.ndim != 2:
            raise ValueError(
                f"Expected feature array with shape (T, {self.feature_dim}) "
                f"or (1, T, {self.feature_dim}) for {path}, got {feat.shape}"
            )

        if feat.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim} for {path}, "
                f"got {feat.shape[1]} from shape {feat.shape}"
            )

        if feat.shape[0] != self.n_segments:
            idx = np.linspace(0, feat.shape[0] - 1, self.n_segments, dtype=int)
            feat = feat[idx]

        return torch.FloatTensor(feat)

    def __len__(self):
        return len(self.anomaly_files)

    def __getitem__(self, idx):
        anomaly_feat = self.load_features(self.anomaly_files[idx])
        normal_idx = np.random.randint(len(self.normal_files))
        normal_feat = self.load_features(self.normal_files[normal_idx])
        return anomaly_feat, normal_feat
