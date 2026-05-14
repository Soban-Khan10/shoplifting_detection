import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.anomaly_net import AnomalyNet


CHECKPOINT_PATH = Path("anomaly_net_weights.pth")
OUTPUT_CHECKPOINT_PATH = Path("anomaly_net_temporal_weights.pth")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
INPUT_DIM = 1024
N_SEGMENTS = 32
EPOCHS = 30
BATCH_SIZE = 16
SMOOTHNESS_WEIGHT = 1e-4


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def load_feature_array(path):
    feat = np.load(path)

    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)

    if feat.ndim != 2:
        raise ValueError(
            f"Expected feature array with shape (T, {INPUT_DIM}) "
            f"or (1, T, {INPUT_DIM}) for {path}, got {feat.shape}"
        )

    if feat.shape[1] != INPUT_DIM:
        raise ValueError(
            f"Expected feature dimension {INPUT_DIM} for {path}, "
            f"got {feat.shape[1]} from shape {feat.shape}"
        )

    return feat


def resample_features(feat):
    if feat.shape[0] != N_SEGMENTS:
        idx = np.linspace(0, feat.shape[0] - 1, N_SEGMENTS, dtype=int)
        feat = feat[idx]
    return feat


def parse_shoplifting_annotations(path):
    annotations = {}

    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue

            video_name, class_name, start1, end1, start2, end2 = parts
            if class_name != "Shoplifting":
                continue

            intervals = []
            for start, end in ((int(start1), int(end1)), (int(start2), int(end2))):
                if start != -1 and end != -1:
                    intervals.append((start, end))

            annotations[Path(video_name).stem] = intervals

    return annotations


def make_segment_labels(intervals, temporal_length):
    labels = np.zeros(N_SEGMENTS, dtype=np.float32)
    if not intervals:
        return labels

    max_end = max(end for _, end in intervals)
    scale = temporal_length / max_end if max_end > temporal_length else 1.0
    segment_edges = np.linspace(0, temporal_length, N_SEGMENTS + 1)

    for start, end in intervals:
        projected_start = max(0.0, min(float(start) * scale, float(temporal_length)))
        projected_end = max(0.0, min(float(end) * scale, float(temporal_length)))
        if projected_end <= projected_start:
            continue

        for segment_idx in range(N_SEGMENTS):
            segment_start = segment_edges[segment_idx]
            segment_end = segment_edges[segment_idx + 1]
            if projected_start < segment_end and projected_end > segment_start:
                labels[segment_idx] = 1.0

    return labels


class TemporalSupervisedDataset(Dataset):
    def __init__(self, anomaly_dir, normal_dir, annotation_path):
        annotations = parse_shoplifting_annotations(annotation_path)
        self.samples = []
        self.anomaly_count = 0
        self.normal_count = 0

        for path in sorted(anomaly_dir.rglob("*.npy")):
            base_name = base_video_name(path)
            if base_name not in annotations:
                print(f"Skipping anomaly crop without annotation: {path}")
                continue

            temporal_length = load_feature_array(path).shape[0]
            labels = make_segment_labels(annotations[base_name], temporal_length)
            self.samples.append((path, labels))
            self.anomaly_count += 1

        for path in sorted(normal_dir.rglob("*.npy")):
            labels = np.zeros(N_SEGMENTS, dtype=np.float32)
            self.samples.append((path, labels))
            self.normal_count += 1

        if not self.samples:
            raise ValueError("No temporal fine-tuning samples found.")

        all_labels = np.stack([labels for _, labels in self.samples], axis=0)
        self.positive_segment_ratio = float(np.mean(all_labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        feat = resample_features(load_feature_array(path))
        return torch.FloatTensor(feat), torch.FloatTensor(labels)


def load_initial_weights(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        print(f"Initial checkpoint not found: {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"Loaded initial weights from: {checkpoint_path}")


def smoothness_loss(scores):
    return torch.mean((scores[:, :-1] - scores[:, 1:]) ** 2)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = TemporalSupervisedDataset(ANOMALY_DIR, NORMAL_DIR, ANNOTATION_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = AnomalyNet(input_dim=INPUT_DIM).to(device)
    load_initial_weights(model, CHECKPOINT_PATH, device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"Device: {device}")
    print(f"Anomaly crop samples: {dataset.anomaly_count}")
    print(f"Normal crop samples: {dataset.normal_count}")
    print(f"Total samples: {len(dataset)}")
    print(f"Positive segment ratio: {dataset.positive_segment_ratio:.6f}")

    final_loss = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(features)
            bce = criterion(scores, labels)
            smooth = smoothness_loss(scores)
            loss = bce + SMOOTHNESS_WEIGHT * smooth
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        final_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d}/{EPOCHS}  |  Loss: {final_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": INPUT_DIM,
            "n_segments": N_SEGMENTS,
            "epochs": EPOCHS,
            "final_loss": final_loss,
            "training_type": "temporal_supervised_finetune",
        },
        OUTPUT_CHECKPOINT_PATH,
    )
    print(f"Model saved: {OUTPUT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
