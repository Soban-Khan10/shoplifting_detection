import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.anomaly_net import AnomalyNet


INITIAL_CHECKPOINT_PATH = Path("anomaly_net_weights.pth")
OUTPUT_CHECKPOINT_PATH = Path("anomaly_net_temporal_split_expanded_weights.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
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


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


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


def make_segment_labels(intervals):
    labels = np.zeros(N_SEGMENTS, dtype=np.float32)
    if not intervals:
        return labels

    max_annotation_end = max(end for _, end in intervals)
    for start, end in intervals:
        projected_start = start / max_annotation_end * N_SEGMENTS
        projected_end = end / max_annotation_end * N_SEGMENTS
        start_idx = max(0, min(int(np.floor(projected_start)), N_SEGMENTS))
        end_idx = max(0, min(int(np.ceil(projected_end)), N_SEGMENTS))
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1.0

    return labels


def expand_labels(labels, radius):
    if radius <= 0:
        return labels.copy()

    expanded = labels.copy()
    positive_indices = np.where(labels == 1)[0]

    for idx in positive_indices:
        start = max(0, idx - radius)
        end = min(N_SEGMENTS, idx + radius + 1)
        expanded[start:end] = 1.0

    return expanded


def group_files_by_base(files):
    grouped = {}
    for path in sorted(files):
        grouped.setdefault(base_video_name(path), []).append(path)
    return grouped


class ExpandedTemporalSplitDataset(Dataset):
    def __init__(self, anomaly_grouped, normal_grouped, annotations, train_anomaly_bases, train_normal_bases, label_expansion):
        self.samples = []
        self.anomaly_count = 0
        self.normal_count = 0
        before_labels = []
        after_labels = []

        for base_name in train_anomaly_bases:
            if base_name not in annotations:
                raise ValueError(f"Missing Shoplifting annotation for anomaly base video: {base_name}")

            base_labels = make_segment_labels(annotations[base_name])
            expanded_labels = expand_labels(base_labels, label_expansion)

            for path in anomaly_grouped[base_name]:
                self.samples.append((path, expanded_labels))
                before_labels.append(base_labels)
                after_labels.append(expanded_labels)
                self.anomaly_count += 1

        for base_name in train_normal_bases:
            labels = np.zeros(N_SEGMENTS, dtype=np.float32)
            for path in normal_grouped[base_name]:
                self.samples.append((path, labels))
                before_labels.append(labels)
                after_labels.append(labels)
                self.normal_count += 1

        if not self.samples:
            raise ValueError("No expanded temporal split fine-tuning samples found.")

        self.positive_segment_ratio_before = float(np.mean(np.stack(before_labels, axis=0)))
        self.positive_segment_ratio_after = float(np.mean(np.stack(after_labels, axis=0)))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal split model with expanded positive labels.")
    parser.add_argument("--label-expansion", type=int, default=1)
    return parser.parse_args()


def train():
    args = parse_args()
    if args.label_expansion < 0:
        raise ValueError("--label-expansion must be >= 0")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    train_anomaly_bases = split["train_anomaly_bases"]
    train_normal_bases = split["train_normal_bases"]
    seed = split.get("seed")
    train_ratio = split.get("train_ratio")

    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    anomaly_grouped = group_files_by_base(ANOMALY_DIR.rglob("*.npy"))
    normal_grouped = group_files_by_base(NORMAL_DIR.rglob("*.npy"))

    dataset = ExpandedTemporalSplitDataset(
        anomaly_grouped,
        normal_grouped,
        annotations,
        train_anomaly_bases,
        train_normal_bases,
        args.label_expansion,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = AnomalyNet(input_dim=INPUT_DIM).to(device)
    load_initial_weights(model, INITIAL_CHECKPOINT_PATH, device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"Device: {device}")
    print(f"Split path: {SPLIT_PATH}")
    print(f"Label expansion: {args.label_expansion}")
    print(f"Train anomaly base video count: {len(train_anomaly_bases)}")
    print(f"Train normal base video count: {len(train_normal_bases)}")
    print(f"Train anomaly crop sample count: {dataset.anomaly_count}")
    print(f"Train normal crop sample count: {dataset.normal_count}")
    print(f"Total train samples: {len(dataset)}")
    print(f"Positive segment ratio before expansion: {dataset.positive_segment_ratio_before:.6f}")
    print(f"Positive segment ratio after expansion: {dataset.positive_segment_ratio_after:.6f}")

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
            "training_type": "temporal_supervised_split_expanded_finetune",
            "split_file": str(SPLIT_PATH),
            "label_expansion": args.label_expansion,
            "seed": seed,
            "train_ratio": train_ratio,
        },
        OUTPUT_CHECKPOINT_PATH,
    )

    print(f"Saved checkpoint path: {OUTPUT_CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
