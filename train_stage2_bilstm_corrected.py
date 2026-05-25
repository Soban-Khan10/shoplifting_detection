import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


FEATURE_DIM = 1024
N_SEGMENTS = 32
FEATURE_FRAME_STRIDE = 16
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
ANOMALY_FEATURE_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_FEATURE_DIRS = [
    Path("data/features/train/normal/Shoplifting_train_normal"),
    Path("data/features/test/normal/Shoplifting_test_normal"),
]
OUTPUT_DIR = Path("outputs/stage2_corrected")
CHECKPOINT_PATH = Path("temporal_model_direct_bilstm_corrected.pth")
SPLIT_PATH = OUTPUT_DIR / "stage2_corrected_split.json"


class DirectBiLSTMLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_DIM,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class Stage2LocalizationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features, original_length = load_and_resample_feature(sample["path"])
        labels = make_32_bin_labels(sample["intervals"], original_length)
        return features, torch.FloatTensor(labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Train corrected Stage 2 Direct BiLSTM localization experiment.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device_name)


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def load_annotations(path):
    annotations = {}
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6 or parts[1] != "Shoplifting":
                continue
            intervals = []
            for idx in range(2, len(parts), 2):
                if idx + 1 >= len(parts):
                    break
                start = int(parts[idx])
                end = int(parts[idx + 1])
                if start >= 0 and end >= 0 and end > start:
                    intervals.append((start, end))
            annotations[Path(parts[0]).stem] = intervals
    return annotations


def load_raw_feature(path):
    feat = np.load(path)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)
    if feat.ndim != 2 or feat.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature shape (T, {FEATURE_DIM}) for {path}, got {feat.shape}")
    return feat.astype(np.float32)


def load_and_resample_feature(path):
    feat = load_raw_feature(path)
    original_length = int(feat.shape[0])
    if original_length != N_SEGMENTS:
        indices = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
        feat = feat[indices]
    return torch.FloatTensor(feat), original_length


def make_32_bin_labels(intervals, feature_length):
    labels = np.zeros(N_SEGMENTS, dtype=np.float32)
    if not intervals:
        return labels
    estimated_total_frames = feature_length * FEATURE_FRAME_STRIDE
    for start, end in intervals:
        start_bin = int(np.floor(start / estimated_total_frames * N_SEGMENTS))
        end_bin = int(np.ceil(end / estimated_total_frames * N_SEGMENTS))
        start_bin = max(0, min(N_SEGMENTS, start_bin))
        end_bin = max(0, min(N_SEGMENTS, end_bin))
        if end_bin > start_bin:
            labels[start_bin:end_bin] = 1.0
    return labels


def collect_grouped_samples():
    annotations = load_annotations(ANNOTATION_PATH)
    anomaly_groups = defaultdict(list)
    for path in sorted(ANOMALY_FEATURE_DIR.glob("*.npy")):
        base = base_video_name(path)
        if base in annotations:
            anomaly_groups[base].append({"path": path, "intervals": annotations[base], "label": 1})

    normal_groups = defaultdict(list)
    for normal_dir in NORMAL_FEATURE_DIRS:
        for path in sorted(normal_dir.glob("*.npy")):
            base = base_video_name(path)
            normal_groups[base].append({"path": path, "intervals": [], "label": 0})

    if not anomaly_groups:
        raise ValueError("No annotated Shoplifting anomaly feature groups found.")
    if not normal_groups:
        raise ValueError("No normal feature groups found.")
    return anomaly_groups, normal_groups


def split_bases(bases, val_fraction, seed):
    bases = sorted(bases)
    rng = random.Random(seed)
    rng.shuffle(bases)
    val_count = max(1, int(round(len(bases) * val_fraction)))
    val_bases = sorted(bases[:val_count])
    train_bases = sorted(bases[val_count:])
    return train_bases, val_bases


def flatten_groups(groups, selected_bases):
    samples = []
    for base in selected_bases:
        samples.extend(groups[base])
    return samples


def compute_pos_weight(samples):
    positives = 0.0
    total = 0.0
    for sample in samples:
        _, original_length = load_and_resample_feature(sample["path"])
        labels = make_32_bin_labels(sample["intervals"], original_length)
        positives += float(np.sum(labels))
        total += float(labels.size)
    negatives = total - positives
    if positives <= 0:
        raise ValueError("Training labels contain zero positive bins.")
    return negatives / positives, positives, negatives


def evaluate_loss_and_peaks(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    peak_bins = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            scores = torch.sigmoid(logits).cpu().numpy()
            peak_bins.extend(np.argmax(scores, axis=1).tolist())
    avg_loss = total_loss / max(batches, 1)
    return avg_loss, peak_bins


def save_split(train_anomaly_bases, val_anomaly_bases, train_normal_bases, val_normal_bases, args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_anomaly_bases": train_anomaly_bases,
        "val_anomaly_bases": val_anomaly_bases,
        "train_normal_bases": train_normal_bases,
        "val_normal_bases": val_normal_bases,
        "feature_frame_stride_assumption": FEATURE_FRAME_STRIDE,
    }
    with SPLIT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    anomaly_groups, normal_groups = collect_grouped_samples()
    train_anomaly_bases, val_anomaly_bases = split_bases(anomaly_groups.keys(), args.val_fraction, args.seed)
    train_normal_bases, val_normal_bases = split_bases(normal_groups.keys(), args.val_fraction, args.seed)
    save_split(train_anomaly_bases, val_anomaly_bases, train_normal_bases, val_normal_bases, args)

    train_samples = flatten_groups(anomaly_groups, train_anomaly_bases) + flatten_groups(normal_groups, train_normal_bases)
    val_samples = flatten_groups(anomaly_groups, val_anomaly_bases) + flatten_groups(normal_groups, val_normal_bases)

    pos_weight_value, positives, negatives = compute_pos_weight(train_samples)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    train_loader = DataLoader(
        Stage2LocalizationDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Stage2LocalizationDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DirectBiLSTMLogits().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = None
    history = []

    print(f"Device: {device}")
    print(f"Train anomaly bases: {len(train_anomaly_bases)} | Val anomaly bases: {len(val_anomaly_bases)}")
    print(f"Train normal bases: {len(train_normal_bases)} | Val normal bases: {len(val_normal_bases)}")
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
    print(f"Positive bins: {positives:.0f} | Negative bins: {negatives:.0f} | pos_weight: {pos_weight_value:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(batches, 1)
        val_loss, val_peak_bins = evaluate_loss_and_peaks(model, val_loader, criterion, device)
        val_peak_bin_31 = sum(1 for peak in val_peak_bins if peak == N_SEGMENTS - 1)
        val_late_bins = sum(1 for peak in val_peak_bins if peak >= 27)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_peak_bin_31": val_peak_bin_31,
                "val_late_bins_27_31": val_late_bins,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "input_dim": FEATURE_DIM,
                    "n_segments": N_SEGMENTS,
                    "pos_weight": pos_weight_value,
                    "feature_frame_stride_assumption": FEATURE_FRAME_STRIDE,
                    "split_path": str(SPLIT_PATH),
                    "train_anomaly_bases": train_anomaly_bases,
                    "val_anomaly_bases": val_anomaly_bases,
                    "train_normal_bases": train_normal_bases,
                    "val_normal_bases": val_normal_bases,
                },
                CHECKPOINT_PATH,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | val_peak31={val_peak_bin_31} | val_late27_31={val_late_bins}"
        )

    history_path = OUTPUT_DIR / "stage2_corrected_training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved corrected checkpoint: {CHECKPOINT_PATH}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Saved split: {SPLIT_PATH}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
