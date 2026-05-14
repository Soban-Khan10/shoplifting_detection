import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from models.anomaly_net import AnomalyNet


SPLIT_PATH = Path("outputs/temporal_split.json")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
OLD_CHECKPOINT_PATH = Path("anomaly_net_weights.pth")
FEATURE_DIM = 1024
N_SEGMENTS = 32
SMOOTHNESS_WEIGHT = 1e-4

CHECKPOINT_PATHS = {
    "two_stage_tcn_refiner": Path("temporal_model_two_stage_tcn_refiner.pth"),
    "direct_tcn": Path("temporal_model_direct_tcn.pth"),
    "direct_bilstm": Path("temporal_model_direct_bilstm.pth"),
    "small_transformer": Path("temporal_model_small_transformer.pth"),
}
SUMMARY_CSV_PATH = Path("outputs/temporal_model_comparison_summary.csv")
PER_VIDEO_CSV_PATH = Path("outputs/temporal_model_comparison_per_video.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


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


def group_files_by_base(files):
    grouped = defaultdict(list)
    for path in sorted(files):
        grouped[base_video_name(path)].append(path)
    return dict(grouped)


def load_feature_array(path):
    feat = np.load(path)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)
    if feat.ndim != 2:
        raise ValueError(f"Expected (T, {FEATURE_DIM}) or (1, T, {FEATURE_DIM}) for {path}, got {feat.shape}")
    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM} for {path}, got {feat.shape[1]}")
    if feat.shape[0] != N_SEGMENTS:
        idx = np.linspace(0, feat.shape[0] - 1, N_SEGMENTS, dtype=int)
        feat = feat[idx]
    return feat.astype(np.float32)


def make_labels(intervals):
    labels = np.zeros(N_SEGMENTS, dtype=np.float32)
    if not intervals:
        return labels
    max_end = max(end for _, end in intervals)
    for start, end in intervals:
        start_idx = max(0, min(int(np.floor(start / max_end * N_SEGMENTS)), N_SEGMENTS))
        end_idx = max(0, min(int(np.ceil(end / max_end * N_SEGMENTS)), N_SEGMENTS))
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1.0
    return labels


def expand_labels(labels, radius):
    if radius <= 0:
        return labels.copy()
    expanded = labels.copy()
    for idx in np.where(labels == 1)[0]:
        expanded[max(0, idx - radius):min(N_SEGMENTS, idx + radius + 1)] = 1.0
    return expanded


class TemporalCropDataset(Dataset):
    def __init__(self, anomaly_grouped, normal_grouped, annotations, anomaly_bases, normal_bases, label_expansion):
        self.samples = []
        self.anomaly_count = 0
        self.normal_count = 0
        for base in anomaly_bases:
            labels = expand_labels(make_labels(annotations[base]), label_expansion)
            for path in anomaly_grouped[base]:
                self.samples.append((path, labels))
                self.anomaly_count += 1
        for base in normal_bases:
            labels = np.zeros(N_SEGMENTS, dtype=np.float32)
            for path in normal_grouped[base]:
                self.samples.append((path, labels))
                self.normal_count += 1
        self.positive_segment_ratio = float(np.mean(np.stack([labels for _, labels in self.samples], axis=0)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        return torch.FloatTensor(load_feature_array(path)), torch.FloatTensor(labels)


class TCNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.net(x).squeeze(1)


class TwoStageTCNRefiner(nn.Module):
    def __init__(self, old_model):
        super().__init__()
        self.old_model = old_model
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.refiner = TCNModel(FEATURE_DIM + 1)

    def forward(self, x):
        with torch.no_grad():
            old_scores = self.old_model(x).unsqueeze(-1)
        return self.refiner(torch.cat([x, old_scores], dim=-1))


class BiLSTMModel(nn.Module):
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
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(FEATURE_DIM, 128)
        self.pos = nn.Parameter(torch.zeros(1, N_SEGMENTS, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x).squeeze(-1)


def load_old_anomaly_net(device):
    model = AnomalyNet(input_dim=FEATURE_DIM).to(device)
    try:
        checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_model(model_name, device):
    if model_name == "two_stage_tcn_refiner":
        return TwoStageTCNRefiner(load_old_anomaly_net(device)).to(device)
    if model_name == "direct_tcn":
        return TCNModel(FEATURE_DIM).to(device)
    if model_name == "direct_bilstm":
        return BiLSTMModel().to(device)
    if model_name == "small_transformer":
        return TransformerModel().to(device)
    raise ValueError(f"Unknown model: {model_name}")


def smoothness_loss(scores):
    return torch.mean((scores[:, :-1] - scores[:, 1:]) ** 2)


def train_one_model(model_name, model, loader, device, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
    final_loss = None
    for epoch in range(1, epochs + 1):
        model.train()
        if model_name == "two_stage_tcn_refiner":
            model.old_model.eval()
        total_loss = 0.0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            scores = model(features)
            loss = criterion(scores, labels) + SMOOTHNESS_WEIGHT * smoothness_loss(scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(loader)
        print(f"  Epoch {epoch:02d}/{epochs}  |  Loss: {final_loss:.6f}")
    return optimizer, final_loss


def positive_indices(labels):
    return [int(idx) for idx in np.where(labels == 1)[0]]


def distance_to_positive(peak, labels):
    positives = np.where(labels == 1)[0]
    if len(positives) == 0:
        return N_SEGMENTS
    return int(np.min(np.abs(positives - peak)))


def top_segments(scores, k=5):
    indices = np.argsort(scores)[::-1][:k]
    return [int(idx) for idx in indices], [float(scores[idx]) for idx in indices]


def evaluate_model(model_name, model, anomaly_grouped, normal_grouped, annotations, eval_anomaly_bases, eval_normal_bases, device):
    model.eval()
    grouped_scores = {}
    per_video_rows = []
    y_true_parts = []
    y_score_parts = []
    anomaly_max_scores = []
    anomaly_mean_scores = []
    normal_max_scores = []
    normal_mean_scores = []
    peak_matches = 0
    distances = []

    for base in sorted(eval_anomaly_bases):
        crop_scores = []
        for path in anomaly_grouped[base]:
            features = torch.FloatTensor(load_feature_array(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                crop_scores.append(model(features).squeeze(0).cpu().numpy())
        scores = np.mean(np.stack(crop_scores, axis=0), axis=0)
        labels = make_labels(annotations[base]).astype(np.int64)
        peak = int(np.argmax(scores))
        peak_match = bool(labels[peak] == 1)
        distance = distance_to_positive(peak, labels)
        top5_idx, top5_scores = top_segments(scores)
        grouped_scores[base] = scores
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        anomaly_max_scores.append(float(np.max(scores)))
        anomaly_mean_scores.append(float(np.mean(scores)))
        peak_matches += int(peak_match)
        distances.append(distance)
        per_video_rows.append({
            "model_name": model_name,
            "base_video": base,
            "video_type": "anomaly",
            "num_crops": len(anomaly_grouped[base]),
            "max_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "video_label": 1,
            "most_suspicious_segment": peak,
            "suspicious_segment_score": float(scores[peak]),
            "positive_segments": " ".join(str(x) for x in positive_indices(labels)),
            "peak_matches_label": peak_match,
            "distance_to_nearest_positive": distance,
            "top5_segments": " ".join(str(x) for x in top5_idx),
            "top5_scores": " ".join(f"{x:.6f}" for x in top5_scores),
        })

    for base in sorted(eval_normal_bases):
        crop_scores = []
        for path in normal_grouped[base]:
            features = torch.FloatTensor(load_feature_array(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                crop_scores.append(model(features).squeeze(0).cpu().numpy())
        scores = np.mean(np.stack(crop_scores, axis=0), axis=0)
        labels = np.zeros(N_SEGMENTS, dtype=np.int64)
        peak = int(np.argmax(scores))
        top5_idx, top5_scores = top_segments(scores)
        grouped_scores[base] = scores
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        normal_max_scores.append(float(np.max(scores)))
        normal_mean_scores.append(float(np.mean(scores)))
        per_video_rows.append({
            "model_name": model_name,
            "base_video": base,
            "video_type": "normal",
            "num_crops": len(normal_grouped[base]),
            "max_score": float(np.max(scores)),
            "mean_score": float(np.mean(scores)),
            "video_label": 0,
            "most_suspicious_segment": peak,
            "suspicious_segment_score": float(scores[peak]),
            "positive_segments": "",
            "peak_matches_label": False,
            "distance_to_nearest_positive": N_SEGMENTS,
            "top5_segments": " ".join(str(x) for x in top5_idx),
            "top5_scores": " ".join(f"{x:.6f}" for x in top5_scores),
        })

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    temporal_auc = float(roc_auc_score(y_true, y_score))
    video_y_true = np.array([1] * len(eval_anomaly_bases) + [0] * len(eval_normal_bases), dtype=np.int64)
    video_y_score = np.array(anomaly_max_scores + normal_max_scores)
    video_auc = float(roc_auc_score(video_y_true, video_y_score))
    fpr, tpr, thresholds = roc_curve(video_y_true, video_y_score)
    best_idx = int(np.argmax(tpr - fpr))
    video_best_threshold = float(thresholds[best_idx])
    return {
        "model_name": model_name,
        "temporal_auc": temporal_auc,
        "video_auc": video_auc,
        "video_best_threshold": video_best_threshold,
        "strict_peak_overlap_ratio": peak_matches / len(eval_anomaly_bases),
        "average_distance_to_positive": float(np.mean(distances)),
        "anomaly_mean_max_score": float(np.mean(anomaly_max_scores)),
        "anomaly_min_max_score": float(np.min(anomaly_max_scores)),
        "anomaly_max_max_score": float(np.max(anomaly_max_scores)),
        "anomaly_mean_mean_score": float(np.mean(anomaly_mean_scores)),
        "normal_mean_max_score": float(np.mean(normal_max_scores)),
        "normal_min_max_score": float(np.min(normal_max_scores)),
        "normal_max_max_score": float(np.max(normal_max_scores)),
        "normal_mean_mean_score": float(np.mean(normal_mean_scores)),
    }, per_video_rows


def save_checkpoint(model_name, model, optimizer, final_loss, epochs, label_expansion, seed):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_name": model_name,
        "input_dim": FEATURE_DIM + 1 if model_name == "two_stage_tcn_refiner" else FEATURE_DIM,
        "n_segments": N_SEGMENTS,
        "epochs": epochs,
        "final_loss": final_loss,
        "training_type": "temporal_model_comparison",
        "split_file": str(SPLIT_PATH),
        "label_expansion": label_expansion,
        "seed": seed,
    }, CHECKPOINT_PATHS[model_name])


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_bar(rows, key, output_path, ylabel):
    names = [row["model_name"] for row in rows]
    values = [row[key] for row in rows]
    plt.figure(figsize=(9, 5))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_ranked(title, rows, key, reverse=True):
    print(title)
    for row in sorted(rows, key=lambda x: x[key], reverse=reverse):
        print(
            f"  {row['model_name']}: temporal_auc={row['temporal_auc']:.6f}, "
            f"video_auc={row['video_auc']:.6f}, peak_overlap={row['strict_peak_overlap_ratio']:.6f}, "
            f"avg_distance={row['average_distance_to_positive']:.6f}"
        )


def recommendation(rows):
    best_auc = max(rows, key=lambda x: x["temporal_auc"])
    best_overlap = max(rows, key=lambda x: x["strict_peak_overlap_ratio"])
    message = f"Choose {best_auc['model_name']} based primarily on temporal AUC."
    if best_overlap["model_name"] != best_auc["model_name"]:
        message += f" Note: {best_overlap['model_name']} has better peak overlap."
    if best_auc["model_name"] == "two_stage_tcn_refiner":
        message += " Use it as the next main inference model."
    elif best_auc["model_name"] == "direct_tcn":
        message += " Replace AnomalyNet with direct TCN for the next iteration."
    else:
        message += " Use it cautiously and check overfitting."
    return message


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare temporal sequence models on the honest split.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-expansion", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    anomaly_grouped = group_files_by_base(ANOMALY_DIR.rglob("*.npy"))
    normal_grouped = group_files_by_base(NORMAL_DIR.rglob("*.npy"))
    train_anomaly_bases = split["train_anomaly_bases"]
    train_normal_bases = split["train_normal_bases"]
    eval_anomaly_bases = split["eval_anomaly_bases"]
    eval_normal_bases = split["eval_normal_bases"]
    dataset = TemporalCropDataset(
        anomaly_grouped,
        normal_grouped,
        annotations,
        train_anomaly_bases,
        train_normal_bases,
        args.label_expansion,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_anomaly_crop_count = sum(len(anomaly_grouped[base]) for base in eval_anomaly_bases)
    eval_normal_crop_count = sum(len(normal_grouped[base]) for base in eval_normal_bases)

    print(f"Device: {device}")
    print(f"Split path: {SPLIT_PATH}")
    print(f"Train anomaly base count: {len(train_anomaly_bases)}")
    print(f"Train normal base count: {len(train_normal_bases)}")
    print(f"Eval anomaly base count: {len(eval_anomaly_bases)}")
    print(f"Eval normal base count: {len(eval_normal_bases)}")
    print(f"Train anomaly crop count: {dataset.anomaly_count}")
    print(f"Train normal crop count: {dataset.normal_count}")
    print(f"Eval anomaly crop count: {eval_anomaly_crop_count}")
    print(f"Eval normal crop count: {eval_normal_crop_count}")
    print(f"Positive segment ratio in train set: {dataset.positive_segment_ratio:.6f}")

    model_names = ["two_stage_tcn_refiner", "direct_tcn", "direct_bilstm", "small_transformer"]
    summary_rows = []
    per_video_rows = []
    for model_name in model_names:
        print(f"Model: {model_name}")
        model = build_model(model_name, device)
        optimizer, final_loss = train_one_model(model_name, model, loader, device, args.epochs, args.lr)
        metrics, rows = evaluate_model(
            model_name,
            model,
            anomaly_grouped,
            normal_grouped,
            annotations,
            eval_anomaly_bases,
            eval_normal_bases,
            device,
        )
        metrics["final_train_loss"] = final_loss
        summary_rows.append(metrics)
        per_video_rows.extend(rows)
        save_checkpoint(model_name, model, optimizer, final_loss, args.epochs, args.label_expansion, args.seed)
        print(f"  Final train loss: {final_loss:.6f}")
        print(f"  Temporal AUC: {metrics['temporal_auc']:.6f}")
        print(f"  Video AUC: {metrics['video_auc']:.6f}")
        print(f"  Video best threshold: {metrics['video_best_threshold']:.6f}")
        print(f"  Strict peak overlap ratio: {metrics['strict_peak_overlap_ratio']:.6f}")
        print(f"  Average distance to positive segment: {metrics['average_distance_to_positive']:.6f}")

    summary_fields = [
        "model_name", "temporal_auc", "video_auc", "video_best_threshold",
        "strict_peak_overlap_ratio", "average_distance_to_positive", "final_train_loss",
        "anomaly_mean_max_score", "normal_mean_max_score",
    ]
    per_video_fields = [
        "model_name", "base_video", "video_type", "num_crops", "max_score", "mean_score",
        "video_label", "most_suspicious_segment", "suspicious_segment_score", "positive_segments",
        "peak_matches_label", "distance_to_nearest_positive", "top5_segments", "top5_scores",
    ]
    write_csv(SUMMARY_CSV_PATH, summary_rows, summary_fields)
    write_csv(PER_VIDEO_CSV_PATH, per_video_rows, per_video_fields)
    plot_bar(summary_rows, "temporal_auc", OUTPUT_DIR / "temporal_model_comparison_temporal_auc.png", "Temporal AUC")
    plot_bar(summary_rows, "video_auc", OUTPUT_DIR / "temporal_model_comparison_video_auc.png", "Video AUC")
    plot_bar(summary_rows, "strict_peak_overlap_ratio", OUTPUT_DIR / "temporal_model_comparison_peak_overlap.png", "Strict peak overlap ratio")
    plot_bar(summary_rows, "average_distance_to_positive", OUTPUT_DIR / "temporal_model_comparison_distance.png", "Average distance to positive")

    print_ranked("Final ranked table sorted by temporal AUC descending:", summary_rows, "temporal_auc")
    print_ranked("Final ranked table sorted by strict peak overlap ratio descending:", summary_rows, "strict_peak_overlap_ratio")
    print(f"Recommendation: {recommendation(summary_rows)}")


if __name__ == "__main__":
    main()
