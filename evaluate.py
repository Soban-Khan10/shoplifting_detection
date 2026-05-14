import os
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
from sklearn.metrics import roc_auc_score, roc_curve

from models.anomaly_net import AnomalyNet


CHECKPOINT_PATH = "anomaly_net_weights.pth"
ANOMALY_DIR = Path("data/features/test/anomaly")
NORMAL_DIR = Path("data/features/test/normal")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
FEATURE_DIM = 1024
DEFAULT_N_SEGMENTS = 32


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim", FEATURE_DIM)
        n_segments = checkpoint.get("n_segments", DEFAULT_N_SEGMENTS)
        state_dict = checkpoint["model_state_dict"]
    else:
        input_dim = FEATURE_DIM
        n_segments = DEFAULT_N_SEGMENTS
        state_dict = checkpoint

    return state_dict, input_dim, n_segments


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def load_features(path, n_segments):
    feat = np.load(path)

    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)

    if feat.ndim != 2:
        raise ValueError(
            f"Expected feature array with shape (T, {FEATURE_DIM}) "
            f"or (1, T, {FEATURE_DIM}) for {path}, got {feat.shape}"
        )

    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM} for {path}, "
            f"got {feat.shape[1]} from shape {feat.shape}"
        )

    original_length = feat.shape[0]
    if original_length != n_segments:
        idx = np.linspace(0, original_length - 1, n_segments, dtype=int)
        feat = feat[idx]

    return torch.FloatTensor(feat).unsqueeze(0), original_length


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


def predict_grouped_scores(files, model, device, n_segments):
    grouped_scores = defaultdict(list)
    grouped_lengths = defaultdict(list)

    for path in files:
        features, original_length = load_features(path, n_segments)
        features = features.to(device)

        with torch.no_grad():
            scores = model(features).squeeze(0).cpu().numpy()

        base_name = base_video_name(path)
        grouped_scores[base_name].append(scores)
        grouped_lengths[base_name].append(original_length)

    averaged = {}
    lengths = {}
    for base_name, scores in grouped_scores.items():
        averaged[base_name] = np.mean(np.stack(scores, axis=0), axis=0)
        lengths[base_name] = max(grouped_lengths[base_name])

    return averaged, lengths


def expand_scores(scores, temporal_length):
    if temporal_length == len(scores):
        return scores

    source_x = np.linspace(0, temporal_length - 1, len(scores))
    target_x = np.arange(temporal_length)
    return np.interp(target_x, source_x, scores)


def make_anomaly_labels(intervals, temporal_length):
    labels = np.zeros(temporal_length, dtype=np.int64)
    if not intervals:
        return labels

    max_end = max(end for _, end in intervals)
    scale = temporal_length / max_end if max_end > temporal_length else 1.0

    for start, end in intervals:
        start_idx = int(np.floor(start * scale))
        end_idx = int(np.ceil(end * scale))
        start_idx = max(0, min(start_idx, temporal_length))
        end_idx = max(0, min(end_idx, temporal_length))
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1

    return labels


def plot_roc(fpr, tpr, auc):
    output_path = OUTPUT_DIR / "roc_curve.png"
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state_dict, input_dim, n_segments = load_checkpoint(CHECKPOINT_PATH, device)

    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    anomaly_files = sorted(ANOMALY_DIR.rglob("*.npy"))
    normal_files = sorted(NORMAL_DIR.rglob("*.npy"))

    anomaly_scores, anomaly_lengths = predict_grouped_scores(anomaly_files, model, device, n_segments)
    normal_scores, normal_lengths = predict_grouped_scores(normal_files, model, device, n_segments)

    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    matched_anomaly_names = sorted(name for name in anomaly_scores if name in annotations)
    unmatched_anomaly_names = sorted(name for name in anomaly_scores if name not in annotations)

    y_true_parts = []
    y_score_parts = []

    for name in matched_anomaly_names:
        temporal_length = anomaly_lengths[name]
        y_true_parts.append(make_anomaly_labels(annotations[name], temporal_length))
        y_score_parts.append(expand_scores(anomaly_scores[name], temporal_length))

    for name, scores in normal_scores.items():
        temporal_length = normal_lengths[name]
        y_true_parts.append(np.zeros(temporal_length, dtype=np.int64))
        y_score_parts.append(expand_scores(scores, temporal_length))

    if not y_true_parts:
        raise ValueError("No evaluation data found.")

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    plot_roc(fpr, tpr, auc)

    print(f"Device: {device}")
    print(f"Anomaly feature file count: {len(anomaly_files)}")
    print(f"Normal feature file count: {len(normal_files)}")
    print(f"Grouped anomaly video count: {len(anomaly_scores)}")
    print(f"Grouped normal video count: {len(normal_scores)}")
    print(f"Matched Shoplifting annotation count: {len(matched_anomaly_names)}")
    print(f"Unmatched anomaly video count: {len(unmatched_anomaly_names)}")
    print(f"Unmatched anomaly video names: {unmatched_anomaly_names[:10]}")
    print(f"Total evaluation points: {len(y_true)}")
    print(f"Frame/temporal-level AUC: {auc:.6f}")
    print(f"Best threshold: {best_threshold:.6f}")


if __name__ == "__main__":
    main()
