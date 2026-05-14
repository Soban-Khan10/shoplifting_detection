import csv
import json
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
from sklearn.metrics import roc_auc_score

from models.anomaly_net import AnomalyNet


CHECKPOINT_PATH = Path("anomaly_net_temporal_split_weights.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
CSV_PATH = Path("outputs/temporal_offset_diagnostics.csv")
AUC_PLOT_PATH = Path("outputs/temporal_offset_diagnostics.png")
OVERLAP_PLOT_PATH = Path("outputs/temporal_offset_overlap.png")
FEATURE_DIM = 1024
N_SEGMENTS = 32
OFFSETS = range(-16, 17)


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim", FEATURE_DIM)
        state_dict = checkpoint["model_state_dict"]
    else:
        input_dim = FEATURE_DIM
        state_dict = checkpoint

    return state_dict, input_dim


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


def group_files_by_base(files, allowed_bases):
    grouped = defaultdict(list)
    allowed_bases = set(allowed_bases)

    for path in sorted(files):
        base_name = base_video_name(path)
        if base_name in allowed_bases:
            grouped[base_name].append(path)

    return dict(grouped)


def load_features(path):
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

    if feat.shape[0] != N_SEGMENTS:
        idx = np.linspace(0, feat.shape[0] - 1, N_SEGMENTS, dtype=int)
        feat = feat[idx]

    return torch.FloatTensor(feat).unsqueeze(0)


def current_projected_labels(intervals):
    labels = np.zeros(N_SEGMENTS, dtype=np.int64)
    if not intervals:
        return labels

    max_annotation_end = max(end for _, end in intervals)
    for start, end in intervals:
        projected_start = start / max_annotation_end * N_SEGMENTS
        projected_end = end / max_annotation_end * N_SEGMENTS
        start_idx = max(0, min(int(np.floor(projected_start)), N_SEGMENTS))
        end_idx = max(0, min(int(np.ceil(projected_end)), N_SEGMENTS))
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1

    return labels


def shift_labels(labels, offset):
    shifted = np.zeros_like(labels)
    positive_indices = np.where(labels == 1)[0]

    for idx in positive_indices:
        shifted_idx = idx + offset
        if 0 <= shifted_idx < len(labels):
            shifted[shifted_idx] = 1

    return shifted


def distance_to_nearest_positive(segment_idx, labels):
    positive_indices = np.where(labels == 1)[0]
    if len(positive_indices) == 0:
        return N_SEGMENTS
    return int(np.min(np.abs(positive_indices - segment_idx)))


def averaged_scores(grouped_files, model, device):
    grouped_scores = {}

    for base_name, files in grouped_files.items():
        crop_scores = []
        for path in files:
            features = load_features(path).to(device)
            with torch.no_grad():
                scores = model(features).squeeze(0).cpu().numpy()
            crop_scores.append(scores)
        grouped_scores[base_name] = np.mean(np.stack(crop_scores, axis=0), axis=0)

    return grouped_scores


def evaluate_offset(offset, anomaly_scores, normal_scores, anomaly_labels):
    y_true_parts = []
    y_score_parts = []
    overlap_count = 0
    distances = []

    for base_name, scores in anomaly_scores.items():
        shifted_labels = shift_labels(anomaly_labels[base_name], offset)
        suspicious_idx = int(np.argmax(scores))
        if shifted_labels[suspicious_idx] == 1:
            overlap_count += 1
        distances.append(distance_to_nearest_positive(suspicious_idx, shifted_labels))
        y_true_parts.append(shifted_labels)
        y_score_parts.append(scores)

    for scores in normal_scores.values():
        y_true_parts.append(np.zeros(N_SEGMENTS, dtype=np.int64))
        y_score_parts.append(scores)

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    temporal_auc = roc_auc_score(y_true, y_score)
    overlap_ratio = overlap_count / len(anomaly_scores) if anomaly_scores else 0.0
    average_distance = float(np.mean(distances)) if distances else 0.0

    return {
        "offset": offset,
        "temporal_auc": float(temporal_auc),
        "anomaly_peak_overlap_count": overlap_count,
        "anomaly_peak_overlap_ratio": float(overlap_ratio),
        "average_distance_to_positive": average_distance,
    }


def write_csv(rows, path):
    fieldnames = [
        "offset",
        "temporal_auc",
        "anomaly_peak_overlap_count",
        "anomaly_peak_overlap_ratio",
        "average_distance_to_positive",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_auc(rows, path, best_offset):
    offsets = [row["offset"] for row in rows]
    aucs = [row["temporal_auc"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(offsets, aucs, marker="o", linewidth=2)
    plt.axvline(best_offset, color="red", linestyle="--", label=f"Best offset: {best_offset}")
    plt.xlabel("Label offset in segments")
    plt.ylabel("Temporal ROC AUC")
    plt.title("Temporal AUC by annotation label offset")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_overlap(rows, path, best_offset):
    offsets = [row["offset"] for row in rows]
    overlaps = [row["anomaly_peak_overlap_ratio"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(offsets, overlaps, marker="o", linewidth=2)
    plt.axvline(best_offset, color="red", linestyle="--", label=f"Best offset: {best_offset}")
    plt.xlabel("Label offset in segments")
    plt.ylabel("Peak overlap ratio")
    plt.title("Anomaly peak overlap by annotation label offset")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def recommendation(best_auc_offset, auc_at_zero, best_auc):
    improvement = best_auc - auc_at_zero
    parts = []

    if best_auc_offset <= -4:
        parts.append("Best offset is strongly negative, so annotations appear projected too late.")
    elif abs(best_auc_offset) <= 2:
        parts.append("Best offset is near zero, so offset alone is not the main issue.")
    else:
        parts.append("Best offset is not near zero, so a label-offset experiment is worth testing.")

    if improvement >= 0.05:
        parts.append("AUC improves substantially, so try corrected annotation projection or an explicit label offset experiment.")
    else:
        parts.append("AUC improvement is modest, so prioritize projection/frame-count verification before retraining.")

    return " ".join(parts)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    eval_anomaly_bases = split["eval_anomaly_bases"]
    eval_normal_bases = split["eval_normal_bases"]
    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)

    state_dict, input_dim = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    anomaly_grouped = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)
    normal_grouped = group_files_by_base(NORMAL_DIR.rglob("*.npy"), eval_normal_bases)
    anomaly_scores = averaged_scores(anomaly_grouped, model, device)
    normal_scores = averaged_scores(normal_grouped, model, device)
    anomaly_labels = {
        base_name: current_projected_labels(annotations[base_name])
        for base_name in anomaly_scores
    }

    rows = [
        evaluate_offset(offset, anomaly_scores, normal_scores, anomaly_labels)
        for offset in OFFSETS
    ]
    write_csv(rows, CSV_PATH)

    best_auc_row = max(rows, key=lambda row: row["temporal_auc"])
    best_overlap_row = max(rows, key=lambda row: row["anomaly_peak_overlap_ratio"])
    zero_row = next(row for row in rows if row["offset"] == 0)

    plot_auc(rows, AUC_PLOT_PATH, best_auc_row["offset"])
    plot_overlap(rows, OVERLAP_PLOT_PATH, best_overlap_row["offset"])

    print(f"Device: {device}")
    print(f"Number of eval anomaly videos: {len(eval_anomaly_bases)}")
    print(f"Number of eval normal videos: {len(eval_normal_bases)}")
    print(f"Best offset by temporal AUC: {best_auc_row['offset']}")
    print(f"Best temporal AUC: {best_auc_row['temporal_auc']:.6f}")
    print(f"Best offset by overlap ratio: {best_overlap_row['offset']}")
    print(f"Best overlap ratio: {best_overlap_row['anomaly_peak_overlap_ratio']:.6f}")
    print(f"Average distance at offset 0: {zero_row['average_distance_to_positive']:.6f}")
    print(f"Average distance at best offset: {best_auc_row['average_distance_to_positive']:.6f}")
    print(f"CSV path: {CSV_PATH}")
    print(f"AUC plot path: {AUC_PLOT_PATH}")
    print(f"Overlap plot path: {OVERLAP_PLOT_PATH}")
    print(f"Recommendation: {recommendation(best_auc_row['offset'], zero_row['temporal_auc'], best_auc_row['temporal_auc'])}")


if __name__ == "__main__":
    main()
