import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "temporal_split_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.anomaly_net import AnomalyNet


CHECKPOINT_PATH = Path("anomaly_net_temporal_split_weights.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
CSV_PATH = Path("outputs/temporal_split_diagnostics.csv")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
FEATURE_DIM = 1024
N_SEGMENTS = 32
THRESHOLD = 0.339390


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def safe_plot_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


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


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


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

    original_length = feat.shape[0]
    if original_length != N_SEGMENTS:
        idx = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
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


def group_files_by_base(files, allowed_bases):
    grouped = defaultdict(list)
    allowed_bases = set(allowed_bases)

    for path in sorted(files):
        base_name = base_video_name(path)
        if base_name in allowed_bases:
            grouped[base_name].append(path)

    return dict(grouped)


def make_segment_labels(intervals, temporal_length):
    labels = np.zeros(N_SEGMENTS, dtype=np.int64)
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
                labels[segment_idx] = 1

    return labels


def averaged_scores_for_video(files, model, device):
    crop_scores = []
    crop_lengths = []

    for path in files:
        features, original_length = load_features(path)
        features = features.to(device)

        with torch.no_grad():
            scores = model(features).squeeze(0).cpu().numpy()

        crop_scores.append(scores)
        crop_lengths.append(original_length)

    return np.mean(np.stack(crop_scores, axis=0), axis=0), max(crop_lengths)


def segment_indices(labels):
    return [int(idx) for idx in np.where(labels == 1)[0]]


def make_plot(base_video, video_type, scores, labels, threshold, suspicious_idx):
    output_path = PLOT_DIR / f"{safe_plot_name(base_video)}.png"
    x = np.arange(N_SEGMENTS)

    plt.figure(figsize=(10, 4))
    for idx, label in enumerate(labels):
        if label == 1:
            plt.axvspan(idx - 0.5, idx + 0.5, color="orange", alpha=0.2)

    plt.plot(x, scores, marker="o", linewidth=2, label="Averaged score")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    plt.scatter(
        [suspicious_idx],
        [scores[suspicious_idx]],
        color="black",
        s=80,
        zorder=3,
        label="Most suspicious",
    )
    plt.xlabel("Segment index")
    plt.ylabel("Anomaly score")
    plt.title(f"{base_video} ({video_type})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def analyze_video(base_video, video_type, files, model, device, annotations):
    scores, temporal_length = averaged_scores_for_video(files, model, device)
    video_label = 1 if video_type == "anomaly" else 0
    labels = (
        make_segment_labels(annotations[base_video], temporal_length)
        if video_label == 1
        else np.zeros(N_SEGMENTS, dtype=np.int64)
    )

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    predicted_alert = bool(max_score >= THRESHOLD)
    video_correct = predicted_alert == bool(video_label)
    suspicious_idx = int(np.argmax(scores))
    suspicious_score = float(scores[suspicious_idx])
    true_positive_segments = segment_indices(labels)
    suspicious_matches = bool(labels[suspicious_idx] == 1) if video_label == 1 else False
    predicted_segments = scores >= THRESHOLD
    false_positives = int(np.sum((predicted_segments == 1) & (labels == 0)))
    false_negatives = int(np.sum((predicted_segments == 0) & (labels == 1)))

    make_plot(base_video, video_type, scores, labels, THRESHOLD, suspicious_idx)

    return {
        "base_video": base_video,
        "video_type": video_type,
        "num_crops": len(files),
        "max_score": max_score,
        "mean_score": mean_score,
        "threshold": THRESHOLD,
        "predicted_alert": predicted_alert,
        "video_label": video_label,
        "video_correct": video_correct,
        "most_suspicious_segment": suspicious_idx,
        "suspicious_segment_score": suspicious_score,
        "true_positive_segments": " ".join(str(idx) for idx in true_positive_segments),
        "suspicious_segment_matches_annotation": suspicious_matches,
        "segment_false_positives": false_positives,
        "segment_false_negatives": false_negatives,
    }


def write_csv(rows, path):
    fieldnames = [
        "base_video",
        "video_type",
        "num_crops",
        "max_score",
        "mean_score",
        "threshold",
        "predicted_alert",
        "video_label",
        "video_correct",
        "most_suspicious_segment",
        "suspicious_segment_score",
        "true_positive_segments",
        "suspicious_segment_matches_annotation",
        "segment_false_positives",
        "segment_false_negatives",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_rows(title, rows, columns):
    print(title)
    if not rows:
        print("  None")
        return

    for row in rows:
        details = ", ".join(f"{col}={row[col]}" for col in columns)
        print(f"  {row['base_video']}: {details}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    eval_anomaly_bases = split["eval_anomaly_bases"]
    eval_normal_bases = split["eval_normal_bases"]

    state_dict, input_dim = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    anomaly_grouped = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)
    normal_grouped = group_files_by_base(NORMAL_DIR.rglob("*.npy"), eval_normal_bases)

    rows = []
    for base_video in sorted(eval_anomaly_bases):
        if base_video not in annotations:
            raise ValueError(f"Missing Shoplifting annotation for eval anomaly video: {base_video}")
        rows.append(analyze_video(base_video, "anomaly", anomaly_grouped[base_video], model, device, annotations))

    for base_video in sorted(eval_normal_bases):
        rows.append(analyze_video(base_video, "normal", normal_grouped[base_video], model, device, annotations))

    write_csv(rows, CSV_PATH)

    anomaly_rows = [row for row in rows if row["video_type"] == "anomaly"]
    normal_rows = [row for row in rows if row["video_type"] == "normal"]
    worst_anomaly_rows = sorted(
        anomaly_rows,
        key=lambda row: (row["segment_false_negatives"], -row["max_score"]),
        reverse=True,
    )[:5]
    worst_normal_rows = sorted(normal_rows, key=lambda row: row["max_score"], reverse=True)[:5]
    missed_suspicious_rows = [
        row for row in anomaly_rows if not row["suspicious_segment_matches_annotation"]
    ]

    print(f"Device: {device}")
    print(f"Number of eval anomaly videos: {len(eval_anomaly_bases)}")
    print(f"Number of eval normal videos: {len(eval_normal_bases)}")
    print(f"CSV path: {CSV_PATH}")
    print(f"Plot directory: {PLOT_DIR}")
    print_rows(
        "Worst anomaly videos by segment false negatives:",
        worst_anomaly_rows,
        ["segment_false_negatives", "max_score", "most_suspicious_segment"],
    )
    print_rows(
        "Worst normal videos by max score:",
        worst_normal_rows,
        ["max_score", "predicted_alert", "segment_false_positives"],
    )
    print_rows(
        "Anomaly videos where suspicious segment did not overlap annotation:",
        missed_suspicious_rows,
        ["most_suspicious_segment", "suspicious_segment_score", "true_positive_segments"],
    )


if __name__ == "__main__":
    main()
