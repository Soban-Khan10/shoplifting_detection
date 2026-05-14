import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "alignment_inspection"
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
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
CSV_PATH = Path("outputs/alignment_inspection.csv")
FEATURE_DIM = 1024
N_SEGMENTS = 32


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def safe_plot_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


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

    original_length = feat.shape[0]
    if original_length != N_SEGMENTS:
        idx = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
        feat = feat[idx]

    return torch.FloatTensor(feat).unsqueeze(0), original_length


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


def feature_length_projected_labels(intervals):
    labels = np.zeros(N_SEGMENTS, dtype=np.int64)
    if not intervals:
        return labels

    max_annotation_end = max(end for _, end in intervals)
    for segment_idx in range(N_SEGMENTS):
        segment_start_frame = segment_idx / N_SEGMENTS * max_annotation_end
        segment_end_frame = (segment_idx + 1) / N_SEGMENTS * max_annotation_end
        for start, end in intervals:
            if start < segment_end_frame and end > segment_start_frame:
                labels[segment_idx] = 1
                break

    return labels


def expanded_labels(labels, radius=1):
    expanded = labels.copy()
    positive_indices = np.where(labels == 1)[0]

    for idx in positive_indices:
        start = max(0, idx - radius)
        end = min(N_SEGMENTS, idx + radius + 1)
        expanded[start:end] = 1

    return expanded


def positive_indices(labels):
    return [int(idx) for idx in np.where(labels == 1)[0]]


def distance_to_nearest_positive(segment_idx, labels):
    positives = positive_indices(labels)
    if not positives:
        return None
    return min(abs(segment_idx - idx) for idx in positives)


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

    return np.mean(np.stack(crop_scores, axis=0), axis=0), crop_lengths


def plot_alignment(base_video, scores, current_labels, expanded, suspicious_idx, distance):
    output_path = PLOT_DIR / f"{safe_plot_name(base_video)}.png"
    x = np.arange(N_SEGMENTS)

    plt.figure(figsize=(10, 4))
    for idx, label in enumerate(expanded):
        if label == 1:
            plt.axvspan(idx - 0.5, idx + 0.5, color="gold", alpha=0.15)
    for idx, label in enumerate(current_labels):
        if label == 1:
            plt.axvspan(idx - 0.5, idx + 0.5, color="orange", alpha=0.35)

    plt.plot(x, scores, marker="o", linewidth=2, label="Averaged score")
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
    plt.title(f"{base_video} | distance to positive: {distance}")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def top_segments(scores, k=5):
    indices = np.argsort(scores)[::-1][:k]
    return [int(idx) for idx in indices], [float(scores[idx]) for idx in indices]


def csv_join(values):
    return " ".join(str(value) for value in values)


def inspect_video(base_video, files, intervals, model, device):
    scores, crop_lengths = averaged_scores_for_video(files, model, device)
    current_labels = current_projected_labels(intervals)
    method_b_labels = feature_length_projected_labels(intervals)
    expanded = expanded_labels(current_labels)
    suspicious_idx = int(np.argmax(scores))
    suspicious_score = float(scores[suspicious_idx])
    current_match = bool(current_labels[suspicious_idx] == 1)
    expanded_match = bool(expanded[suspicious_idx] == 1)
    distance = distance_to_nearest_positive(suspicious_idx, current_labels)
    top5_indices, top5_scores = top_segments(scores)
    max_annotation_end = max(end for _, end in intervals)

    plot_alignment(base_video, scores, current_labels, expanded, suspicious_idx, distance)

    print(f"Base video: {base_video}")
    print(f"  Number of crops: {len(files)}")
    print(f"  Crop temporal lengths: {crop_lengths}")
    print(f"  Annotation intervals: {intervals}")
    print(f"  Max annotation end frame: {max_annotation_end}")
    print(f"  Current projected positive segments: {positive_indices(current_labels)}")
    print(f"  Method B positive segments: {positive_indices(method_b_labels)}")
    print(f"  Expanded positive segments: {positive_indices(expanded)}")
    print(f"  Most suspicious segment index: {suspicious_idx}")
    print(f"  Suspicious segment score: {suspicious_score:.6f}")
    print(f"  Matches current labels: {current_match}")
    print(f"  Matches expanded labels: {expanded_match}")
    print(f"  Distance to nearest positive segment: {distance}")
    print(
        "  Top 5 highest scoring segments: "
        + ", ".join(f"{idx}:{score:.6f}" for idx, score in zip(top5_indices, top5_scores))
    )

    return {
        "base_video": base_video,
        "num_crops": len(files),
        "crop_temporal_lengths": csv_join(crop_lengths),
        "annotation_intervals": "; ".join(f"{start}-{end}" for start, end in intervals),
        "max_annotation_end_frame": max_annotation_end,
        "current_positive_segments": csv_join(positive_indices(current_labels)),
        "expanded_positive_segments": csv_join(positive_indices(expanded)),
        "most_suspicious_segment": suspicious_idx,
        "suspicious_segment_score": suspicious_score,
        "matches_current_labels": current_match,
        "matches_expanded_labels": expanded_match,
        "distance_to_nearest_positive": distance,
        "top5_segments": csv_join(top5_indices),
        "top5_scores": csv_join(f"{score:.6f}" for score in top5_scores),
    }


def write_csv(rows, path):
    fieldnames = [
        "base_video",
        "num_crops",
        "crop_temporal_lengths",
        "annotation_intervals",
        "max_annotation_end_frame",
        "current_positive_segments",
        "expanded_positive_segments",
        "most_suspicious_segment",
        "suspicious_segment_score",
        "matches_current_labels",
        "matches_expanded_labels",
        "distance_to_nearest_positive",
        "top5_segments",
        "top5_scores",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    eval_anomaly_bases = split["eval_anomaly_bases"]
    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    grouped_files = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)

    state_dict, input_dim = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    for base_video in sorted(eval_anomaly_bases):
        if base_video not in annotations:
            raise ValueError(f"Missing Shoplifting annotation for eval anomaly video: {base_video}")
        if base_video not in grouped_files:
            raise ValueError(f"Missing anomaly feature files for eval video: {base_video}")
        rows.append(inspect_video(base_video, grouped_files[base_video], annotations[base_video], model, device))

    write_csv(rows, CSV_PATH)

    current_matches = sum(row["matches_current_labels"] for row in rows)
    expanded_matches = sum(row["matches_expanded_labels"] for row in rows)
    distances = [row["distance_to_nearest_positive"] for row in rows if row["distance_to_nearest_positive"] is not None]
    avg_distance = float(np.mean(distances)) if distances else 0.0
    far_videos = [row["base_video"] for row in rows if row["distance_to_nearest_positive"] is not None and row["distance_to_nearest_positive"] > 3]

    print("Final alignment summary")
    print(f"  Device: {device}")
    print(f"  Total eval anomaly videos inspected: {len(rows)}")
    print(f"  Count where suspicious segment matches current labels: {current_matches}")
    print(f"  Count where suspicious segment matches expanded labels: {expanded_matches}")
    print(f"  Average distance to nearest positive segment: {avg_distance:.3f}")
    print(f"  Videos with distance greater than 3 segments: {far_videos}")
    print(f"  CSV path: {CSV_PATH}")
    print(f"  Plot directory: {PLOT_DIR}")

    if expanded_matches > current_matches and expanded_matches >= len(rows) / 2:
        recommendation = "Training with label expansion is likely worth trying."
    elif far_videos:
        recommendation = "Distances are large; revisit annotation scaling and try to recover full video frame counts."
    else:
        recommendation = "Scores are close to labels; try temporal offset diagnostics next."

    early_fire_count = 0
    for row in rows:
        positives = [int(value) for value in row["current_positive_segments"].split()] if row["current_positive_segments"] else []
        if positives and row["most_suspicious_segment"] < min(positives):
            early_fire_count += 1
    if early_fire_count >= len(rows) / 2:
        recommendation += " Scores consistently fire before annotations, so temporal offset diagnostics should be prioritized."

    print(f"  Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
