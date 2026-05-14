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
SUMMARY_CSV_PATH = Path("outputs/label_mapping_strategy_comparison.csv")
PER_VIDEO_CSV_PATH = Path("outputs/label_mapping_strategy_per_video.csv")
AUC_PLOT_PATH = Path("outputs/label_mapping_strategy_comparison.png")
OVERLAP_PLOT_PATH = Path("outputs/label_mapping_peak_overlap.png")
FEATURE_DIM = 1024
N_SEGMENTS = 32


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


def current_max_end_labels(intervals):
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


def expand_labels(labels, radius):
    expanded = labels.copy()
    positive_indices = np.where(labels == 1)[0]

    for idx in positive_indices:
        start = max(0, idx - radius)
        end = min(N_SEGMENTS, idx + radius + 1)
        expanded[start:end] = 1

    return expanded


def shift_labels(labels, offset):
    shifted = np.zeros_like(labels)
    positive_indices = np.where(labels == 1)[0]

    for idx in positive_indices:
        shifted_idx = idx + offset
        if 0 <= shifted_idx < N_SEGMENTS:
            shifted[shifted_idx] = 1

    return shifted


def centered_peak_labels(scores, radius=1):
    labels = np.zeros(N_SEGMENTS, dtype=np.int64)
    peak = int(np.argmax(scores))
    start = max(0, peak - radius)
    end = min(N_SEGMENTS, peak + radius + 1)
    labels[start:end] = 1
    return labels


def positive_indices(labels):
    return [int(idx) for idx in np.where(labels == 1)[0]]


def distance_to_nearest_positive(segment_idx, labels):
    positives = np.where(labels == 1)[0]
    if len(positives) == 0:
        return N_SEGMENTS
    return int(np.min(np.abs(positives - segment_idx)))


def top_segments(scores, k=5):
    indices = np.argsort(scores)[::-1][:k]
    return [int(idx) for idx in indices], [float(scores[idx]) for idx in indices]


def csv_join(values):
    return " ".join(str(value) for value in values)


def build_strategy_labels(strategy, base_labels, scores):
    if strategy == "current_max_end":
        return base_labels.copy()
    if strategy == "current_plus_expand_1":
        return expand_labels(base_labels, radius=1)
    if strategy == "current_plus_expand_2":
        return expand_labels(base_labels, radius=2)
    if strategy == "offset_plus_8":
        return shift_labels(base_labels, offset=8)
    if strategy == "offset_minus_16":
        return shift_labels(base_labels, offset=-16)
    if strategy == "centered_around_peak_window_3":
        return centered_peak_labels(scores, radius=1)
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_strategy(strategy, anomaly_scores, normal_scores, base_anomaly_labels, video_auc):
    y_true_parts = []
    y_score_parts = []
    per_video_rows = []
    peak_overlap_count = 0
    distances = []
    positive_counts = []

    for base_video, scores in anomaly_scores.items():
        labels = build_strategy_labels(strategy, base_anomaly_labels[base_video], scores)
        peak = int(np.argmax(scores))
        suspicious_score = float(scores[peak])
        matches = bool(labels[peak] == 1)
        distance = distance_to_nearest_positive(peak, labels)
        top5_segments, top5_scores = top_segments(scores)

        if matches:
            peak_overlap_count += 1
        distances.append(distance)
        positive_counts.append(int(np.sum(labels)))
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        per_video_rows.append(
            {
                "strategy": strategy,
                "base_video": base_video,
                "positive_segments": csv_join(positive_indices(labels)),
                "most_suspicious_segment": peak,
                "suspicious_segment_score": suspicious_score,
                "matches_positive_label": matches,
                "distance_to_nearest_positive": distance,
                "top5_segments": csv_join(top5_segments),
                "top5_scores": csv_join(f"{score:.6f}" for score in top5_scores),
            }
        )

    for scores in normal_scores.values():
        y_true_parts.append(np.zeros(N_SEGMENTS, dtype=np.int64))
        y_score_parts.append(scores)

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    temporal_auc = roc_auc_score(y_true, y_score)
    overlap_ratio = peak_overlap_count / len(anomaly_scores) if anomaly_scores else 0.0

    summary = {
        "strategy": strategy,
        "temporal_auc": float(temporal_auc),
        "peak_overlap_count": peak_overlap_count,
        "peak_overlap_ratio": float(overlap_ratio),
        "average_distance_to_positive": float(np.mean(distances)),
        "average_positive_segments_per_anomaly_video": float(np.mean(positive_counts)),
        "video_auc": float(video_auc),
        "notes": "diagnostic/leaky upper-bound, not valid real evaluation"
        if strategy == "centered_around_peak_window_3"
        else "",
    }

    return summary, per_video_rows


def compute_video_auc(anomaly_scores, normal_scores):
    y_true = np.array([1] * len(anomaly_scores) + [0] * len(normal_scores), dtype=np.int64)
    y_score = np.array(
        [np.max(scores) for scores in anomaly_scores.values()]
        + [np.max(scores) for scores in normal_scores.values()]
    )
    return float(roc_auc_score(y_true, y_score))


def write_summary_csv(rows, path):
    fieldnames = [
        "strategy",
        "temporal_auc",
        "peak_overlap_count",
        "peak_overlap_ratio",
        "average_distance_to_positive",
        "average_positive_segments_per_anomaly_video",
        "video_auc",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_video_csv(rows, path):
    fieldnames = [
        "strategy",
        "base_video",
        "positive_segments",
        "most_suspicious_segment",
        "suspicious_segment_score",
        "matches_positive_label",
        "distance_to_nearest_positive",
        "top5_segments",
        "top5_scores",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric(rows, metric, ylabel, output_path):
    strategies = [row["strategy"] for row in rows]
    values = [row[metric] for row in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(strategies, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_summary_table(title, rows, sort_key):
    print(title)
    for row in sorted(rows, key=lambda item: item[sort_key], reverse=True):
        print(
            f"  {row['strategy']}: "
            f"temporal_auc={row['temporal_auc']:.6f}, "
            f"peak_overlap_ratio={row['peak_overlap_ratio']:.6f}, "
            f"avg_distance={row['average_distance_to_positive']:.6f}, "
            f"avg_positive_segments={row['average_positive_segments_per_anomaly_video']:.3f}, "
            f"video_auc={row['video_auc']:.6f}"
        )


def make_recommendation(summary_rows):
    non_leaky = [row for row in summary_rows if row["strategy"] != "centered_around_peak_window_3"]
    current = next(row for row in summary_rows if row["strategy"] == "current_max_end")
    expand_1 = next(row for row in summary_rows if row["strategy"] == "current_plus_expand_1")
    expand_2 = next(row for row in summary_rows if row["strategy"] == "current_plus_expand_2")
    offset_plus = next(row for row in summary_rows if row["strategy"] == "offset_plus_8")
    best_non_leaky_overlap = max(non_leaky, key=lambda row: row["peak_overlap_ratio"])

    if (
        max(expand_1["peak_overlap_ratio"], expand_2["peak_overlap_ratio"])
        > current["peak_overlap_ratio"]
        and max(expand_1["temporal_auc"], expand_2["temporal_auc"]) < 0.95
    ):
        return "Expansion improves peak overlap without an unrealistic AUC; try a label expansion training experiment."

    if (
        offset_plus["temporal_auc"] > current["temporal_auc"]
        and offset_plus["average_distance_to_positive"] > current["average_distance_to_positive"]
    ):
        return "Offset improves AUC but worsens peak distance; do not trust offset alone."

    if best_non_leaky_overlap["peak_overlap_ratio"] < 0.5:
        return "All non-leaky strategies have poor peak overlap; recover actual raw videos/frame counts before further training."

    return "Use the best non-leaky overlap strategy for a small controlled training experiment."


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
    base_anomaly_labels = {
        base_video: current_max_end_labels(annotations[base_video])
        for base_video in anomaly_scores
    }
    video_auc = compute_video_auc(anomaly_scores, normal_scores)

    strategies = [
        "current_max_end",
        "current_plus_expand_1",
        "current_plus_expand_2",
        "offset_plus_8",
        "offset_minus_16",
        "centered_around_peak_window_3",
    ]

    summary_rows = []
    per_video_rows = []
    for strategy in strategies:
        summary, per_video = evaluate_strategy(
            strategy,
            anomaly_scores,
            normal_scores,
            base_anomaly_labels,
            video_auc,
        )
        summary_rows.append(summary)
        per_video_rows.extend(per_video)

    write_summary_csv(summary_rows, SUMMARY_CSV_PATH)
    write_per_video_csv(per_video_rows, PER_VIDEO_CSV_PATH)
    plot_metric(summary_rows, "temporal_auc", "Temporal ROC AUC", AUC_PLOT_PATH)
    plot_metric(summary_rows, "peak_overlap_ratio", "Peak overlap ratio", OVERLAP_PLOT_PATH)

    print(f"Device: {device}")
    print(f"Eval anomaly video count: {len(eval_anomaly_bases)}")
    print(f"Eval normal video count: {len(eval_normal_bases)}")
    print(f"Video-level AUC: {video_auc:.6f}")
    print_summary_table("Summary sorted by temporal_auc descending:", summary_rows, "temporal_auc")
    print_summary_table("Summary sorted by peak_overlap_ratio descending:", summary_rows, "peak_overlap_ratio")
    print("Warning: centered_around_peak_window_3 is diagnostic/leaky and not valid for real evaluation.")
    print(f"Recommendation: {make_recommendation(summary_rows)}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Per-video CSV: {PER_VIDEO_CSV_PATH}")
    print(f"Temporal AUC plot: {AUC_PLOT_PATH}")
    print(f"Peak overlap plot: {OVERLAP_PLOT_PATH}")


if __name__ == "__main__":
    main()
