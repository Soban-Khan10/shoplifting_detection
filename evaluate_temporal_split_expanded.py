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
from sklearn.metrics import roc_auc_score, roc_curve

from models.anomaly_net import AnomalyNet


CHECKPOINT_PATH = Path("anomaly_net_temporal_split_expanded_weights.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
PER_VIDEO_CSV_PATH = Path("outputs/temporal_split_expanded_eval_per_video.csv")
TEMPORAL_ROC_PATH = Path("outputs/temporal_split_expanded_eval_roc_curve.png")
HISTOGRAM_PATH = Path("outputs/temporal_split_expanded_eval_score_histogram.png")
VIDEO_ROC_PATH = Path("outputs/temporal_split_expanded_eval_video_level_roc_curve.png")
FEATURE_DIM = 1024
N_SEGMENTS = 32
BASELINE_MIL_TEMPORAL_AUC = 0.544123
BASELINE_MIL_VIDEO_AUC = 0.866213
OPTIMISTIC_TEMPORAL_AUC = 0.851954
OPTIMISTIC_VIDEO_AUC = 0.950113
PREVIOUS_STRICT_SPLIT_TEMPORAL_AUC = 0.665318
PREVIOUS_STRICT_SPLIT_VIDEO_AUC = 0.938776


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


def predict_grouped_scores(grouped_files, model, device):
    grouped_scores = {}
    crop_count = 0

    for base_name, files in grouped_files.items():
        crop_scores = []
        for path in files:
            features = load_features(path).to(device)
            with torch.no_grad():
                scores = model(features).squeeze(0).cpu().numpy()
            crop_scores.append(scores)
            crop_count += 1
        grouped_scores[base_name] = np.mean(np.stack(crop_scores, axis=0), axis=0)

    return grouped_scores, crop_count


def make_strict_labels(intervals):
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


def expand_labels(labels, radius=1):
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
    positives = np.where(labels == 1)[0]
    if len(positives) == 0:
        return N_SEGMENTS
    return int(np.min(np.abs(positives - segment_idx)))


def score_diagnostics(grouped_scores):
    video_max_scores = np.array([np.max(scores) for scores in grouped_scores.values()])
    video_mean_scores = np.array([np.mean(scores) for scores in grouped_scores.values()])
    return {
        "mean_video_max": float(np.mean(video_max_scores)),
        "min_video_max": float(np.min(video_max_scores)),
        "max_video_max": float(np.max(video_max_scores)),
        "mean_video_mean": float(np.mean(video_mean_scores)),
    }


def plot_roc(fpr, tpr, auc, output_path, title):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_score_histogram(anomaly_scores, normal_scores):
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=50, alpha=0.65, label="Normal", density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.65, label="Anomaly", density=True)
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title("Expanded-label temporal split score distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(HISTOGRAM_PATH)
    plt.close()


def write_per_video_csv(rows):
    fieldnames = [
        "base_video",
        "video_type",
        "num_crops",
        "max_score",
        "mean_score",
        "video_label",
        "predicted_alert_using_best_threshold",
        "most_suspicious_segment",
        "suspicious_segment_score",
        "strict_positive_segments",
        "expanded_positive_segments",
        "peak_matches_strict",
        "peak_matches_expanded",
        "distance_to_nearest_strict_positive",
        "distance_to_nearest_expanded_positive",
        "segment_false_positives_strict",
        "segment_false_negatives_strict",
    ]

    with PER_VIDEO_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    eval_anomaly_bases = split["eval_anomaly_bases"]
    eval_normal_bases = split["eval_normal_bases"]

    state_dict, input_dim = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    anomaly_grouped_files = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)
    normal_grouped_files = group_files_by_base(NORMAL_DIR.rglob("*.npy"), eval_normal_bases)
    anomaly_scores, anomaly_crop_count = predict_grouped_scores(anomaly_grouped_files, model, device)
    normal_scores, normal_crop_count = predict_grouped_scores(normal_grouped_files, model, device)

    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    matched_anomaly_names = sorted(name for name in anomaly_scores if name in annotations)
    unmatched_anomaly_names = sorted(name for name in anomaly_scores if name not in annotations)

    y_true_parts = []
    y_score_parts = []
    anomaly_segment_score_parts = []
    normal_segment_score_parts = []
    per_video_rows = []
    strict_peak_matches = 0
    expanded_peak_matches = 0
    strict_distances = []
    expanded_distances = []

    for name in matched_anomaly_names:
        scores = anomaly_scores[name]
        strict_labels = make_strict_labels(annotations[name])
        expanded_labels = expand_labels(strict_labels, radius=1)
        peak = int(np.argmax(scores))
        predicted_segments = scores >= 0.5
        strict_fp = int(np.sum((predicted_segments == 1) & (strict_labels == 0)))
        strict_fn = int(np.sum((predicted_segments == 0) & (strict_labels == 1)))
        strict_match = bool(strict_labels[peak] == 1)
        expanded_match = bool(expanded_labels[peak] == 1)
        strict_distance = distance_to_nearest_positive(peak, strict_labels)
        expanded_distance = distance_to_nearest_positive(peak, expanded_labels)

        strict_peak_matches += int(strict_match)
        expanded_peak_matches += int(expanded_match)
        strict_distances.append(strict_distance)
        expanded_distances.append(expanded_distance)
        y_true_parts.append(strict_labels)
        y_score_parts.append(scores)
        anomaly_segment_score_parts.append(scores)
        per_video_rows.append(
            {
                "base_video": name,
                "video_type": "anomaly",
                "num_crops": len(anomaly_grouped_files[name]),
                "max_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "video_label": 1,
                "predicted_alert_using_best_threshold": "",
                "most_suspicious_segment": peak,
                "suspicious_segment_score": float(scores[peak]),
                "strict_positive_segments": " ".join(str(idx) for idx in positive_indices(strict_labels)),
                "expanded_positive_segments": " ".join(str(idx) for idx in positive_indices(expanded_labels)),
                "peak_matches_strict": strict_match,
                "peak_matches_expanded": expanded_match,
                "distance_to_nearest_strict_positive": strict_distance,
                "distance_to_nearest_expanded_positive": expanded_distance,
                "segment_false_positives_strict": strict_fp,
                "segment_false_negatives_strict": strict_fn,
            }
        )

    for name, scores in normal_scores.items():
        labels = np.zeros(N_SEGMENTS, dtype=np.int64)
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        normal_segment_score_parts.append(scores)
        per_video_rows.append(
            {
                "base_video": name,
                "video_type": "normal",
                "num_crops": len(normal_grouped_files[name]),
                "max_score": float(np.max(scores)),
                "mean_score": float(np.mean(scores)),
                "video_label": 0,
                "predicted_alert_using_best_threshold": "",
                "most_suspicious_segment": int(np.argmax(scores)),
                "suspicious_segment_score": float(np.max(scores)),
                "strict_positive_segments": "",
                "expanded_positive_segments": "",
                "peak_matches_strict": False,
                "peak_matches_expanded": False,
                "distance_to_nearest_strict_positive": N_SEGMENTS,
                "distance_to_nearest_expanded_positive": N_SEGMENTS,
                "segment_false_positives_strict": int(np.sum(scores >= 0.5)),
                "segment_false_negatives_strict": 0,
            }
        )

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    anomaly_segment_scores = np.concatenate(anomaly_segment_score_parts)
    normal_segment_scores = np.concatenate(normal_segment_score_parts)

    temporal_auc = roc_auc_score(y_true, y_score)
    temporal_fpr, temporal_tpr, _ = roc_curve(y_true, y_score)

    video_y_true = np.array(
        [1] * len(anomaly_scores) + [0] * len(normal_scores),
        dtype=np.int64,
    )
    video_y_score = np.array(
        [np.max(scores) for scores in anomaly_scores.values()]
        + [np.max(scores) for scores in normal_scores.values()]
    )
    video_auc = roc_auc_score(video_y_true, video_y_score)
    video_fpr, video_tpr, video_thresholds = roc_curve(video_y_true, video_y_score)
    best_idx = int(np.argmax(video_tpr - video_fpr))
    best_video_threshold = float(video_thresholds[best_idx])

    for row in per_video_rows:
        row["predicted_alert_using_best_threshold"] = bool(row["max_score"] >= best_video_threshold)
        if row["video_type"] == "anomaly":
            scores = anomaly_scores[row["base_video"]]
            strict_labels = make_strict_labels(annotations[row["base_video"]])
            predicted_segments = scores >= best_video_threshold
            row["segment_false_positives_strict"] = int(np.sum((predicted_segments == 1) & (strict_labels == 0)))
            row["segment_false_negatives_strict"] = int(np.sum((predicted_segments == 0) & (strict_labels == 1)))
        else:
            scores = normal_scores[row["base_video"]]
            row["segment_false_positives_strict"] = int(np.sum(scores >= best_video_threshold))

    strict_peak_overlap_ratio = strict_peak_matches / len(matched_anomaly_names)
    expanded_peak_overlap_ratio = expanded_peak_matches / len(matched_anomaly_names)
    avg_strict_distance = float(np.mean(strict_distances))
    avg_expanded_distance = float(np.mean(expanded_distances))
    anomaly_diagnostics = score_diagnostics(anomaly_scores)
    normal_diagnostics = score_diagnostics(normal_scores)

    plot_roc(temporal_fpr, temporal_tpr, temporal_auc, TEMPORAL_ROC_PATH, "Expanded-label temporal split temporal-level ROC")
    plot_score_histogram(anomaly_segment_scores, normal_segment_scores)
    plot_roc(video_fpr, video_tpr, video_auc, VIDEO_ROC_PATH, "Expanded-label temporal split video-level ROC")
    write_per_video_csv(sorted(per_video_rows, key=lambda row: (row["video_type"], row["base_video"])))

    print(f"Device: {device}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Split path: {SPLIT_PATH}")
    print(f"Eval anomaly base video count: {len(eval_anomaly_bases)}")
    print(f"Eval normal base video count: {len(eval_normal_bases)}")
    print(f"Eval anomaly crop file count: {anomaly_crop_count}")
    print(f"Eval normal crop file count: {normal_crop_count}")
    print(f"Matched Shoplifting annotation count: {len(matched_anomaly_names)}")
    print(f"Unmatched anomaly video count: {len(unmatched_anomaly_names)}")
    print(f"Total temporal evaluation points: {len(y_true)}")
    print(f"Baseline MIL temporal AUC: {BASELINE_MIL_TEMPORAL_AUC:.6f}")
    print(f"Baseline MIL video AUC: {BASELINE_MIL_VIDEO_AUC:.6f}")
    print(f"Optimistic temporal fine-tuned temporal AUC: {OPTIMISTIC_TEMPORAL_AUC:.6f}")
    print(f"Optimistic temporal fine-tuned video AUC: {OPTIMISTIC_VIDEO_AUC:.6f}")
    print(f"Previous strict-label honest split temporal AUC: {PREVIOUS_STRICT_SPLIT_TEMPORAL_AUC:.6f}")
    print(f"Previous strict-label honest split video AUC: {PREVIOUS_STRICT_SPLIT_VIDEO_AUC:.6f}")
    print(f"Expanded-label checkpoint honest temporal AUC: {temporal_auc:.6f}")
    print(f"Expanded-label checkpoint honest video AUC: {video_auc:.6f}")
    print(f"Video-level best threshold: {best_video_threshold:.6f}")
    print(f"Strict peak overlap ratio: {strict_peak_overlap_ratio:.6f}")
    print(f"Expanded-label peak overlap ratio: {expanded_peak_overlap_ratio:.6f}")
    print(f"Average distance to strict positive segment: {avg_strict_distance:.6f}")
    print(f"Average distance to expanded positive segment: {avg_expanded_distance:.6f}")
    print("Anomaly eval video score diagnostics:")
    print(f"  Mean video max score: {anomaly_diagnostics['mean_video_max']:.6f}")
    print(f"  Min video max score: {anomaly_diagnostics['min_video_max']:.6f}")
    print(f"  Max video max score: {anomaly_diagnostics['max_video_max']:.6f}")
    print(f"  Mean video mean score: {anomaly_diagnostics['mean_video_mean']:.6f}")
    print("Normal eval video score diagnostics:")
    print(f"  Mean video max score: {normal_diagnostics['mean_video_max']:.6f}")
    print(f"  Min video max score: {normal_diagnostics['min_video_max']:.6f}")
    print(f"  Max video max score: {normal_diagnostics['max_video_max']:.6f}")
    print(f"  Mean video mean score: {normal_diagnostics['mean_video_mean']:.6f}")
    print(f"Saved temporal ROC plot: {TEMPORAL_ROC_PATH}")
    print(f"Saved score histogram plot: {HISTOGRAM_PATH}")
    print(f"Saved video-level ROC plot: {VIDEO_ROC_PATH}")
    print(f"Saved per-video CSV path: {PER_VIDEO_CSV_PATH}")

    conclusions = []
    if temporal_auc > PREVIOUS_STRICT_SPLIT_TEMPORAL_AUC:
        conclusions.append("Temporal AUC improved.")
    else:
        conclusions.append("Temporal AUC decreased.")
    if expanded_peak_overlap_ratio > strict_peak_overlap_ratio:
        conclusions.append("Localization tolerance improved under expanded labels.")
    else:
        conclusions.append("Expanded-label tolerance did not improve peak overlap.")
    if abs(video_auc - PREVIOUS_STRICT_SPLIT_VIDEO_AUC) <= 0.05:
        conclusions.append("Video-level detection stayed strong.")
    else:
        conclusions.append("Video-level detection changed meaningfully.")
    print(f"Comparison conclusion: {' '.join(conclusions)}")


if __name__ == "__main__":
    main()
