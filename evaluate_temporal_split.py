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


CHECKPOINT_PATH = Path("anomaly_net_temporal_split_weights.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
FEATURE_DIM = 1024
DEFAULT_N_SEGMENTS = 32
BASELINE_MIL_TEMPORAL_AUC = 0.544123
BASELINE_MIL_VIDEO_AUC = 0.866213
OPTIMISTIC_TEMPORAL_AUC = 0.851954
OPTIMISTIC_VIDEO_AUC = 0.950113


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


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


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


def group_files_by_base(files, allowed_bases):
    grouped = defaultdict(list)
    allowed_bases = set(allowed_bases)

    for path in sorted(files):
        base_name = base_video_name(path)
        if base_name in allowed_bases:
            grouped[base_name].append(path)

    return dict(grouped)


def predict_grouped_scores(grouped_files, model, device, n_segments):
    averaged_scores = {}
    max_lengths = {}
    crop_count = 0

    for base_name, files in grouped_files.items():
        crop_scores = []
        crop_lengths = []

        for path in files:
            features, original_length = load_features(path, n_segments)
            features = features.to(device)

            with torch.no_grad():
                scores = model(features).squeeze(0).cpu().numpy()

            crop_scores.append(scores)
            crop_lengths.append(original_length)
            crop_count += 1

        averaged_scores[base_name] = np.mean(np.stack(crop_scores, axis=0), axis=0)
        max_lengths[base_name] = max(crop_lengths)

    return averaged_scores, max_lengths, crop_count


def make_segment_labels(intervals, temporal_length, n_segments):
    labels = np.zeros(n_segments, dtype=np.int64)
    if not intervals:
        return labels

    max_end = max(end for _, end in intervals)
    scale = temporal_length / max_end if max_end > temporal_length else 1.0
    segment_edges = np.linspace(0, temporal_length, n_segments + 1)

    for start, end in intervals:
        projected_start = max(0.0, min(float(start) * scale, float(temporal_length)))
        projected_end = max(0.0, min(float(end) * scale, float(temporal_length)))
        if projected_end <= projected_start:
            continue

        for segment_idx in range(n_segments):
            segment_start = segment_edges[segment_idx]
            segment_end = segment_edges[segment_idx + 1]
            if projected_start < segment_end and projected_end > segment_start:
                labels[segment_idx] = 1

    return labels


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


def plot_score_histogram(anomaly_scores, normal_scores, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=50, alpha=0.65, label="Normal", density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.65, label="Anomaly", density=True)
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title("Honest temporal split score distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = load_split(SPLIT_PATH)
    eval_anomaly_bases = split["eval_anomaly_bases"]
    eval_normal_bases = split["eval_normal_bases"]

    state_dict, input_dim, n_segments = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    anomaly_grouped_files = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)
    normal_grouped_files = group_files_by_base(NORMAL_DIR.rglob("*.npy"), eval_normal_bases)

    anomaly_scores, anomaly_lengths, anomaly_crop_count = predict_grouped_scores(
        anomaly_grouped_files,
        model,
        device,
        n_segments,
    )
    normal_scores, normal_lengths, normal_crop_count = predict_grouped_scores(
        normal_grouped_files,
        model,
        device,
        n_segments,
    )

    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)
    matched_anomaly_names = sorted(name for name in anomaly_scores if name in annotations)
    unmatched_anomaly_names = sorted(name for name in anomaly_scores if name not in annotations)

    y_true_parts = []
    y_score_parts = []
    anomaly_score_parts = []
    normal_score_parts = []

    for name in matched_anomaly_names:
        labels = make_segment_labels(annotations[name], anomaly_lengths[name], n_segments)
        scores = anomaly_scores[name]
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        anomaly_score_parts.append(scores)

    for name, scores in normal_scores.items():
        labels = np.zeros(n_segments, dtype=np.int64)
        y_true_parts.append(labels)
        y_score_parts.append(scores)
        normal_score_parts.append(scores)

    if not y_true_parts:
        raise ValueError("No held-out split evaluation data found.")

    y_true = np.concatenate(y_true_parts)
    y_score = np.concatenate(y_score_parts)
    anomaly_segment_scores = np.concatenate(anomaly_score_parts)
    normal_segment_scores = np.concatenate(normal_score_parts)

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

    anomaly_diagnostics = score_diagnostics(anomaly_scores)
    normal_diagnostics = score_diagnostics(normal_scores)

    temporal_roc_path = OUTPUT_DIR / "temporal_split_eval_roc_curve.png"
    histogram_path = OUTPUT_DIR / "temporal_split_eval_score_histogram.png"
    video_roc_path = OUTPUT_DIR / "temporal_split_eval_video_level_roc_curve.png"

    plot_roc(
        temporal_fpr,
        temporal_tpr,
        temporal_auc,
        temporal_roc_path,
        "Honest temporal split temporal-level ROC curve",
    )
    plot_score_histogram(anomaly_segment_scores, normal_segment_scores, histogram_path)
    plot_roc(
        video_fpr,
        video_tpr,
        video_auc,
        video_roc_path,
        "Honest temporal split video-level ROC curve",
    )

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
    print(f"Honest split temporal AUC: {temporal_auc:.6f}")
    print(f"Honest split video AUC: {video_auc:.6f}")
    print(f"Video-level best threshold: {best_video_threshold:.6f}")
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
    print(f"Saved temporal ROC plot: {temporal_roc_path}")
    print(f"Saved score histogram plot: {histogram_path}")
    print(f"Saved video-level ROC plot: {video_roc_path}")


if __name__ == "__main__":
    main()
