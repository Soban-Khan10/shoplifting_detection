import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs/stage2_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from train_stage2_bilstm_corrected import (
    ANOMALY_FEATURE_DIR,
    ANNOTATION_PATH,
    CHECKPOINT_PATH,
    FEATURE_DIM,
    FEATURE_FRAME_STRIDE,
    N_SEGMENTS,
    NORMAL_FEATURE_DIRS,
    DirectBiLSTMLogits,
    load_raw_feature,
    make_32_bin_labels,
)


SUMMARY_CSV = OUTPUT_DIR / "stage2_corrected_summary.csv"
PER_VIDEO_CSV = OUTPUT_DIR / "stage2_corrected_per_video.csv"
PEAK_HISTOGRAM = OUTPUT_DIR / "stage2_corrected_peak_histogram.png"
SCORE_PLOT_DIR = OUTPUT_DIR / "stage2_corrected_score_plots"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate corrected Stage 2 Direct BiLSTM experiment.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
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


def load_checkpoint(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Corrected checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


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


def collect_groups():
    annotations = load_annotations(ANNOTATION_PATH)
    anomaly_groups = defaultdict(list)
    for path in sorted(ANOMALY_FEATURE_DIR.glob("*.npy")):
        base = base_video_name(path)
        if base in annotations:
            anomaly_groups[base].append(path)

    normal_groups = defaultdict(list)
    for normal_dir in NORMAL_FEATURE_DIRS:
        for path in sorted(normal_dir.glob("*.npy")):
            normal_groups[base_video_name(path)].append(path)
    return annotations, anomaly_groups, normal_groups


def load_and_resample(path):
    feat = load_raw_feature(path)
    original_length = int(feat.shape[0])
    if original_length != N_SEGMENTS:
        indices = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
        feat = feat[indices]
    return torch.FloatTensor(feat), original_length


def predict_file(model, path, device):
    features, original_length = load_and_resample(path)
    features = features.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(features).squeeze(0)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores, original_length


def predict_group(model, paths, device):
    scores = []
    lengths = []
    for path in paths:
        score, original_length = predict_file(model, path, device)
        scores.append(score)
        lengths.append(original_length)
    return np.mean(np.stack(scores, axis=0), axis=0), int(max(lengths))


def gt_bin_ranges(intervals, feature_length):
    labels = make_32_bin_labels(intervals, feature_length)
    ranges = []
    start = None
    for idx, value in enumerate(labels.tolist() + [0.0]):
        if value == 1.0 and start is None:
            start = idx
        elif value == 0.0 and start is not None:
            ranges.append((start, idx - 1))
            start = None
    return labels.astype(np.int64), ranges


def peak_alignment(peak_bin, ranges):
    if not ranges:
        return False, None
    if any(start <= peak_bin <= end for start, end in ranges):
        return True, 0
    distances = []
    for start, end in ranges:
        if peak_bin < start:
            distances.append(start - peak_bin)
        else:
            distances.append(peak_bin - end)
    return False, int(min(distances))


def format_ranges(ranges):
    return ";".join(f"{start}-{end}" for start, end in ranges)


def plot_peak_histogram(peak_bins):
    plt.figure(figsize=(8, 4))
    plt.hist(peak_bins, bins=np.arange(N_SEGMENTS + 1) - 0.5, edgecolor="black")
    plt.xlabel("Predicted peak bin")
    plt.ylabel("Video count")
    plt.title("Corrected Stage 2 peak-bin histogram")
    plt.xticks(range(0, N_SEGMENTS, 2))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PEAK_HISTOGRAM)
    plt.close()


def plot_scores(video_name, scores, labels, output_path):
    x = np.arange(N_SEGMENTS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(x, scores, marker="o", linewidth=1.8, label="Corrected Stage 2 score")
    plt.fill_between(x, 0, labels, color="green", alpha=0.2, step="mid", label="GT label")
    plt.scatter([int(np.argmax(scores))], [float(np.max(scores))], color="black", s=70, zorder=3, label="Peak")
    plt.xlabel("32-bin segment index")
    plt.ylabel("Score")
    plt.title(video_name)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_per_video(rows):
    fieldnames = [
        "video_name",
        "label",
        "num_feature_files",
        "feature_length_used",
        "gt_bin_ranges",
        "video_score",
        "predicted_peak_bin",
        "predicted_peak_score",
        "peak_overlaps_gt",
        "distance_bins",
        "peak_at_bin_31",
        "peak_in_late_bins_27_31",
    ]
    with PER_VIDEO_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary):
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main():
    args = parse_args()
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = DirectBiLSTMLogits().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    annotations, anomaly_groups, normal_groups = collect_groups()
    rows = []
    temporal_true = []
    temporal_scores = []
    video_true = []
    video_scores = []
    anomaly_peak_overlaps = []
    anomaly_peak_bins = []

    for video_name, paths in sorted(anomaly_groups.items()):
        scores, feature_length = predict_group(model, paths, device)
        labels, ranges = gt_bin_ranges(annotations[video_name], feature_length)
        peak_bin = int(np.argmax(scores))
        peak_score = float(scores[peak_bin])
        overlap, distance = peak_alignment(peak_bin, ranges)
        anomaly_peak_overlaps.append(overlap)
        anomaly_peak_bins.append(peak_bin)
        temporal_true.extend(labels.tolist())
        temporal_scores.extend(scores.tolist())
        video_true.append(1)
        video_scores.append(float(np.max(scores)))
        plot_scores(video_name, scores, labels, SCORE_PLOT_DIR / f"{video_name}_corrected_scores.png")
        rows.append(
            {
                "video_name": video_name,
                "label": 1,
                "num_feature_files": len(paths),
                "feature_length_used": feature_length,
                "gt_bin_ranges": format_ranges(ranges),
                "video_score": float(np.max(scores)),
                "predicted_peak_bin": peak_bin,
                "predicted_peak_score": peak_score,
                "peak_overlaps_gt": overlap,
                "distance_bins": distance,
                "peak_at_bin_31": peak_bin == N_SEGMENTS - 1,
                "peak_in_late_bins_27_31": peak_bin >= 27,
            }
        )

    for video_name, paths in sorted(normal_groups.items()):
        scores, feature_length = predict_group(model, paths, device)
        labels = np.zeros(N_SEGMENTS, dtype=np.int64)
        peak_bin = int(np.argmax(scores))
        temporal_true.extend(labels.tolist())
        temporal_scores.extend(scores.tolist())
        video_true.append(0)
        video_scores.append(float(np.max(scores)))
        rows.append(
            {
                "video_name": video_name,
                "label": 0,
                "num_feature_files": len(paths),
                "feature_length_used": feature_length,
                "gt_bin_ranges": "",
                "video_score": float(np.max(scores)),
                "predicted_peak_bin": peak_bin,
                "predicted_peak_score": float(scores[peak_bin]),
                "peak_overlaps_gt": "",
                "distance_bins": "",
                "peak_at_bin_31": peak_bin == N_SEGMENTS - 1,
                "peak_in_late_bins_27_31": peak_bin >= 27,
            }
        )

    temporal_auc = float(roc_auc_score(np.array(temporal_true), np.array(temporal_scores)))
    video_auc = float(roc_auc_score(np.array(video_true), np.array(video_scores)))
    peak_overlap_ratio = float(np.mean(anomaly_peak_overlaps)) if anomaly_peak_overlaps else 0.0
    peak_at_31_count = int(sum(1 for peak in anomaly_peak_bins if peak == N_SEGMENTS - 1))
    late_bin_count = int(sum(1 for peak in anomaly_peak_bins if peak >= 27))
    shoplifting001 = next((row for row in rows if row["video_name"] == "Shoplifting001_x264"), None)

    plot_peak_histogram(anomaly_peak_bins)
    write_per_video(rows)
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "temporal_auc": temporal_auc,
        "video_auc": video_auc,
        "anomaly_video_count": len(anomaly_groups),
        "normal_video_count": len(normal_groups),
        "peak_overlap_ratio": peak_overlap_ratio,
        "peak_overlap_count": int(sum(anomaly_peak_overlaps)),
        "peak_at_bin_31_count": peak_at_31_count,
        "late_bin_27_31_count": late_bin_count,
        "shoplifting001_peak_bin": shoplifting001["predicted_peak_bin"] if shoplifting001 else "",
        "shoplifting001_peak_overlaps_gt": shoplifting001["peak_overlaps_gt"] if shoplifting001 else "",
        "per_video_csv": str(PER_VIDEO_CSV),
        "peak_histogram": str(PEAK_HISTOGRAM),
        "score_plot_dir": str(SCORE_PLOT_DIR),
    }
    write_summary(summary)

    print(f"temporal_auc: {temporal_auc:.6f}")
    print(f"video_auc: {video_auc:.6f}")
    print(f"peak_overlap_ratio: {peak_overlap_ratio:.6f}")
    print(f"peak_overlap_count: {int(sum(anomaly_peak_overlaps))}/{len(anomaly_peak_overlaps)}")
    print(f"peak_at_bin_31_count: {peak_at_31_count}")
    print(f"late_bin_27_31_count: {late_bin_count}")
    if shoplifting001:
        print(f"shoplifting001_peak_bin: {shoplifting001['predicted_peak_bin']}")
        print(f"shoplifting001_peak_overlaps_gt: {shoplifting001['peak_overlaps_gt']}")
    print(f"saved_summary_csv: {SUMMARY_CSV}")
    print(f"saved_per_video_csv: {PER_VIDEO_CSV}")
    print(f"saved_peak_histogram: {PEAK_HISTOGRAM}")
    print(f"saved_score_plot_dir: {SCORE_PLOT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
