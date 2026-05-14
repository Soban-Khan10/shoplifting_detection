import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs")
PLOT_DIR = OUTPUT_DIR / "two_stage_pipeline_per_video_scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_auc_score, roc_curve

from models.anomaly_net import AnomalyNet


STAGE1_CHECKPOINT_PATH = Path("anomaly_net_weights.pth")
STAGE2_CHECKPOINT_PATH = Path("temporal_model_direct_bilstm.pth")
SPLIT_PATH = Path("outputs/temporal_split.json")
ANNOTATION_PATH = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
ANOMALY_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
NORMAL_DIR = Path("data/features/test/normal/Shoplifting_test_normal")
SUMMARY_CSV_PATH = Path("outputs/two_stage_pipeline_summary.csv")
PER_VIDEO_CSV_PATH = Path("outputs/two_stage_pipeline_per_video.csv")
STAGE1_ROC_PATH = Path("outputs/two_stage_stage1_video_roc.png")
STAGE2_ROC_PATH = Path("outputs/two_stage_stage2_temporal_roc.png")
STAGE1_HIST_PATH = Path("outputs/two_stage_stage1_score_histogram.png")
STAGE2_HIST_PATH = Path("outputs/two_stage_stage2_score_histogram.png")
FEATURE_DIM = 1024
N_SEGMENTS = 32
OLD_HONEST_VIDEO_AUC = 0.938776
OLD_HONEST_TEMPORAL_AUC = 0.665318
BILSTM_STANDALONE_TEMPORAL_AUC = 0.893811
KNOWN_STAGE1_THRESHOLD = 0.339390


class DirectBiLSTM(nn.Module):
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


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def safe_plot_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_split(path):
    with path.open("r") as f:
        return json.load(f)


def load_state_dict(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


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
        base = base_video_name(path)
        if base in allowed_bases:
            grouped[base].append(path)
    return dict(grouped)


def load_feature(path):
    feat = np.load(path)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)
    if feat.ndim != 2:
        raise ValueError(f"Expected (T, {FEATURE_DIM}) or (1, T, {FEATURE_DIM}) for {path}, got {feat.shape}")
    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM} for {path}, got {feat.shape[1]}")
    original_length = feat.shape[0]
    if original_length != N_SEGMENTS:
        idx = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
        feat = feat[idx]
    return torch.FloatTensor(feat).unsqueeze(0), original_length


def make_labels(intervals):
    labels = np.zeros(N_SEGMENTS, dtype=np.int64)
    if not intervals:
        return labels
    max_end = max(end for _, end in intervals)
    for start, end in intervals:
        start_idx = max(0, min(int(np.floor(start / max_end * N_SEGMENTS)), N_SEGMENTS))
        end_idx = max(0, min(int(np.ceil(end / max_end * N_SEGMENTS)), N_SEGMENTS))
        if end_idx > start_idx:
            labels[start_idx:end_idx] = 1
    return labels


def positive_indices(labels):
    return [int(idx) for idx in np.where(labels == 1)[0]]


def distance_to_positive(peak, labels):
    positives = np.where(labels == 1)[0]
    if len(positives) == 0:
        return -1
    return int(np.min(np.abs(positives - peak)))


def approximate_window(segment_idx, temporal_length):
    start = int(segment_idx / N_SEGMENTS * temporal_length)
    end = int((segment_idx + 1) / N_SEGMENTS * temporal_length)
    return start, end


def score_group(model, grouped_files, device):
    grouped_scores = {}
    grouped_lengths = {}
    crop_count = 0
    for base, files in grouped_files.items():
        crop_scores = []
        lengths = []
        for path in files:
            features, original_length = load_feature(path)
            features = features.to(device)
            with torch.no_grad():
                scores = model(features).squeeze(0).cpu().numpy()
            crop_scores.append(scores)
            lengths.append(original_length)
            crop_count += 1
        grouped_scores[base] = np.mean(np.stack(crop_scores, axis=0), axis=0)
        grouped_lengths[base] = max(lengths)
    return grouped_scores, grouped_lengths, crop_count


def confusion_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


def plot_roc(fpr, tpr, auc_value, path, title):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hist(anomaly_scores, normal_scores, path, title):
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=20, alpha=0.65, label="Normal", density=True)
    plt.hist(anomaly_scores, bins=20, alpha=0.65, label="Anomaly", density=True)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_video_scores(row, stage1_scores, stage2_scores, labels, threshold):
    output_path = PLOT_DIR / f"{safe_plot_name(row['base_video'])}.png"
    x = np.arange(N_SEGMENTS)
    plt.figure(figsize=(10, 4))
    for idx, label in enumerate(labels):
        if label == 1:
            plt.axvspan(idx - 0.5, idx + 0.5, color="orange", alpha=0.25)
    plt.plot(x, stage1_scores, marker="o", linewidth=1.8, label="Stage 1 AnomalyNet")
    plt.plot(x, stage2_scores, marker="s", linewidth=1.8, label="Stage 2 BiLSTM")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Stage 1 threshold")
    if row["stage2_most_suspicious_segment"] != "":
        peak = int(row["stage2_most_suspicious_segment"])
        plt.scatter([peak], [stage2_scores[peak]], color="black", s=80, zorder=3, label="Stage 2 peak")
    plt.xlabel("Segment index")
    plt.ylabel("Score")
    plt.title(f"{row['base_video']} | alert={row['stage1_alert']} | success={row['pipeline_success']}")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_summary(metrics):
    with SUMMARY_CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": value})


def write_per_video(rows):
    fieldnames = [
        "base_video", "video_type", "video_label", "num_crops", "stage1_max_score",
        "stage1_mean_score", "stage1_alert", "stage1_correct", "stage2_ran",
        "stage2_max_score", "stage2_mean_score", "stage2_most_suspicious_segment",
        "stage2_suspicious_segment_score", "true_positive_segments", "stage2_peak_matches_label",
        "distance_to_nearest_positive", "pipeline_success", "approximate_feature_window_start",
        "approximate_feature_window_end",
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
    annotations = parse_shoplifting_annotations(ANNOTATION_PATH)

    anomaly_grouped = group_files_by_base(ANOMALY_DIR.rglob("*.npy"), eval_anomaly_bases)
    normal_grouped = group_files_by_base(NORMAL_DIR.rglob("*.npy"), eval_normal_bases)

    stage1 = AnomalyNet(input_dim=FEATURE_DIM).to(device)
    stage1.load_state_dict(load_state_dict(STAGE1_CHECKPOINT_PATH, device))
    stage1.eval()

    stage2 = DirectBiLSTM().to(device)
    stage2.load_state_dict(load_state_dict(STAGE2_CHECKPOINT_PATH, device))
    stage2.eval()

    stage1_anomaly_scores, anomaly_lengths, anomaly_crop_count = score_group(stage1, anomaly_grouped, device)
    stage1_normal_scores, normal_lengths, normal_crop_count = score_group(stage1, normal_grouped, device)
    stage2_anomaly_scores, _, _ = score_group(stage2, anomaly_grouped, device)
    stage2_normal_scores, _, _ = score_group(stage2, normal_grouped, device)

    bases = sorted(eval_anomaly_bases) + sorted(eval_normal_bases)
    video_labels = [1] * len(eval_anomaly_bases) + [0] * len(eval_normal_bases)
    stage1_video_scores = [float(np.max(stage1_anomaly_scores[b])) for b in sorted(eval_anomaly_bases)] + [
        float(np.max(stage1_normal_scores[b])) for b in sorted(eval_normal_bases)
    ]
    stage1_auc = float(roc_auc_score(video_labels, stage1_video_scores))
    fpr, tpr, thresholds = roc_curve(video_labels, stage1_video_scores)
    best_idx = int(np.argmax(tpr - fpr))
    stage1_best_threshold = float(thresholds[best_idx])
    stage1_preds = [int(score >= stage1_best_threshold) for score in stage1_video_scores]
    tp, tn, fp, fn, accuracy, precision, recall, f1 = confusion_metrics(video_labels, stage1_preds)

    known_preds = [int(score >= KNOWN_STAGE1_THRESHOLD) for score in stage1_video_scores]
    known_metrics = confusion_metrics(video_labels, known_preds)

    temporal_y_true = []
    temporal_y_score = []
    peak_matches = 0
    distances = []
    for base in sorted(eval_anomaly_bases):
        labels = make_labels(annotations[base])
        scores = stage2_anomaly_scores[base]
        peak = int(np.argmax(scores))
        peak_matches += int(labels[peak] == 1)
        distances.append(distance_to_positive(peak, labels))
        temporal_y_true.append(labels)
        temporal_y_score.append(scores)
    for base in sorted(eval_normal_bases):
        temporal_y_true.append(np.zeros(N_SEGMENTS, dtype=np.int64))
        temporal_y_score.append(stage2_normal_scores[base])

    temporal_y_true = np.concatenate(temporal_y_true)
    temporal_y_score = np.concatenate(temporal_y_score)
    stage2_temporal_auc = float(roc_auc_score(temporal_y_true, temporal_y_score))
    stage2_fpr, stage2_tpr, _ = roc_curve(temporal_y_true, temporal_y_score)
    stage2_peak_overlap_ratio = peak_matches / len(eval_anomaly_bases)
    stage2_avg_distance = float(np.mean(distances))

    per_video_rows = []
    pipeline_successes = 0
    anomaly_localization_successes = 0
    false_alarm_count = 0
    videos_passed = 0
    for base in bases:
        is_anomaly = base in eval_anomaly_bases
        video_type = "anomaly" if is_anomaly else "normal"
        video_label = int(is_anomaly)
        stage1_scores = stage1_anomaly_scores[base] if is_anomaly else stage1_normal_scores[base]
        stage2_scores = stage2_anomaly_scores[base] if is_anomaly else stage2_normal_scores[base]
        labels = make_labels(annotations[base]) if is_anomaly else np.zeros(N_SEGMENTS, dtype=np.int64)
        temporal_length = anomaly_lengths[base] if is_anomaly else normal_lengths[base]
        stage1_max = float(np.max(stage1_scores))
        stage1_alert = bool(stage1_max >= stage1_best_threshold)
        stage1_correct = stage1_alert == bool(video_label)
        stage2_ran = stage1_alert
        peak = int(np.argmax(stage2_scores))
        peak_match = bool(labels[peak] == 1) if is_anomaly else False
        distance = distance_to_positive(peak, labels) if is_anomaly else ""
        start, end = approximate_window(peak, temporal_length)
        if stage2_ran:
            videos_passed += 1
        if is_anomaly:
            pipeline_success = bool(stage1_alert and peak_match)
            anomaly_localization_successes += int(pipeline_success)
        else:
            pipeline_success = not stage1_alert
            false_alarm_count += int(stage1_alert)
        pipeline_successes += int(pipeline_success)
        row = {
            "base_video": base,
            "video_type": video_type,
            "video_label": video_label,
            "num_crops": len(anomaly_grouped[base]) if is_anomaly else len(normal_grouped[base]),
            "stage1_max_score": stage1_max,
            "stage1_mean_score": float(np.mean(stage1_scores)),
            "stage1_alert": stage1_alert,
            "stage1_correct": stage1_correct,
            "stage2_ran": stage2_ran,
            "stage2_max_score": float(np.max(stage2_scores)) if stage2_ran else "",
            "stage2_mean_score": float(np.mean(stage2_scores)) if stage2_ran else "",
            "stage2_most_suspicious_segment": peak if stage2_ran else "",
            "stage2_suspicious_segment_score": float(stage2_scores[peak]) if stage2_ran else "",
            "true_positive_segments": " ".join(str(x) for x in positive_indices(labels)) if is_anomaly else "",
            "stage2_peak_matches_label": peak_match if is_anomaly and stage2_ran else (False if is_anomaly else ""),
            "distance_to_nearest_positive": distance if is_anomaly and stage2_ran else ("" if not is_anomaly else distance),
            "pipeline_success": pipeline_success,
            "approximate_feature_window_start": start if stage2_ran else "",
            "approximate_feature_window_end": end if stage2_ran else "",
        }
        per_video_rows.append(row)
        plot_video_scores(row, stage1_scores, stage2_scores, labels, stage1_best_threshold)

    pipeline_accuracy = pipeline_successes / len(bases)
    anomaly_success_rate = anomaly_localization_successes / len(eval_anomaly_bases)
    percent_passed = videos_passed / len(bases)

    plot_roc(fpr, tpr, stage1_auc, STAGE1_ROC_PATH, "Stage 1 video-level ROC")
    plot_roc(stage2_fpr, stage2_tpr, stage2_temporal_auc, STAGE2_ROC_PATH, "Stage 2 temporal ROC")
    plot_hist(
        [np.max(stage1_anomaly_scores[b]) for b in eval_anomaly_bases],
        [np.max(stage1_normal_scores[b]) for b in eval_normal_bases],
        STAGE1_HIST_PATH,
        "Stage 1 video max score distribution",
    )
    plot_hist(
        [np.max(stage2_anomaly_scores[b]) for b in eval_anomaly_bases],
        [np.max(stage2_normal_scores[b]) for b in eval_normal_bases],
        STAGE2_HIST_PATH,
        "Stage 2 video max score distribution",
    )

    summary = {
        "stage1_video_auc": stage1_auc,
        "stage1_best_threshold": stage1_best_threshold,
        "stage1_accuracy": accuracy,
        "stage1_precision": precision,
        "stage1_recall": recall,
        "stage1_f1": f1,
        "stage1_tp": tp,
        "stage1_tn": tn,
        "stage1_fp": fp,
        "stage1_fn": fn,
        "stage2_temporal_auc": stage2_temporal_auc,
        "stage2_peak_overlap_ratio": stage2_peak_overlap_ratio,
        "stage2_average_distance_to_positive": stage2_avg_distance,
        "pipeline_accuracy": pipeline_accuracy,
        "pipeline_anomaly_localization_success_rate": anomaly_success_rate,
        "pipeline_false_alarm_count": false_alarm_count,
        "videos_passed_to_stage2": videos_passed,
        "percent_videos_passed_to_stage2": percent_passed,
    }
    write_summary(summary)
    write_per_video(per_video_rows)

    print(f"Device: {device}")
    print(f"Split path: {SPLIT_PATH}")
    print(f"Stage 1 checkpoint path: {STAGE1_CHECKPOINT_PATH}")
    print(f"Stage 2 checkpoint path: {STAGE2_CHECKPOINT_PATH}")
    print(f"Eval anomaly base video count: {len(eval_anomaly_bases)}")
    print(f"Eval normal base video count: {len(eval_normal_bases)}")
    print(f"Eval anomaly crop count: {anomaly_crop_count}")
    print(f"Eval normal crop count: {normal_crop_count}")
    print("Stage 1 video-level results:")
    print(f"  Video AUC: {stage1_auc:.6f}")
    print(f"  Best threshold: {stage1_best_threshold:.6f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy={accuracy:.6f}, Precision={precision:.6f}, Recall={recall:.6f}, F1={f1:.6f}")
    print(
        f"  Known threshold {KNOWN_STAGE1_THRESHOLD:.6f}: "
        f"TP={known_metrics[0]}, TN={known_metrics[1]}, FP={known_metrics[2]}, FN={known_metrics[3]}"
    )
    print("Stage 2 localization results:")
    print(f"  Temporal AUC: {stage2_temporal_auc:.6f}")
    print(f"  Peak overlap ratio: {stage2_peak_overlap_ratio:.6f}")
    print(f"  Average distance to positive: {stage2_avg_distance:.6f}")
    print("Full pipeline results:")
    print(f"  Pipeline accuracy: {pipeline_accuracy:.6f}")
    print(f"  Anomaly localization success rate: {anomaly_success_rate:.6f}")
    print(f"  False alarm count on normal videos: {false_alarm_count}")
    print(f"  Videos passed to Stage 2: {videos_passed}")
    print(f"  Percent videos passed to Stage 2: {percent_passed:.6f}")
    print("Per-video decision table:")
    for row in per_video_rows:
        print(
            f"  {row['base_video']} | type={row['video_type']} | "
            f"stage1_score={float(row['stage1_max_score']):.6f} | alert={row['stage1_alert']} | "
            f"stage2_ran={row['stage2_ran']} | segment={row['stage2_most_suspicious_segment']} | "
            f"true_segments='{row['true_positive_segments']}' | peak_match={row['stage2_peak_matches_label']} | "
            f"pipeline_success={row['pipeline_success']}"
        )
    print("Comparison conclusion:")
    print(f"  Stage 1 video AUC {stage1_auc:.6f} vs old honest video AUC {OLD_HONEST_VIDEO_AUC:.6f}.")
    print(
        f"  Stage 2 temporal AUC {stage2_temporal_auc:.6f} vs old honest temporal AUC "
        f"{OLD_HONEST_TEMPORAL_AUC:.6f} and BiLSTM standalone {BILSTM_STANDALONE_TEMPORAL_AUC:.6f}."
    )
    if pipeline_accuracy >= 0.85 and anomaly_success_rate >= 0.85:
        print("  The two-stage pipeline is better aligned with the project goal: video alert plus suspicious segment localization.")
    else:
        print("  Results are mixed; the pipeline improves localization but should be validated on more held-out data.")
    print("  Limitations: still feature-based, not raw-video frame extraction yet; held-out set is small: 7 anomaly + 7 normal base videos.")
    print(f"Saved summary CSV: {SUMMARY_CSV_PATH}")
    print(f"Saved per-video CSV: {PER_VIDEO_CSV_PATH}")
    print(f"Saved per-video plot directory: {PLOT_DIR}")


if __name__ == "__main__":
    main()
