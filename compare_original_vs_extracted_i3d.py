import argparse
import json
import math
import os
import sys
from pathlib import Path

OUTPUT_ROOT = Path("outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_ROOT / ".matplotlib"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from final_two_stage_inference import (
    DEFAULT_THRESHOLD,
    FEATURE_DIM,
    N_SEGMENTS,
    STAGE1_CHECKPOINT_PATH,
    STAGE2_CHECKPOINT_PATH,
    DirectBiLSTM,
    load_state_dict,
    validate_and_resample_feature,
)
from models.anomaly_net import AnomalyNet


DEFAULT_ANNOTATION_FILE = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
DEFAULT_OUTPUT_DIR = Path("outputs/i3d_alignment_debug")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original dataset features against newly extracted I3D features."
    )
    parser.add_argument("--original-feature", required=True, help="Path to original dataset .npy feature file.")
    parser.add_argument("--extracted-feature", required=True, help="Path to generated I3D .npy feature file.")
    parser.add_argument("--video-name", required=True, help="Video filename, e.g. Shoplifting001_x264.mp4.")
    parser.add_argument(
        "--annotation-file",
        default=str(DEFAULT_ANNOTATION_FILE),
        help="Path to UCF-Crime temporal annotation file.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for comparison outputs.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Stage 1 alert threshold.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"], help="Inference device.")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device_name)


def load_feature_shape(path):
    return tuple(np.load(path).shape)


def load_models(device):
    stage1 = AnomalyNet(input_dim=FEATURE_DIM).to(device)
    stage1.load_state_dict(load_state_dict(STAGE1_CHECKPOINT_PATH, device))
    stage1.eval()

    stage2 = DirectBiLSTM().to(device)
    stage2.load_state_dict(load_state_dict(STAGE2_CHECKPOINT_PATH, device))
    stage2.eval()
    return stage1, stage2


def run_scores(model, feature_path, device):
    features, original_length = validate_and_resample_feature(feature_path)
    features = features.to(device)
    with torch.no_grad():
        scores = model(features).squeeze(0).cpu().numpy()
    return scores, original_length


def run_two_stage_for_feature(stage1, stage2, feature_path, device, threshold):
    stage1_scores, original_length = run_scores(stage1, feature_path, device)
    stage1_max_score = float(np.max(stage1_scores))
    alert = bool(stage1_max_score >= threshold)

    # Run Stage 2 for diagnostics even when Stage 1 is below threshold.
    stage2_scores, _ = run_scores(stage2, feature_path, device)
    suspicious_segment_index = int(np.argmax(stage2_scores)) if alert else None
    suspicious_segment_score = (
        float(stage2_scores[suspicious_segment_index]) if suspicious_segment_index is not None else None
    )

    return {
        "feature_path": str(feature_path),
        "original_feature_shape": load_feature_shape(feature_path),
        "temporal_length_before_resample": int(original_length),
        "stage1_scores": stage1_scores.tolist(),
        "stage1_max_score": stage1_max_score,
        "threshold": float(threshold),
        "alert": alert,
        "stage2_scores": stage2_scores.tolist(),
        "suspicious_segment_index": suspicious_segment_index,
        "suspicious_segment_score": suspicious_segment_score,
    }


def load_annotation(video_name, annotation_file):
    with annotation_file.open("r") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == video_name:
                return parts
    return None


def parse_gt_intervals(annotation_parts):
    if annotation_parts is None:
        return []
    intervals = []
    values = annotation_parts[2:]
    for idx in range(0, len(values), 2):
        if idx + 1 >= len(values):
            break
        start = int(values[idx])
        end = int(values[idx + 1])
        if start >= 0 and end >= 0 and end > start:
            intervals.append((start, end))
    return intervals


def metadata_path_for_feature(feature_path):
    return feature_path.with_suffix(".metadata.json")


def find_video_duration(video_name, extracted_feature):
    candidates = [Path.home() / "Downloads" / video_name]
    metadata_path = metadata_path_for_feature(extracted_feature)
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        fps = float(metadata["original_fps"])
        total_frames = int(metadata["total_frames"])
        return fps, total_frames, total_frames / fps, str(metadata_path)

    for video_path in candidates:
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        try:
            if not cap.isOpened():
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps > 0 and total_frames > 0:
                return fps, total_frames, total_frames / fps, str(video_path)
        finally:
            cap.release()

    raise FileNotFoundError(
        "Could not determine video duration. Expected an extracted feature metadata JSON "
        f"beside {extracted_feature} or a local video at ~/Downloads/{video_name}."
    )


def segment_to_time(segment_idx, duration_sec):
    if segment_idx is None:
        return None
    start = segment_idx / N_SEGMENTS * duration_sec
    end = (segment_idx + 1) / N_SEGMENTS * duration_sec
    return start, end


def frames_to_seconds(intervals, fps):
    return [(start / fps, end / fps) for start, end in intervals]


def interval_overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_distance(a_start, a_end, b_start, b_end):
    if interval_overlap(a_start, a_end, b_start, b_end) > 0:
        return 0.0
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end


def compare_interval(pred_interval, gt_intervals):
    if pred_interval is None or not gt_intervals:
        return {"overlap": False, "overlap_duration_sec": 0.0, "distance_to_nearest_gt_sec": None}
    pred_start, pred_end = pred_interval
    overlaps = [interval_overlap(pred_start, pred_end, gt_start, gt_end) for gt_start, gt_end in gt_intervals]
    distances = [interval_distance(pred_start, pred_end, gt_start, gt_end) for gt_start, gt_end in gt_intervals]
    max_overlap = max(overlaps)
    return {
        "overlap": bool(max_overlap > 0),
        "overlap_duration_sec": float(max_overlap),
        "distance_to_nearest_gt_sec": float(min(distances)),
    }


def gt_intervals_to_bins(gt_time_intervals, duration_sec):
    bins = []
    if duration_sec <= 0:
        return bins
    for start, end in gt_time_intervals:
        start_bin = max(0.0, min(N_SEGMENTS, start / duration_sec * N_SEGMENTS))
        end_bin = max(0.0, min(N_SEGMENTS, end / duration_sec * N_SEGMENTS))
        bins.append((start_bin, end_bin))
    return bins


def save_plot(original_result, extracted_result, gt_bins, output_path):
    x = np.arange(N_SEGMENTS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 4.5))
    plt.plot(x, original_result["stage2_scores"], marker="o", linewidth=1.8, label="Original features Stage 2")
    plt.plot(x, extracted_result["stage2_scores"], marker="s", linewidth=1.8, label="Extracted I3D Stage 2")
    for idx, (start_bin, end_bin) in enumerate(gt_bins):
        plt.axvspan(start_bin, end_bin, color="green", alpha=0.18, label="GT interval" if idx == 0 else None)
    for label, result, color in [
        ("Original peak", original_result, "black"),
        ("Extracted peak", extracted_result, "red"),
    ]:
        peak_idx = result["suspicious_segment_index"]
        if peak_idx is not None:
            plt.scatter([peak_idx], [result["stage2_scores"][peak_idx]], s=70, color=color, zorder=3, label=label)
    plt.xlabel("32-bin segment index")
    plt.ylabel("Stage 2 suspicious score")
    plt.title("Original vs extracted I3D localization comparison")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def print_result(label, result):
    print(f"{label}_feature_shape: {tuple(result['original_feature_shape'])}")
    print(f"{label}_stage1_max_score: {result['stage1_max_score']}")
    print(f"{label}_alert: {result['alert']}")
    print(f"{label}_suspicious_segment_index: {result['suspicious_segment_index']}")
    print(f"{label}_suspicious_score: {result['suspicious_segment_score']}")
    print(f"{label}_predicted_time_interval_sec: {result['predicted_time_interval_sec']}")
    print(f"{label}_overlap: {result['alignment']['overlap']}")
    print(f"{label}_overlap_duration_sec: {result['alignment']['overlap_duration_sec']:.3f}")
    distance = result["alignment"]["distance_to_nearest_gt_sec"]
    print(f"{label}_distance_to_nearest_gt_sec: {None if distance is None else round(distance, 3)}")


def main():
    args = parse_args()
    original_feature = Path(args.original_feature)
    extracted_feature = Path(args.extracted_feature)
    annotation_file = Path(args.annotation_file)
    output_dir = Path(args.output_dir)

    if not original_feature.exists():
        raise FileNotFoundError(f"Original feature file does not exist: {original_feature}")
    if not extracted_feature.exists():
        raise FileNotFoundError(f"Extracted I3D feature file does not exist: {extracted_feature}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")

    device = select_device(args.device)
    fps, total_frames, duration_sec, duration_source = find_video_duration(args.video_name, extracted_feature)
    annotation_parts = load_annotation(args.video_name, annotation_file)
    gt_frame_intervals = parse_gt_intervals(annotation_parts)
    gt_time_intervals = frames_to_seconds(gt_frame_intervals, fps)
    gt_bins = gt_intervals_to_bins(gt_time_intervals, duration_sec)

    stage1, stage2 = load_models(device)
    original_result = run_two_stage_for_feature(stage1, stage2, original_feature, device, args.threshold)
    extracted_result = run_two_stage_for_feature(stage1, stage2, extracted_feature, device, args.threshold)

    for result in (original_result, extracted_result):
        predicted_interval = segment_to_time(result["suspicious_segment_index"], duration_sec)
        result["predicted_time_interval_sec"] = (
            [float(predicted_interval[0]), float(predicted_interval[1])] if predicted_interval is not None else None
        )
        result["alignment"] = compare_interval(predicted_interval, gt_time_intervals)

    output_stem = Path(args.video_name).stem
    json_path = output_dir / f"{output_stem}_original_vs_extracted_i3d_comparison.json"
    plot_path = output_dir / f"{output_stem}_original_vs_extracted_i3d_stage2.png"

    comparison = {
        "video_name": args.video_name,
        "device": str(device),
        "threshold": float(args.threshold),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "duration_source": duration_source,
        "gt_annotation_found": annotation_parts is not None,
        "gt_class": annotation_parts[1] if annotation_parts is not None else None,
        "gt_frame_intervals": gt_frame_intervals,
        "gt_time_intervals_sec": gt_time_intervals,
        "gt_bins_32": gt_bins,
        "original": original_result,
        "extracted_i3d": extracted_result,
        "json_path": str(json_path),
        "plot_path": str(plot_path),
    }

    write_json(json_path, comparison)
    save_plot(original_result, extracted_result, gt_bins, plot_path)

    print(f"video_name: {args.video_name}")
    print(f"fps: {fps}")
    print(f"duration_sec: {duration_sec:.3f}")
    print(f"total_frames: {total_frames}")
    print(f"gt_annotation_found: {annotation_parts is not None}")
    print(f"gt_frame_intervals: {gt_frame_intervals}")
    print(f"gt_time_intervals_sec: {[(round(start, 3), round(end, 3)) for start, end in gt_time_intervals]}")
    print(f"gt_bins_32: {[(round(start, 3), round(end, 3)) for start, end in gt_bins]}")
    print_result("original", original_result)
    print_result("extracted_i3d", extracted_result)
    print(f"comparison_json_path: {json_path}")
    print(f"comparison_plot_path: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
