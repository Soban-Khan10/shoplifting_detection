import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

from final_two_stage_inference import (
    FEATURE_DIM,
    N_SEGMENTS,
    STAGE2_CHECKPOINT_PATH,
    DirectBiLSTM,
    load_state_dict,
    validate_and_resample_feature,
)


DEFAULT_ANNOTATION_FILE = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
DEFAULT_FEATURE_DIR = Path("data/features/test/anomaly/Shoplifting_anomally_test")
DEFAULT_OUTPUT_CSV = Path("outputs/stage2_label_alignment_debug.csv")
FEATURE_FRAME_STRIDE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Stage 2 Direct BiLSTM localization alignment on annotated Shoplifting videos."
    )
    parser.add_argument(
        "--annotation-file",
        default=str(DEFAULT_ANNOTATION_FILE),
        help="Path to UCF-Crime temporal anomaly annotation file.",
    )
    parser.add_argument(
        "--feature-dir",
        default=str(DEFAULT_FEATURE_DIR),
        help="Directory containing original Shoplifting .npy feature files.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to save diagnostic CSV.",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"], help="Inference device.")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device_name)


def load_shoplifting_annotations(annotation_file):
    rows = []
    with annotation_file.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6 or parts[1] != "Shoplifting":
                continue
            rows.append(parts)
    return rows


def parse_frame_intervals(parts):
    intervals = []
    values = parts[2:]
    for idx in range(0, len(values), 2):
        if idx + 1 >= len(values):
            break
        start = int(values[idx])
        end = int(values[idx + 1])
        if start >= 0 and end >= 0 and end > start:
            intervals.append((start, end))
    return intervals


def feature_path_for(video_name, feature_dir):
    return feature_dir / f"{Path(video_name).stem}.npy"


def load_feature_length(feature_path):
    feature = np.load(feature_path)
    if feature.ndim == 3 and feature.shape[0] == 1:
        return int(feature.shape[1])
    if feature.ndim == 2:
        return int(feature.shape[0])
    raise ValueError(f"Unsupported feature shape for {feature_path}: {feature.shape}")


def frame_interval_to_bins(interval, estimated_total_frames):
    start, end = interval
    start_bin = int(np.floor(start / estimated_total_frames * N_SEGMENTS))
    end_bin = int(np.ceil(end / estimated_total_frames * N_SEGMENTS)) - 1
    start_bin = max(0, min(N_SEGMENTS - 1, start_bin))
    end_bin = max(start_bin, min(N_SEGMENTS - 1, end_bin))
    return start_bin, end_bin


def run_stage2_peak(model, feature_path, device):
    features, original_length = validate_and_resample_feature(feature_path)
    features = features.to(device)
    with torch.no_grad():
        scores = model(features).squeeze(0).cpu().numpy()
    peak_bin = int(np.argmax(scores))
    peak_score = float(scores[peak_bin])
    return peak_bin, peak_score, scores, int(original_length)


def peak_alignment(peak_bin, gt_bin_intervals):
    if not gt_bin_intervals:
        return False, None
    if any(start <= peak_bin <= end for start, end in gt_bin_intervals):
        return True, 0
    distances = []
    for start, end in gt_bin_intervals:
        if peak_bin < start:
            distances.append(start - peak_bin)
        elif peak_bin > end:
            distances.append(peak_bin - end)
        else:
            distances.append(0)
    return False, min(distances)


def format_intervals(intervals):
    return ";".join(f"{start}-{end}" for start, end in intervals)


def write_csv(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_name",
        "feature_path",
        "feature_shape_t",
        "estimated_total_frames",
        "gt_frame_intervals",
        "gt_bin_ranges_32",
        "predicted_peak_bin",
        "predicted_peak_score",
        "overlap",
        "distance_bins",
        "is_last_bin",
        "is_late_bin_27_31",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    annotation_file = Path(args.annotation_file)
    feature_dir = Path(args.feature_dir)
    output_csv = Path(args.output_csv)

    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_file}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    device = select_device(args.device)
    model = DirectBiLSTM().to(device)
    model.load_state_dict(load_state_dict(STAGE2_CHECKPOINT_PATH, device))
    model.eval()

    annotations = load_shoplifting_annotations(annotation_file)
    rows = []
    missing_features = []

    print(f"annotated_shoplifting_videos: {len(annotations)}")
    print(f"feature_dir: {feature_dir}")
    print(f"feature_frame_stride_assumption: {FEATURE_FRAME_STRIDE}")
    print()

    for parts in annotations:
        video_name = parts[0]
        gt_frame_intervals = parse_frame_intervals(parts)
        feature_path = feature_path_for(video_name, feature_dir)
        if not feature_path.exists():
            missing_features.append(video_name)
            print(f"{video_name}: missing feature file {feature_path}")
            continue

        feature_length = load_feature_length(feature_path)
        estimated_total_frames = feature_length * FEATURE_FRAME_STRIDE
        gt_bin_intervals = [
            frame_interval_to_bins(interval, estimated_total_frames)
            for interval in gt_frame_intervals
        ]
        peak_bin, peak_score, scores, original_length = run_stage2_peak(model, feature_path, device)
        overlap, distance_bins = peak_alignment(peak_bin, gt_bin_intervals)

        print(
            f"{video_name}: "
            f"GT bins {format_intervals(gt_bin_intervals)} | "
            f"predicted peak {peak_bin} score {peak_score:.6f} | "
            f"overlap={overlap} distance_bins={distance_bins}"
        )

        rows.append(
            {
                "video_name": video_name,
                "feature_path": str(feature_path),
                "feature_shape_t": feature_length,
                "estimated_total_frames": estimated_total_frames,
                "gt_frame_intervals": format_intervals(gt_frame_intervals),
                "gt_bin_ranges_32": format_intervals(gt_bin_intervals),
                "predicted_peak_bin": peak_bin,
                "predicted_peak_score": peak_score,
                "overlap": overlap,
                "distance_bins": distance_bins,
                "is_last_bin": peak_bin == N_SEGMENTS - 1,
                "is_late_bin_27_31": peak_bin >= 27,
            }
        )

    write_csv(output_csv, rows)

    evaluated = len(rows)
    overlaps = sum(1 for row in rows if row["overlap"])
    last_bin = sum(1 for row in rows if row["is_last_bin"])
    late_bins = sum(1 for row in rows if row["is_late_bin_27_31"])
    peak_bins = [int(row["predicted_peak_bin"]) for row in rows]
    mean_peak = float(np.mean(peak_bins)) if peak_bins else 0.0
    median_peak = float(np.median(peak_bins)) if peak_bins else 0.0

    print()
    print(f"evaluated_videos: {evaluated}")
    print(f"missing_feature_files: {len(missing_features)}")
    if missing_features:
        print(f"missing_feature_video_names: {', '.join(missing_features)}")
    print(f"predicted_peaks_overlap_gt: {overlaps}")
    print(f"predicted_peaks_do_not_overlap_gt: {evaluated - overlaps}")
    print(f"last_bin_31_peak_count: {last_bin}")
    print(f"late_bin_27_31_peak_count: {late_bins}")
    print(f"mean_peak_bin: {mean_peak:.3f}")
    print(f"median_peak_bin: {median_peak:.3f}")
    print(f"saved_csv: {output_csv}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
