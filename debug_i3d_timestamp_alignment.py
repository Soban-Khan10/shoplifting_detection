import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_ANNOTATION_FILE = Path("data/list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare I3D raw-video suspicious timestamps against UCF-Crime temporal annotations."
    )
    parser.add_argument("--video-name", required=True, help="Video filename, e.g. Shoplifting001_x264.mp4.")
    parser.add_argument("--result-json", required=True, help="Path to final inference result JSON.")
    parser.add_argument("--metadata-json", required=True, help="Path to I3D feature metadata JSON.")
    parser.add_argument(
        "--annotation-file",
        default=str(DEFAULT_ANNOTATION_FILE),
        help="Path to UCF-Crime temporal anomaly annotation file.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def load_annotation(video_name, annotation_file):
    with annotation_file.open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == video_name:
                return parts
    return None


def parse_gt_intervals(annotation_parts):
    intervals = []
    if annotation_parts is None:
        return intervals

    frame_values = annotation_parts[2:]
    for idx in range(0, len(frame_values), 2):
        if idx + 1 >= len(frame_values):
            break
        start = int(frame_values[idx])
        end = int(frame_values[idx + 1])
        if start >= 0 and end >= 0 and end > start:
            intervals.append((start, end))
    return intervals


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


def compare_prediction_to_gt(pred_start, pred_end, gt_time_intervals):
    if pred_start is None or pred_end is None or not gt_time_intervals:
        return False, 0.0, None

    overlaps = [
        interval_overlap(pred_start, pred_end, gt_start, gt_end)
        for gt_start, gt_end in gt_time_intervals
    ]
    distances = [
        interval_distance(pred_start, pred_end, gt_start, gt_end)
        for gt_start, gt_end in gt_time_intervals
    ]
    max_overlap = max(overlaps) if overlaps else 0.0
    min_distance = min(distances) if distances else None
    return max_overlap > 0, max_overlap, min_distance


def format_intervals(intervals, precision=None):
    if precision is None:
        return "[" + ", ".join(f"({start}, {end})" for start, end in intervals) + "]"
    return "[" + ", ".join(f"({start:.{precision}f}, {end:.{precision}f})" for start, end in intervals) + "]"


def main():
    args = parse_args()
    result_path = Path(args.result_json)
    metadata_path = Path(args.metadata_json)
    annotation_file = Path(args.annotation_file)

    if not result_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {metadata_path}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    result = load_json(result_path)
    metadata = load_json(metadata_path)
    fps = float(metadata["original_fps"])
    total_frames = int(metadata["total_frames"])
    duration = total_frames / fps if fps > 0 else math.nan

    annotation_parts = load_annotation(args.video_name, annotation_file)
    gt_frame_intervals = parse_gt_intervals(annotation_parts)
    gt_time_intervals = frames_to_seconds(gt_frame_intervals, fps) if fps > 0 else []

    pred_start = result.get("suspicious_start_time_sec")
    pred_end = result.get("suspicious_end_time_sec")
    pred_score = result.get("suspicious_segment_score", result.get("suspicious_score"))
    overlap, overlap_duration, nearest_distance = compare_prediction_to_gt(
        pred_start,
        pred_end,
        gt_time_intervals,
    )

    print(f"video_name: {args.video_name}")
    print(f"original_fps: {fps}")
    print(f"duration_sec: {duration:.3f}")
    print(f"total_frames: {total_frames}")
    if annotation_parts is None:
        print(f"gt_annotation_found: False")
        print(f"gt_note: No ground-truth line exists for {args.video_name} in {annotation_file}.")
    else:
        print(f"gt_annotation_found: True")
        print(f"gt_class: {annotation_parts[1]}")
    print(f"gt_frame_intervals: {format_intervals(gt_frame_intervals)}")
    print(f"gt_time_intervals_sec: {format_intervals(gt_time_intervals, precision=3)}")
    print(f"alert: {result.get('alert')}")
    print(f"stage1_max_score: {result.get('stage1_max_score')}")
    print(f"suspicious_segment_index: {result.get('suspicious_segment_index')}")
    print(f"suspicious_score: {pred_score}")
    print(f"predicted_time_interval_sec: ({pred_start}, {pred_end})")
    print(f"overlap: {overlap}")
    print(f"overlap_duration_sec: {overlap_duration:.3f}")
    if nearest_distance is None:
        print("distance_to_nearest_gt_sec: None")
    else:
        print(f"distance_to_nearest_gt_sec: {nearest_distance:.3f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
