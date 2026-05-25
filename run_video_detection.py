import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cv2


DEFAULT_THRESHOLD = 0.052090
N_MODEL_SEGMENTS = 32
TIMESTAMP_MAPPING_NOTE = (
    "Approximate timestamp mapped from the Stage 2 32-segment model index to the "
    "S3D clip timeline using clip_len, stride, FPS, and total frame metadata. "
    "This is a review window, not an exact frame-level event boundary."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run raw video S3D feature extraction followed by two-stage anomaly inference."
    )
    parser.add_argument("--video", required=True, help="Path to the input .mp4 video.")
    parser.add_argument("--output-dir", default="outputs/video_detection", help="Directory for generated outputs.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Device for both stages. Default is CPU because S3D MPS max_pool3d is unsupported in this environment.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Stage 1 inference threshold.")
    parser.add_argument("--clip-len", type=int, default=32, help="S3D clip length in frames.")
    parser.add_argument("--stride", type=int, default=16, help="S3D clip stride in frames.")
    parser.add_argument("--resize", type=int, default=256, help="S3D resize size before center crop.")
    parser.add_argument("--crop-size", type=int, default=224, help="S3D center crop size.")
    return parser.parse_args()


def safe_stem(path):
    return path.stem.replace(" ", "_")


def safe_output_stem(input_path):
    name = input_path.stem if input_path.is_file() else input_path.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "inference"


def metadata_path_for(feature_path):
    return feature_path.with_suffix(".metadata.json")


def inference_json_path_for(feature_path, output_dir):
    return output_dir / f"{safe_output_stem(feature_path)}_result.json"


def read_json(path):
    with path.open("r") as f:
        return json.load(f)


def write_json(path, data):
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def run_stage(command):
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode
    return 0


def require_positive_number(metadata, key):
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"S3D metadata is missing required field: {key}")
    value = float(value)
    if value <= 0:
        raise ValueError(f"S3D metadata field {key} must be positive, got {value}")
    return value


def require_positive_int(metadata, key):
    value = require_positive_number(metadata, key)
    return int(round(value))


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def map_segment_to_video_window(segment_idx, metadata):
    fps = require_positive_number(metadata, "original_fps")
    total_frames = require_positive_int(metadata, "total_frames")
    clip_len = require_positive_int(metadata, "clip_len")
    stride = require_positive_int(metadata, "stride")
    num_features = require_positive_int(metadata, "number_of_clips_features")

    if not 0 <= segment_idx < N_MODEL_SEGMENTS:
        raise ValueError(
            f"suspicious_segment_index must be in [0, {N_MODEL_SEGMENTS - 1}], got {segment_idx}"
        )

    feature_start = int(math.floor(segment_idx / N_MODEL_SEGMENTS * num_features))
    feature_end = int(math.ceil((segment_idx + 1) / N_MODEL_SEGMENTS * num_features))
    feature_start = clamp(feature_start, 0, max(num_features - 1, 0))
    feature_end = clamp(feature_end, feature_start + 1, num_features)

    frame_start = feature_start * stride
    frame_end = ((feature_end - 1) * stride) + clip_len
    frame_start = clamp(frame_start, 0, max(total_frames - 1, 0))
    frame_end = clamp(frame_end, frame_start + 1, total_frames)

    if frame_end <= frame_start:
        frame_end = min(total_frames, frame_start + max(clip_len, 1))
    if frame_end <= frame_start and frame_start > 0:
        frame_start = max(0, frame_end - 1)

    duration_sec = total_frames / fps
    start_sec = clamp(frame_start / fps, 0.0, duration_sec)
    end_sec = clamp(frame_end / fps, start_sec, duration_sec)

    return {
        "feature_start": feature_start,
        "feature_end": feature_end,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def open_video_capture(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for suspicious clip extraction: {video_path}")
    return cap


def save_suspicious_clip(video_path, output_path, frame_start, frame_end):
    cap = open_video_capture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Could not read valid video properties from: {video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open suspicious clip writer: {output_path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            current_frame = frame_start
            while current_frame < frame_end:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                current_frame += 1
        finally:
            writer.release()
    finally:
        cap.release()


def save_key_frame(video_path, output_path, frame_index):
    cap = open_video_capture(video_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read key frame {frame_index} from: {video_path}")
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save suspicious key frame: {output_path}")
    finally:
        cap.release()


def add_null_suspicious_outputs(result):
    result.setdefault("suspicious_start_time_sec", None)
    result.setdefault("suspicious_end_time_sec", None)
    result.setdefault("suspicious_clip_path", None)
    result.setdefault("suspicious_key_frame_path", None)
    result.setdefault("timestamp_mapping_note", None)


def extract_suspicious_outputs(video_path, output_dir, feature_path):
    json_path = inference_json_path_for(feature_path, output_dir)
    metadata_path = metadata_path_for(feature_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Expected inference JSON was not found: {json_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Expected S3D metadata JSON was not found: {metadata_path}")

    result = read_json(json_path)
    if not result.get("alert", False):
        add_null_suspicious_outputs(result)
        write_json(json_path, result)
        print("Stage 3: Alert is False; suspicious clip and key frame were not created.")
        return

    segment_idx = result.get("suspicious_segment_index")
    if segment_idx is None:
        raise ValueError("Inference JSON has alert=True but no suspicious_segment_index.")
    segment_idx = int(segment_idx)

    metadata = read_json(metadata_path)
    source_video_path = Path(metadata.get("video_path") or video_path).expanduser()
    if not source_video_path.exists():
        source_video_path = video_path

    window = map_segment_to_video_window(segment_idx, metadata)
    output_stem = safe_output_stem(video_path)
    clip_path = output_dir / f"{output_stem}_suspicious_clip.mp4"
    key_frame_path = output_dir / f"{output_stem}_suspicious_key_frame.jpg"
    key_frame_index = (window["frame_start"] + window["frame_end"]) // 2

    save_suspicious_clip(
        video_path=source_video_path,
        output_path=clip_path,
        frame_start=window["frame_start"],
        frame_end=window["frame_end"],
    )
    save_key_frame(
        video_path=source_video_path,
        output_path=key_frame_path,
        frame_index=key_frame_index,
    )

    result.update(
        {
            "suspicious_start_time_sec": round(float(window["start_sec"]), 3),
            "suspicious_end_time_sec": round(float(window["end_sec"]), 3),
            "suspicious_clip_path": str(clip_path),
            "suspicious_key_frame_path": str(key_frame_path),
            "timestamp_mapping_note": TIMESTAMP_MAPPING_NOTE,
        }
    )
    write_json(json_path, result)

    print("Stage 3: Saved suspicious clip and key frame")
    print(f"Suspicious timestamp: {window['start_sec']:.3f}s-{window['end_sec']:.3f}s")
    print(f"Saved suspicious clip path: {clip_path}")
    print(f"Saved suspicious key frame path: {key_frame_path}")


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Error: video path does not exist: {video_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{safe_stem(video_path)}_s3d_features.npy"

    print(
        "Warning: S3D features are shape-compatible with the existing (T, 1024) pipeline, "
        "but may not exactly match the original UCF I3D features used to train the anomaly models."
    )
    print()

    print("Stage 1: Extracting S3D features")
    extract_command = [
        "python",
        "extract_s3d_features_from_video.py",
        "--video",
        str(video_path),
        "--output",
        str(feature_path),
        "--device",
        args.device,
        "--clip-len",
        str(args.clip_len),
        "--stride",
        str(args.stride),
        "--resize",
        str(args.resize),
        "--crop-size",
        str(args.crop_size),
    ]
    status = run_stage(extract_command)
    if status != 0:
        print(f"Error: S3D feature extraction failed with exit code {status}.", file=sys.stderr)
        return status

    print()
    print("Stage 2: Running two-stage anomaly inference")
    inference_command = [
        "python",
        "final_two_stage_inference.py",
        "--input",
        str(feature_path),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--threshold",
        f"{args.threshold:.6f}",
    ]
    status = run_stage(inference_command)
    if status != 0:
        print(f"Error: two-stage anomaly inference failed with exit code {status}.", file=sys.stderr)
        return status

    print()
    print("Stage 3: Extracting suspicious timestamp, clip, and key frame")
    try:
        extract_suspicious_outputs(
            video_path=video_path,
            output_dir=output_dir,
            feature_path=feature_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: suspicious clip extraction failed: {exc}", file=sys.stderr)
        return 1

    print()
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
