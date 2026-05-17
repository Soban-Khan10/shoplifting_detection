import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


RAW_VIDEO_STAGE_DIR = Path("outputs/raw_video_stage")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare raw video frames for future I3D feature extraction. "
            "This script does not generate final 1024-d I3D features yet."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Future output path for extracted .npy features.")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=10.0,
        help="Frame sampling FPS. Must be greater than zero.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Square resize dimension for sampled frames.",
    )
    return parser.parse_args()


def validate_args(video_path, sample_fps, resize):
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if sample_fps <= 0:
        raise ValueError(f"--sample-fps must be greater than zero, got {sample_fps}")
    if resize <= 0:
        raise ValueError(f"--resize must be greater than zero, got {resize}")


def metadata_path_for(output_path):
    return output_path.with_suffix(".metadata.json")


def sampled_frames_path_for(video_path):
    RAW_VIDEO_STAGE_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_VIDEO_STAGE_DIR / f"{video_path.stem}_sampled_frames.npy"


def read_sampled_frames(video_path, sample_fps, resize):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if original_fps <= 0:
            raise ValueError(f"Invalid original FPS reported by video: {original_fps}")

        duration_seconds = total_frames / original_fps if total_frames > 0 else 0.0
        sample_interval = max(1, int(round(original_fps / sample_fps)))
        sampled_frames = []
        frame_index = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % sample_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
                sampled_frames.append(frame_rgb)
            frame_index += 1
    finally:
        cap.release()

    if not sampled_frames:
        raise RuntimeError(f"Zero frames sampled from video: {video_path}")

    return (
        np.stack(sampled_frames, axis=0).astype(np.uint8),
        original_fps,
        total_frames,
        duration_seconds,
    )


def save_metadata(
    metadata_path,
    video_path,
    original_fps,
    total_frames,
    sampled_frame_count,
    duration_seconds,
    resize,
    output_feature_path,
    sampled_frames_path,
):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "video_path": str(video_path),
        "original_fps": original_fps,
        "total_frames": total_frames,
        "sampled_frame_count": sampled_frame_count,
        "duration_seconds": duration_seconds,
        "resize_size": resize,
        "output_feature_path": str(output_feature_path),
        "sampled_frames_path": str(sampled_frames_path),
        "note": "Sampled frames only. Real I3D 1024-d feature extraction is not implemented yet.",
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_feature_path = Path(args.output)
    metadata_path = metadata_path_for(output_feature_path)
    sampled_frames_path = sampled_frames_path_for(video_path)

    validate_args(video_path, args.sample_fps, args.resize)

    # TODO: Insert actual I3D model loading and 1024-d feature extraction here.
    # The current pipeline expects final features shaped (T, 1024). Until that
    # model stage is added, this script saves sampled/preprocessed frames only.
    frames, original_fps, total_frames, duration_seconds = read_sampled_frames(
        video_path=video_path,
        sample_fps=args.sample_fps,
        resize=args.resize,
    )

    np.save(sampled_frames_path, frames)
    save_metadata(
        metadata_path=metadata_path,
        video_path=video_path,
        original_fps=original_fps,
        total_frames=total_frames,
        sampled_frame_count=int(frames.shape[0]),
        duration_seconds=duration_seconds,
        resize=args.resize,
        output_feature_path=output_feature_path,
        sampled_frames_path=sampled_frames_path,
    )

    print(f"Input video path: {video_path}")
    print(f"Original FPS: {original_fps:.6f}")
    print(f"Total frames: {total_frames}")
    print(f"Sampled frame count: {frames.shape[0]}")
    print(f"Duration seconds: {duration_seconds:.6f}")
    print(f"Saved sampled frames path: {sampled_frames_path}")
    print(f"Saved metadata path: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
