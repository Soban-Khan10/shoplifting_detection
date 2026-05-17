import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_THRESHOLD = 0.052090


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


def run_stage(command):
    result = subprocess.run(command)
    if result.returncode != 0:
        return result.returncode
    return 0


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
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
