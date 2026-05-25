import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from feature_extractors.i3d import InceptionI3d


FEATURE_DIM = 1024
DEFAULT_OUTPUT_ROOT = Path("outputs/video_detection_i3d")
WEIGHTS_HELP = (
    "Path to real pretrained RGB Inception-I3D Kinetics weights, for example "
    "a converted PyTorch rgb_imagenet.pt/rgb_kinetics.pt state dict. "
    "Random weights, missing weights, and S3D fallback are not allowed."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract real RGB Inception-I3D 1024-d features from an input video."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--output", required=True, help="Path to save output .npy features.")
    parser.add_argument("--weights", required=True, help=WEIGHTS_HELP)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Inference device. Default is CPU.",
    )
    parser.add_argument("--clip-len", type=int, default=64, help="I3D clip length in frames.")
    parser.add_argument("--stride", type=int, default=32, help="Frame stride between clip starts.")
    parser.add_argument("--resize", type=int, default=256, help="Square resize size before center crop.")
    parser.add_argument("--crop-size", type=int, default=224, help="Square center crop size.")
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device_name)


def metadata_path_for(output_path):
    return output_path.with_suffix(".metadata.json")


def validate_args(video_path, output_path, weights_path, clip_len, stride, resize, crop_size):
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if output_path.suffix.lower() != ".npy":
        raise ValueError(f"--output must end in .npy, got: {output_path}")
    if not weights_path.exists() or not weights_path.is_file():
        raise FileNotFoundError(
            "Real RGB I3D pretrained weights are required and were not found. "
            f"Provide a valid local file with --weights. Requested path: {weights_path}. "
            "This extractor does not use random weights, fake features, or S3D fallback."
        )
    if clip_len <= 0:
        raise ValueError(f"--clip-len must be greater than zero, got {clip_len}")
    if stride <= 0:
        raise ValueError(f"--stride must be greater than zero, got {stride}")
    if resize <= 0:
        raise ValueError(f"--resize must be greater than zero, got {resize}")
    if crop_size <= 0:
        raise ValueError(f"--crop-size must be greater than zero, got {crop_size}")
    if resize < crop_size:
        raise ValueError(f"--resize must be >= --crop-size, got resize={resize}, crop_size={crop_size}")


def center_crop(frame, crop_size):
    height, width = frame.shape[:2]
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    return frame[top:top + crop_size, left:left + crop_size]


def read_preprocessed_frames(video_path, resize, crop_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    try:
        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (resize, resize), interpolation=cv2.INTER_AREA)
            frame_rgb = center_crop(frame_rgb, crop_size)
            frames.append(frame_rgb)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from video: {video_path}")
    if original_fps <= 0:
        raise RuntimeError(f"Invalid FPS reported by video: {original_fps}")

    return np.stack(frames, axis=0).astype(np.uint8), original_fps, total_frames


def build_clips(frames, clip_len, stride):
    clips = []
    for start in range(0, frames.shape[0] - clip_len + 1, stride):
        clips.append(frames[start:start + clip_len])
    if not clips:
        raise RuntimeError(
            f"No clips created: video has {frames.shape[0]} usable frames, "
            f"but clip_len={clip_len} and stride={stride}."
        )
    return clips


def preprocess_clip(clip):
    clip = clip.astype(np.float32)
    clip = (clip / 127.5) - 1.0
    tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
    return tensor.contiguous()


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def load_checkpoint_state_dict(weights_path, device):
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return normalize_state_dict_keys(checkpoint[key])
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return normalize_state_dict_keys(checkpoint)

    raise RuntimeError(
        f"Could not find a PyTorch state_dict in weights file: {weights_path}. "
        "Expected a plain state dict or a checkpoint containing state_dict/model_state_dict."
    )


def load_i3d_model(weights_path, device):
    model = InceptionI3d(num_classes=400, in_channels=3)
    state_dict = load_checkpoint_state_dict(weights_path, device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load RGB I3D weights into the local Inception-I3D model definition. "
            "Confirm the checkpoint is for RGB Inception-I3D with compatible key names and 400-class logits. "
            f"Original error: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model


def extract_features(model, clips, device):
    features = []
    with torch.no_grad():
        for clip in clips:
            tensor = preprocess_clip(clip).to(device)
            feature = model.extract_features(tensor).squeeze(0).cpu().numpy()
            if feature.shape[0] != FEATURE_DIM:
                raise RuntimeError(f"Expected I3D feature dimension {FEATURE_DIM}, got {feature.shape[0]}.")
            features.append(feature)

    output = np.stack(features, axis=0).astype(np.float32)
    if output.ndim != 2 or output.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Expected output feature shape (T, {FEATURE_DIM}), got {output.shape}.")
    return output


def save_metadata(
    metadata_path,
    video_path,
    weights_path,
    original_fps,
    total_frames,
    used_frame_count,
    clip_len,
    stride,
    num_features,
    feature_dim,
    device,
    output_path,
    resize,
    crop_size,
):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "video_path": str(video_path),
        "weights_path": str(weights_path),
        "original_fps": original_fps,
        "total_frames": total_frames,
        "sampled_or_used_frame_count": used_frame_count,
        "clip_len": clip_len,
        "stride": stride,
        "number_of_clips_features": num_features,
        "feature_dim": feature_dim,
        "device": str(device),
        "extractor": "inception_i3d_rgb",
        "output_feature_path": str(output_path),
        "preprocessing": {
            "color": "RGB",
            "resize": "square",
            "resize_size": resize,
            "center_crop": True,
            "crop_size": crop_size,
            "normalization": "pixel / 127.5 - 1.0",
        },
        "compatibility_warning": (
            "These features require real RGB Inception-I3D Kinetics weights. "
            "Scores may still differ from training-time features if preprocessing or weights do not match."
        ),
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    output_path = Path(args.output)
    weights_path = Path(args.weights).expanduser()
    metadata_path = metadata_path_for(output_path)

    validate_args(
        video_path=video_path,
        output_path=output_path,
        weights_path=weights_path,
        clip_len=args.clip_len,
        stride=args.stride,
        resize=args.resize,
        crop_size=args.crop_size,
    )
    if DEFAULT_OUTPUT_ROOT not in output_path.parents:
        print(
            "Warning: I3D outputs should be kept separate from S3D outputs. "
            f"Recommended output root: {DEFAULT_OUTPUT_ROOT}",
            file=sys.stderr,
        )

    device = select_device(args.device)
    model = load_i3d_model(weights_path, device)
    frames, original_fps, total_frames = read_preprocessed_frames(
        video_path=video_path,
        resize=args.resize,
        crop_size=args.crop_size,
    )
    clips = build_clips(frames, clip_len=args.clip_len, stride=args.stride)
    features = extract_features(model, clips, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    save_metadata(
        metadata_path=metadata_path,
        video_path=video_path,
        weights_path=weights_path,
        original_fps=original_fps,
        total_frames=total_frames,
        used_frame_count=int(frames.shape[0]),
        clip_len=args.clip_len,
        stride=args.stride,
        num_features=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        device=device,
        output_path=output_path,
        resize=args.resize,
        crop_size=args.crop_size,
    )

    print(f"Input video path: {video_path}")
    print(f"I3D weights path: {weights_path}")
    print(f"Device: {device}")
    print(f"Number of clips: {features.shape[0]}")
    print(f"Output feature shape: {features.shape}")
    print(f"Saved feature path: {output_path}")
    print(f"Saved metadata path: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
