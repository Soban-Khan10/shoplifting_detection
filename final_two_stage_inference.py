import argparse
import json
import os
import re
import sys
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.anomaly_net import AnomalyNet


STAGE1_CHECKPOINT_PATH = Path("anomaly_net_weights.pth")
STAGE2_CHECKPOINT_PATH = Path("temporal_model_direct_bilstm.pth")
FEATURE_DIM = 1024
N_SEGMENTS = 32
DEFAULT_THRESHOLD = 0.052090


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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run finalized two-stage feature-based inference on one .npy feature file "
            "or a folder of crop .npy files for one base video."
        )
    )
    parser.add_argument("--input", required=True, help="Path to a .npy feature file or folder of crop .npy files.")
    parser.add_argument("--output-dir", default="outputs/final_inference", help="Directory for plot and JSON outputs.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Finalized Stage 1 alert threshold.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Inference device. Default uses MPS when available, otherwise CPU.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top Stage 2 suspicious segments to report when alert is True.",
    )
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Device 'mps' was requested, but MPS is not available.")
    return torch.device(device_name)


def load_state_dict(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def discover_feature_files(input_path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".npy":
            raise ValueError(f"Input file must be a .npy feature file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found in input folder: {input_path}")
        grouped = {}
        for path in files:
            grouped.setdefault(base_video_name(path), []).append(path)
        if len(grouped) != 1:
            examples = ", ".join(sorted(grouped)[:5])
            raise ValueError(
                "Folder input must contain crop .npy files for one base video only. "
                f"Found {len(grouped)} base video names in {input_path}. "
                f"Examples: {examples}. "
                "Pass a single .npy file or a folder containing only crops for one video."
            )
        return files
    raise ValueError(f"Input path must be a .npy file or folder: {input_path}")


def base_video_name(path):
    return re.sub(r"__\d+$", "", path.stem)


def validate_and_resample_feature(path):
    feat = np.load(path)

    if feat.ndim == 1:
        raise ValueError(
            f"Expected feature array shape (T, {FEATURE_DIM}) or (1, T, {FEATURE_DIM}) "
            f"for {path}, got 1D shape {feat.shape}."
        )
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)
    elif feat.ndim == 3:
        raise ValueError(
            f"Expected 3D feature array with leading batch size 1 for {path}, got shape {feat.shape}."
        )
    if feat.ndim != 2:
        raise ValueError(
            f"Expected feature array shape (T, {FEATURE_DIM}) or (1, T, {FEATURE_DIM}) "
            f"for {path}, got shape {feat.shape}."
        )
    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM} for {path}, got {feat.shape[1]} from shape {feat.shape}."
        )
    if feat.shape[0] < 1:
        raise ValueError(f"Feature array must contain at least one temporal row for {path}, got shape {feat.shape}.")

    original_length = int(feat.shape[0])
    if original_length != N_SEGMENTS:
        indices = np.linspace(0, original_length - 1, N_SEGMENTS, dtype=int)
        feat = feat[indices]

    return torch.FloatTensor(feat).unsqueeze(0), original_length


def run_model_on_files(model, feature_files, device):
    all_scores = []
    original_lengths = []
    for path in feature_files:
        features, original_length = validate_and_resample_feature(path)
        features = features.to(device)
        with torch.no_grad():
            scores = model(features).squeeze(0).cpu().numpy()
        all_scores.append(scores)
        original_lengths.append(original_length)
    return np.mean(np.stack(all_scores, axis=0), axis=0), original_lengths


def approximate_window(segment_idx, temporal_length):
    start = int(segment_idx / N_SEGMENTS * temporal_length)
    end = int((segment_idx + 1) / N_SEGMENTS * temporal_length)
    return start, end


def top_k_segments(stage2_scores, temporal_length, top_k):
    if top_k <= 0:
        raise ValueError(f"--top-k must be greater than zero, got {top_k}")
    limit = min(int(top_k), len(stage2_scores))
    ranked_indices = np.argsort(stage2_scores)[::-1][:limit]
    segments = []
    for rank, segment_idx in enumerate(ranked_indices, start=1):
        segment_idx = int(segment_idx)
        start, end = approximate_window(segment_idx, temporal_length)
        segments.append(
            {
                "rank": int(rank),
                "segment_index": segment_idx,
                "score": float(stage2_scores[segment_idx]),
                "approx_feature_start": int(start),
                "approx_feature_end": int(end),
            }
        )
    return segments


def safe_output_stem(input_path):
    name = input_path.stem if input_path.is_file() else input_path.name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "inference"


def save_score_plot(stage1_scores, stage2_scores, threshold, alert, peak_idx, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(N_SEGMENTS)

    plt.figure(figsize=(10, 4))
    plt.plot(x, stage1_scores, marker="o", linewidth=1.8, label="Stage 1 AnomalyNet")
    if stage2_scores is not None:
        plt.plot(x, stage2_scores, marker="s", linewidth=1.8, label="Stage 2 BiLSTM")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Stage 1 threshold")
    if alert and stage2_scores is not None and peak_idx is not None:
        plt.scatter([peak_idx], [stage2_scores[peak_idx]], color="black", s=80, zorder=3, label="Stage 2 peak")
    plt.xlabel("Segment index")
    plt.ylabel("Score")
    plt.title(f"Final two-stage inference | alert={alert}")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_json(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    device = select_device(args.device)

    feature_files = discover_feature_files(input_path)

    stage1 = AnomalyNet(input_dim=FEATURE_DIM).to(device)
    stage1.load_state_dict(load_state_dict(STAGE1_CHECKPOINT_PATH, device))
    stage1.eval()

    stage2 = DirectBiLSTM().to(device)
    stage2.load_state_dict(load_state_dict(STAGE2_CHECKPOINT_PATH, device))
    stage2.eval()

    stage1_scores, original_lengths = run_model_on_files(stage1, feature_files, device)
    stage1_max_score = float(np.max(stage1_scores))
    alert = bool(stage1_max_score >= args.threshold)

    stage2_scores = None
    suspicious_segment_index = None
    suspicious_segment_score = None
    window_start = None
    window_end = None
    top_suspicious_segments = []
    temporal_length = int(max(original_lengths))

    if alert:
        stage2_scores, _ = run_model_on_files(stage2, feature_files, device)
        suspicious_segment_index = int(np.argmax(stage2_scores))
        suspicious_segment_score = float(stage2_scores[suspicious_segment_index])
        window_start, window_end = approximate_window(suspicious_segment_index, temporal_length)
        top_suspicious_segments = top_k_segments(stage2_scores, temporal_length, args.top_k)

    output_stem = safe_output_stem(input_path)
    plot_path = output_dir / f"{output_stem}_scores.png"
    json_path = output_dir / f"{output_stem}_result.json"
    save_score_plot(stage1_scores, stage2_scores, args.threshold, alert, suspicious_segment_index, plot_path)

    result = {
        "input_path": str(input_path),
        "feature_files": [str(path) for path in feature_files],
        "num_feature_files_used": len(feature_files),
        "device": str(device),
        "stage1_max_score": stage1_max_score,
        "threshold": float(args.threshold),
        "alert": alert,
        "stage1_scores": stage1_scores.tolist(),
        "stage2_ran": alert,
        "stage2_scores": stage2_scores.tolist() if stage2_scores is not None else None,
        "suspicious_segment_index": suspicious_segment_index,
        "suspicious_segment_score": suspicious_segment_score,
        "approximate_feature_window_start": window_start,
        "approximate_feature_window_end": window_end,
        "top_k": int(args.top_k),
        "top_suspicious_segments": top_suspicious_segments,
        "original_feature_lengths": original_lengths,
        "temporal_length_used_for_window": temporal_length,
        "plot_path": str(plot_path),
        "json_path": str(json_path),
    }
    if not alert:
        result["message"] = "No suspicious segment was localized because Stage 1 alert is False."
    write_json(result, json_path)

    print(f"Input path: {input_path}")
    print(f"Number of feature files used: {len(feature_files)}")
    print(f"Device: {device}")
    print(f"Stage 1 max score: {stage1_max_score:.6f}")
    print(f"Threshold: {args.threshold:.6f}")
    print(f"Alert: {alert}")
    if alert:
        print(f"Suspicious segment index: {suspicious_segment_index}")
        print(f"Suspicious segment score: {suspicious_segment_score:.6f}")
        print(f"Approximate feature index window: {window_start}-{window_end}")
        print(f"Top suspicious segments: {top_suspicious_segments}")
    else:
        print("No suspicious segment was localized.")
        print("Approximate feature index window: None")
    print(f"Saved plot path: {plot_path}")
    print(f"Saved JSON path: {json_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
