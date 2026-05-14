import argparse
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.anomaly_net import AnomalyNet


DEFAULT_CHECKPOINT_PATH = Path("anomaly_net_temporal_weights.pth")
FEATURE_DIM = 1024
N_SEGMENTS = 32
DEFAULT_THRESHOLD = 0.198889


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim", FEATURE_DIM)
        n_segments = checkpoint.get("n_segments", N_SEGMENTS)
        state_dict = checkpoint["model_state_dict"]
    else:
        input_dim = FEATURE_DIM
        n_segments = N_SEGMENTS
        state_dict = checkpoint

    return state_dict, input_dim, n_segments


def load_features(path, n_segments):
    feat = np.load(path)
    original_shape = feat.shape

    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)

    if feat.ndim != 2:
        raise ValueError(
            f"Expected feature array with shape (T, {FEATURE_DIM}) "
            f"or (1, T, {FEATURE_DIM}) for {path}, got {original_shape}"
        )

    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM} for {path}, "
            f"got {feat.shape[1]} from shape {feat.shape}"
        )

    original_length = feat.shape[0]
    if original_length != n_segments:
        idx = np.linspace(0, original_length - 1, n_segments, dtype=int)
        feat = feat[idx]

    return torch.FloatTensor(feat).unsqueeze(0), original_shape, original_length


def suspicious_window(segment_idx, original_length, n_segments):
    start = int(segment_idx / n_segments * original_length)
    end = int((segment_idx + 1) / n_segments * original_length)
    return start, end


def save_score_plot(scores, threshold, suspicious_idx, output_path):
    x = np.arange(len(scores))

    plt.figure(figsize=(10, 4))
    plt.plot(x, scores, marker="o", linewidth=2, label="Segment score")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")
    plt.scatter(
        [suspicious_idx],
        [scores[suspicious_idx]],
        color="black",
        s=80,
        zorder=3,
        label="Most suspicious segment",
    )
    plt.xlabel("Segment index")
    plt.ylabel("Anomaly score")
    plt.title("Temporal fine-tuned inference scores")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run temporal fine-tuned anomaly inference on one extracted feature file."
    )
    parser.add_argument("--feature", required=True, help="Path to a .npy feature file.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to temporal fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Alert threshold from temporal fine-tuned video-level ROC evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = Path(args.feature)
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    state_dict, input_dim, n_segments = load_checkpoint(checkpoint_path, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    features, original_shape, original_length = load_features(feature_path, n_segments)
    features = features.to(device)

    with torch.no_grad():
        scores = model(features).squeeze(0).cpu().numpy()

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    alert = max_score >= args.threshold
    suspicious_idx = int(np.argmax(scores))
    suspicious_score = float(scores[suspicious_idx])
    start_idx, end_idx = suspicious_window(suspicious_idx, original_length, n_segments)

    output_path = OUTPUT_DIR / "temporal_inference_scores.png"
    save_score_plot(scores, args.threshold, suspicious_idx, output_path)

    print("Note: This is temporal inference on extracted features, not raw-video frame extraction yet.")
    print(f"Device: {device}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Feature path: {feature_path}")
    print(f"Original feature shape: {original_shape}")
    print(f"Segment scores: {scores.tolist()}")
    print(f"Max score: {max_score:.6f}")
    print(f"Mean score: {mean_score:.6f}")
    print(f"Threshold: {args.threshold:.6f}")
    print(f"Alert: {alert}")
    print(f"Most suspicious segment index: {suspicious_idx}")
    print(f"Suspicious segment score: {suspicious_score:.6f}")
    print(f"Approx suspicious feature window: {start_idx} to {end_idx}")


if __name__ == "__main__":
    main()
