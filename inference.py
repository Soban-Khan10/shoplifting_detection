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


CHECKPOINT_PATH = "anomaly_net_weights.pth"
FEATURE_DIM = 1024
DEFAULT_N_SEGMENTS = 32


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim", FEATURE_DIM)
        n_segments = checkpoint.get("n_segments", DEFAULT_N_SEGMENTS)
        state_dict = checkpoint["model_state_dict"]
    else:
        input_dim = FEATURE_DIM
        n_segments = DEFAULT_N_SEGMENTS
        state_dict = checkpoint

    return state_dict, input_dim, n_segments


def load_features(path, n_segments):
    feat = np.load(path)

    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = np.squeeze(feat, axis=0)

    if feat.ndim != 2:
        raise ValueError(
            f"Expected feature array with shape (T, {FEATURE_DIM}) "
            f"or (1, T, {FEATURE_DIM}) for {path}, got {feat.shape}"
        )

    if feat.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected feature dimension {FEATURE_DIM} for {path}, "
            f"got {feat.shape[1]} from shape {feat.shape}"
        )

    if feat.shape[0] != n_segments:
        idx = np.linspace(0, feat.shape[0] - 1, n_segments, dtype=int)
        feat = feat[idx]

    return torch.FloatTensor(feat).unsqueeze(0)


def save_score_plot(scores, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, scores, marker="o", linewidth=2)
    plt.xlabel("Segment")
    plt.ylabel("Anomaly score")
    plt.title("Inference segment scores")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run anomaly inference on a UCF Crime feature file.")
    parser.add_argument("--feature", required=True, help="Path to a .npy feature file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.052090,
        help="Alert threshold for max anomaly score from video-level ROC evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_path = Path(args.feature)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    state_dict, input_dim, n_segments = load_checkpoint(CHECKPOINT_PATH, device)
    model = AnomalyNet(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    features = load_features(feature_path, n_segments).to(device)

    with torch.no_grad():
        scores = model(features).squeeze(0).cpu().numpy()

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    alert = max_score >= args.threshold

    output_path = OUTPUT_DIR / "inference_scores.png"
    save_score_plot(scores, output_path)

    print(f"Device: {device}")
    print(f"Feature path: {feature_path}")
    print(f"Segment scores: {scores.tolist()}")
    print(f"Max score: {max_score:.6f}")
    print(f"Mean score: {mean_score:.6f}")
    print(f"Threshold: {args.threshold:.6f}")
    print(f"Alert: {alert}")


if __name__ == "__main__":
    main()
