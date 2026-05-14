from pathlib import Path
import shutil


BACKUP_DIR = Path("outputs/cleanup_backup")
MANIFEST_PATH = BACKUP_DIR / "removed_files_manifest.txt"


EXPERIMENTAL_SCRIPTS = [
    "train_temporal.py",
    "evaluate_temporal.py",
    "inference_temporal.py",
    "train_temporal_split.py",
    "evaluate_temporal_split.py",
    "diagnose_temporal_split.py",
    "inspect_alignment.py",
    "diagnose_temporal_offset.py",
    "compare_label_mapping_strategies.py",
    "train_temporal_split_expanded.py",
    "evaluate_temporal_split_expanded.py",
    "train_compare_temporal_models.py",
]


EXPERIMENTAL_CHECKPOINTS = [
    "anomaly_net_temporal_weights.pth",
    "anomaly_net_temporal_split_weights.pth",
    "anomaly_net_temporal_split_expanded_weights.pth",
    "temporal_model_two_stage_tcn_refiner.pth",
    "temporal_model_direct_tcn.pth",
    "temporal_model_small_transformer.pth",
]


EXPERIMENTAL_OUTPUTS = [
    "outputs/inference_scores.png",
    "outputs/roc_curve.png",
    "outputs/score_histogram.png",
    "outputs/video_level_roc_curve.png",
    "outputs/temporal_finetuned_roc_curve.png",
    "outputs/temporal_finetuned_score_histogram.png",
    "outputs/temporal_finetuned_video_level_roc_curve.png",
    "outputs/temporal_inference_scores.png",
    "outputs/temporal_split_eval_roc_curve.png",
    "outputs/temporal_split_eval_score_histogram.png",
    "outputs/temporal_split_eval_video_level_roc_curve.png",
    "outputs/temporal_split_diagnostics.csv",
    "outputs/temporal_split_diagnostics",
    "outputs/alignment_inspection.csv",
    "outputs/alignment_inspection",
    "outputs/temporal_offset_diagnostics.csv",
    "outputs/temporal_offset_diagnostics.png",
    "outputs/temporal_offset_overlap.png",
    "outputs/label_mapping_strategy_comparison.csv",
    "outputs/label_mapping_strategy_per_video.csv",
    "outputs/label_mapping_strategy_comparison.png",
    "outputs/label_mapping_peak_overlap.png",
    "outputs/temporal_split_expanded_eval_roc_curve.png",
    "outputs/temporal_split_expanded_eval_score_histogram.png",
    "outputs/temporal_split_expanded_eval_video_level_roc_curve.png",
    "outputs/temporal_split_expanded_eval_per_video.csv",
    "outputs/temporal_model_comparison_summary.csv",
    "outputs/temporal_model_comparison_per_video.csv",
    "outputs/temporal_model_comparison_temporal_auc.png",
    "outputs/temporal_model_comparison_video_auc.png",
    "outputs/temporal_model_comparison_peak_overlap.png",
    "outputs/temporal_model_comparison_distance.png",
]


# The finalized two-stage evaluator depends on this exact held-out split file.
# It is kept so the final pipeline can still be re-run after cleanup.
PRESERVED_FINAL_DEPENDENCIES = [
    "outputs/temporal_split.json",
]


def remove_path(path):
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    removed = []
    skipped = []

    for raw_path in EXPERIMENTAL_SCRIPTS + EXPERIMENTAL_CHECKPOINTS + EXPERIMENTAL_OUTPUTS:
        path = Path(raw_path)
        if remove_path(path):
            removed.append(raw_path)

    for raw_path in PRESERVED_FINAL_DEPENDENCIES:
        if Path(raw_path).exists():
            skipped.append(f"{raw_path}  # kept because evaluate_two_stage_pipeline.py requires it")

    with MANIFEST_PATH.open("w") as f:
        f.write("Removed files and folders\n")
        f.write("=========================\n")
        for item in removed:
            f.write(f"{item}\n")
        f.write("\nPreserved final pipeline dependencies\n")
        f.write("=====================================\n")
        for item in skipped:
            f.write(f"{item}\n")

    print(f"Cleanup completed. Removed {len(removed)} files/folders.")
    print(f"Manifest written to: {MANIFEST_PATH}")
    if skipped:
        print("Preserved required final pipeline dependencies:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
