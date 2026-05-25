# Final Product Pipeline Summary

## Pipeline

```text
mp4 video
-> real I3D feature extraction
-> Stage 1 AnomalyNet alert
-> corrected Stage 2 BiLSTM localization
-> suspicious timestamp
-> suspicious clip
-> suspicious key frame
-> optional simulated person matching
```

The raw-video wrapper is `run_video_detection.py`. It uses `extract_i3d_features_from_video_real.py` for feature extraction, then `final_two_stage_inference.py` for localization, then the existing clip/key-frame extraction path, and optionally `simulate_person_match.py` for a demo-only profile match.

## Evaluation

Final two-stage evaluation:

- Stage 1 video AUC: `0.979592`
- Stage 1 accuracy: `0.928571`
- Stage 1 precision: `0.875000`
- Stage 1 recall: `1.000000`
- Stage 1 F1: `0.933333`
- Old Stage 2 temporal AUC: `0.893811`

Corrected Stage 2 metrics:

- temporal AUC: `0.924834`
- video AUC: `0.988290`
- peak overlap ratio: `0.476190`
- peak overlap count: `10/21`
- peak at bin 31 count: `0`
- late-bin 27-31 count: `0`

## Real Raw-Video Example

`Shoplifting001_x264.mp4` on the corrected I3D pipeline:

- alert: `True`
- predicted suspicious time: `57.6s-64.0s`
- ground truth: `51.667s-66.667s`

This is the example that confirms the corrected Stage 2 checkpoint fixes the late-bin bias seen in the old checkpoint.

## Assets

- I3D weights are stored locally at `weights/i3d/rgb_imagenet.pt` and are ignored by git.
- Generated demo outputs can be recreated from the scripts and source data.
- The old Stage 2 checkpoint is still available for comparison if needed.
- The corrected Stage 2 checkpoint is the one used by the final product wrapper when selected explicitly.

## Notes

- The simulated person matching stage is a demo-only placeholder and does not perform real identity recognition.
- The cleaned repository keeps the final product scripts, checkpoints, evaluation metrics, model code, and required reports while removing deprecated S3D and debug artifacts.
