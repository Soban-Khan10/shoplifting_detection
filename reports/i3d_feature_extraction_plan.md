# I3D Feature Extraction Plan

## Summary

True I3D feature extraction is not available locally in the current `shoplifting` conda environment. The repo has a placeholder `extract_i3d_features_from_video.py`, but it only samples frames and writes metadata; it does not run an I3D model or produce `(T, 1024)` features.

The safest next implementation should keep the existing S3D/raw-video pipeline unchanged and add a separate I3D path that writes only to:

```text
outputs/video_detection_i3d/
```

Do not reuse or overwrite:

```text
outputs/video_detection/
```

## Local Investigation Results

Checked inside the `shoplifting` conda environment.

- `pytorchvideo`: not installed.
- `torchvision`: installed, version `0.20.1`.
- imported `torch.__version__`: `2.5.1`.
- `python -m pip show torch`: reports `2.11.0`, so the environment has inconsistent Torch package metadata.
- `opencv-python`: installed, version `4.13.0.92`.
- local Torch model cache contains only:
  - `/Users/mac/.cache/torch/hub/checkpoints/s3d-d76dad2f.pth`

Torchvision emits a missing `libjpeg.9.dylib` warning on import, but the current video frame path uses OpenCV and still runs.

## Option 1: PyTorchVideo I3D

Local result: not available because `pytorchvideo` is not installed.

PyTorchVideo documentation/model zoo lists an I3D model, including an I3D R50 Kinetics entry. This would require installing or loading PyTorchVideo externally, and likely downloading model weights. It is a viable external path, but it is not currently local.

Important compatibility concern: PyTorchVideo's I3D R50 is a ResNet-style I3D variant, not necessarily the original Inception-v1 I3D feature extractor commonly used for UCF-Crime anomaly features. Its penultimate feature dimension may not naturally match the existing `(T, 1024)` downstream anomaly model input without an additional projection or a carefully chosen hook.

Sources:

- https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html
- https://pytorchvideo.readthedocs.io/en/latest/models.html

## Option 2: Torchvision I3D

Local result: not available.

Torchvision video models available locally include:

- `s3d`
- `r3d_18`
- `mc3_18`
- `r2plus1d_18`
- `mvit_v1_b`
- `mvit_v2_s`
- `swin3d_t`
- `swin3d_s`
- `swin3d_b`

There is no torchvision `i3d` model in the installed version. The existing S3D path is the closest local 1024-dimensional video feature extractor and already works, but it is not true I3D.

## Option 3: Vendor A Small I3D Definition And Use Kinetics Weights

This is the best route if the goal is true Inception-I3D-like `(T, 1024)` features compatible with the existing anomaly model expectations.

Recommended implementation source:

- `piergiaj/pytorch-i3d`
  - PyTorch port of DeepMind Kinetics-I3D.
  - Includes feature extraction code.
  - Mentions converted DeepMind pretrained RGB/Flow weights such as `rgb_imagenet.pt`.
  - Caveat: the repo was originally written for old PyTorch versions, so it may need compatibility fixes for current PyTorch.

Reference/original model source:

- `google-deepmind/kinetics-i3d`
  - Original TensorFlow I3D implementation.
  - Provides pretrained RGB and Flow checkpoints for Kinetics/ImageNet+Kinetics variants.
  - Not directly usable from the current PyTorch pipeline without TensorFlow or weight conversion.

Sources:

- https://github.com/piergiaj/pytorch-i3d
- https://github.com/google-deepmind/kinetics-i3d

Recommended model stream for this project:

- RGB-only I3D first.
- Avoid optical flow initially because it adds a second extraction dependency and more preprocessing risk.
- Extract a 1024-dimensional feature vector from the final mixed/logits-preclassifier feature level, then save as `(T, 1024)`.

## Option 4: External Package Required

An external package or vendored model code is required for true I3D.

Two practical choices:

1. Vendor a small PyTorch Inception-I3D implementation plus RGB Kinetics pretrained weights.
2. Install PyTorchVideo and use its I3D R50 model if a ResNet-style I3D variant is acceptable.

For this anomaly pipeline, vendoring an Inception-I3D implementation is preferable because the existing downstream models expect 1024-dimensional I3D-style features. PyTorchVideo is cleaner as a dependency, but its I3D variant may produce a different feature representation and shape.

## Pretrained Weights Availability

No true I3D pretrained weights are currently present locally.

Downloadable candidates:

- DeepMind TensorFlow Kinetics-I3D checkpoints from `google-deepmind/kinetics-i3d`.
- Converted PyTorch weights from `piergiaj/pytorch-i3d`, including RGB ImageNet+Kinetics style weights.
- PyTorchVideo model zoo weights for I3D R50, if using PyTorchVideo.

The implementation should not assume weights are present. It should fail with a clear error if the configured I3D weight file is missing, or provide an explicit download/setup command documented separately.

## Expected Output Shape

The target output should remain compatible with `final_two_stage_inference.py`:

```text
(T, 1024)
```

Where:

- `T` is the number of I3D clips/windows extracted from the source video.
- each row is one 1024-dimensional RGB I3D feature vector.
- `final_two_stage_inference.py` will resample `T` to its fixed 32 model segments as it already does.

Metadata should include:

- `video_path`
- `original_fps`
- `total_frames`
- `clip_len`
- `stride`
- `number_of_clips_features`
- `feature_dim`
- `extractor`: for example `inception_i3d_rgb`
- `weights_path`
- `preprocessing`
- `output_feature_path`

## Output Separation From S3D

Keep S3D outputs unchanged:

```text
outputs/video_detection/
```

Write all I3D outputs to:

```text
outputs/video_detection_i3d/
```

Expected I3D output names:

```text
outputs/video_detection_i3d/Shoplifting001_x264_i3d_features.npy
outputs/video_detection_i3d/Shoplifting001_x264_i3d_features.metadata.json
outputs/video_detection_i3d/Shoplifting001_x264_i3d_features_result.json
outputs/video_detection_i3d/Shoplifting001_x264_i3d_features_scores.png
outputs/video_detection_i3d/Shoplifting001_x264_suspicious_clip.mp4
outputs/video_detection_i3d/Shoplifting001_x264_suspicious_key_frame.jpg
```

## Implementation Status

Prepared an isolated I3D code path:

- `feature_extractors/i3d/__init__.py`
- `feature_extractors/i3d/README.md`
- `feature_extractors/i3d/i3d_model.py`
- `extract_i3d_features_from_video_real.py`

The new extractor is intentionally strict:

- real pretrained RGB Inception-I3D Kinetics weights are still required via `--weights`
- missing weights fail clearly before extraction
- random weights are not allowed
- fake features are not generated
- S3D fallback is not used
- S3D outputs in `outputs/video_detection/` are not overwritten

Once a compatible RGB I3D weights file is available, test with:

```text
python extract_i3d_features_from_video_real.py \
  --video ~/Downloads/Shoplifting001_x264.mp4 \
  --output outputs/video_detection_i3d/Shoplifting001_x264_i3d_features.npy \
  --weights /path/to/rgb_i3d_kinetics_weights.pt \
  --device cpu
```

Then run:

```text
python final_two_stage_inference.py \
  --input outputs/video_detection_i3d/Shoplifting001_x264_i3d_features.npy \
  --output-dir outputs/video_detection_i3d \
  --device cpu
```

The current implementation is ready to load real I3D RGB Kinetics weights once provided, but feature extraction should not be tested until that weights file exists locally.

## Exact Implementation Steps

1. Keep `run_video_detection.py` unchanged as the S3D pipeline.
2. Add a separate raw-video I3D runner, likely `run_video_detection_i3d.py`, with default `--output-dir outputs/video_detection_i3d`.
3. Replace the placeholder behavior in `extract_i3d_features_from_video.py` with real RGB I3D extraction, or add a new `extract_true_i3d_features_from_video.py` if preserving the placeholder is useful.
4. Vendor a small Inception-I3D PyTorch model definition under a clearly named module such as `models/i3d/`.
5. Add a `--weights` argument pointing to a local RGB I3D Kinetics `.pt` weight file.
6. Implement OpenCV frame loading, RGB conversion, resize/crop, normalization, clip creation, and stride-based temporal windows.
7. Run the I3D model on CPU by default, extract 1024-dimensional features, and save `(T, 1024)` `.npy`.
8. Save metadata with timing fields matching the S3D metadata contract so suspicious timestamp extraction can be reused.
9. Run `final_two_stage_inference.py` on the I3D feature file, writing results to `outputs/video_detection_i3d/`.
10. Reuse the suspicious timestamp/clip/key-frame logic for I3D results, but ensure paths remain under `outputs/video_detection_i3d/`.
11. Test on `~/Downloads/Shoplifting001_x264.mp4` with CPU.
12. Compare output score behavior against the existing S3D run, but do not overwrite S3D artifacts.

## Risks And Limitations

- The current environment does not include true I3D code or I3D weights.
- Vendored I3D implementations may require compatibility updates for current PyTorch.
- Different I3D implementations expose different feature hook points; extracting the wrong tensor can break the expected `(T, 1024)` contract.
- RGB-only I3D is simpler but may not match two-stream RGB+Flow features if the original UCF-Crime feature set used flow.
- Preprocessing differences can shift anomaly scores significantly.
- The finalized Stage 1 threshold may not transfer cleanly to a new raw-video extractor.
- CPU extraction on full UCF-Crime videos will be slow.
- The environment's Torch package metadata mismatch should be treated carefully before adding new dependencies.
