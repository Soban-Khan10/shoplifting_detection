# Raw Video I3D Extraction Plan

## Current environment

Checked inside the `shoplifting` conda environment.

- Python runtime: `3.10.20`
- Imported PyTorch runtime: `torch 2.5.1`
- `pip show torch` metadata reports `2.11.0`, so the environment has inconsistent Torch package metadata. The imported runtime is the value that matters for execution.
- Torchvision: `0.20.1`
- OpenCV: `4.13.0`
- PyTorchVideo: not installed
- `requirements.txt`: empty

Torchvision import works, but it prints an image extension warning because `libjpeg.9.dylib` is missing. The current frame scaffold uses OpenCV, not `torchvision.io`, so this warning should not block raw video frame loading.

## Available video models

Torchvision exposes these video model families:

- `s3d`
- `r3d_18`
- `mc3_18`
- `r2plus1d_18`
- `mvit_v1_b`
- `mvit_v2_s`
- `swin3d_t`
- `swin3d_s`
- `swin3d_b`

There is no local I3D implementation or I3D checkpoint in the repository. The only local model checkpoints found are:

- `anomaly_net_weights.pth`
- `temporal_model_direct_bilstm.pth`

Those are downstream anomaly/localization models, not raw-video feature extractors.

## Is true I3D extraction currently possible?

Not yet.

Strict I3D extraction is not currently available because the environment does not include a dedicated I3D implementation or local I3D pretrained weights. PyTorchVideo is also not installed.

Torchvision's `s3d` model is available and is the closest practical option already exposed by the environment. It has a 1024-channel representation immediately before the classifier:

- `S3D.classifier`: `Dropout` then `Conv3d(1024, 400, kernel_size=1)`
- Available weights enum: `S3D_Weights.KINETICS400_V1`

However, pretrained S3D weights are not confirmed to be present locally. Loading `S3D_Weights.KINETICS400_V1` may require downloading weights unless they already exist in the local Torch cache.

## Recommended extractor option

Recommended next option: use `torchvision.models.video.s3d` with `S3D_Weights.KINETICS400_V1`, then extract the 1024-channel pooled feature before the final classifier.

Reasons:

- It is available through the installed `torchvision`.
- Its pre-classifier channel dimension is 1024, matching the downstream pipeline's expected feature dimension.
- It avoids adding a larger dependency such as PyTorchVideo unless strict I3D compatibility is required.
- It is close to the existing I3D-style feature shape while keeping implementation and dependency risk lower.

If exact original I3D features are required, the safer path is to install or vendor the same I3D extractor and pretrained checkpoint used to generate the existing dataset features. Without that match, the downstream anomaly models may see a shifted feature distribution.

## Can output realistically be made `(T, 1024)`?

Yes, with S3D it is technically realistic to produce `(T, 1024)` features by:

1. Sampling and resizing video frames.
2. Splitting frames into fixed-length temporal clips.
3. Applying the S3D preprocessing transform expected by its weights.
4. Running clips through `model.features`.
5. Applying the model's average pooling or equivalent spatial/temporal pooling to produce one 1024-dimensional vector per clip.
6. Stacking clip vectors into a `(T, 1024)` array.

The exact meaning of `T` will depend on the chosen clip length and stride. That must be documented because the existing inference script later resamples variable-length features to 32 segments.

## Compatibility risks

- The existing anomaly models were trained on pre-extracted I3D-style features. If the original extractor was not torchvision S3D, scores may be poorly calibrated.
- Even with a 1024-dimensional output, feature semantics may differ across I3D, S3D, R3D, and other 3D CNNs.
- Preprocessing details matter: RGB order, normalization, resize/crop, clip length, temporal stride, frame rate, and pooling can all change feature distribution.
- The finalized Stage 1 threshold `0.052090` may not transfer to new raw-video features unless the extractor matches the training-time extractor.
- MPS support should be validated. CPU fallback is likely reliable but slower.
- The current Torch installation has a runtime/package metadata mismatch (`torch.__version__` vs `pip show torch`), which should be cleaned up before relying on new model downloads or production extraction.
- Torchvision emits a missing `libjpeg.9.dylib` warning. OpenCV decoding works for the current scaffold, but torchvision image/video I/O should be avoided unless the environment is fixed.

## Exact next implementation plan

1. Confirm whether pretrained S3D weights can load without a network download:
   `python -c "from torchvision.models.video import s3d, S3D_Weights; s3d(weights=S3D_Weights.KINETICS400_V1)"`
2. If weights are unavailable locally, decide whether to allow a one-time download or manually place the weight file in the Torch cache.
3. Add an optional extractor mode to `extract_i3d_features_from_video.py` without changing `final_two_stage_inference.py`.
4. Keep the current sampled-frame saving behavior as a debug/intermediate artifact.
5. Implement deterministic clip creation with explicit parameters:
   - sample FPS
   - clip length
   - clip stride
   - resize/crop size
   - device
6. Use the official `S3D_Weights.KINETICS400_V1.transforms()` preprocessing when possible.
7. Run S3D in eval mode and extract 1024-dimensional features before the classifier.
8. Save the real feature array to the user-provided `--output` path with shape `(T, 1024)`.
9. Extend metadata with extractor name, weights, clip length, clip stride, feature shape, device, and preprocessing details.
10. Validate output shape on `~/Downloads/sample-5s.mp4`.
11. Only after raw feature extraction is validated, run `final_two_stage_inference.py` manually on the generated feature file and treat scores as experimental until compatibility is benchmarked against known videos.
