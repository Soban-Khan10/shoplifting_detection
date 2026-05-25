# Raw Video Pipeline Status

## Current Official Path

The raw-video path is now:

```text
mp4 -> I3D 1024-d features -> two-stage inference -> suspicious timestamp/clip/frame
```

`run_video_detection.py` now uses:

```text
extract_i3d_features_from_video_real.py
```

Default output directory:

```text
outputs/video_detection_i3d/
```

Default I3D weights:

```text
weights/i3d/rgb_imagenet.pt
```

## Why S3D Was Rejected

S3D was a temporary raw-video bridge because it produced shape-compatible `(T, 1024)` features. It did not match the feature distribution expected by the anomaly models.

Observed result on `Shoplifting001_x264.mp4`:

- S3D Stage 1 max score was near `0`
- S3D alert was `False`

That behavior was inconsistent with the known shoplifting content, so S3D is no longer the official raw-video extractor.

## I3D Validation

RGB Inception-I3D with Kinetics pretrained weights is now used for raw video.

Validated on `Shoplifting001_x264.mp4`:

- feature shape: `(134, 1024)`
- Stage 1 max score: `0.9999210834503174`
- alert: `True`
- suspicious segment index: `31`
- suspicious score: `0.9663602709770203`

## Weights And Outputs

The I3D weights are stored locally at:

```text
weights/i3d/rgb_imagenet.pt
```

The weights are ignored by git via `*.pt` and should not be committed.

Generated outputs are ignored by git and can be recreated from the source video, weights, and scripts.

## S3D Files

`extract_s3d_features_from_video.py` and existing `outputs/video_detection/` artifacts were left in place because the current project rules say not to delete anything. They should be treated as deprecated bridge artifacts unless explicitly needed for comparison.
