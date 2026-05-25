# Inception-I3D Feature Extractor

This folder contains an isolated RGB Inception-I3D model path for raw-video feature extraction.

The implementation follows the public PyTorch Inception-I3D architecture style from `piergiaj/pytorch-i3d`, which is a PyTorch port of DeepMind's Kinetics-I3D model.

References:

- https://github.com/piergiaj/pytorch-i3d
- https://github.com/google-deepmind/kinetics-i3d

## Required Weights

Real pretrained RGB I3D Kinetics weights are required. The extractor refuses to run without a valid `--weights` file and does not generate random, fake, or S3D fallback features.

Recommended weight type:

- RGB Inception-I3D pretrained on ImageNet + Kinetics, usually distributed as a PyTorch `.pt`/`.pth` state dict such as `rgb_imagenet.pt`.

Keep weights out of source control unless project policy explicitly allows storing them.

## Output Contract

The real extractor is expected to write:

```text
(T, 1024)
```

Where `T` is the number of temporal clips extracted from the input video, and each row is one RGB I3D feature vector after global average pooling of the final mixed feature block.

Use a separate output directory for this I3D path, for example:

```text
outputs/video_detection_i3d/
```

Do not overwrite S3D outputs in:

```text
outputs/video_detection/
```
