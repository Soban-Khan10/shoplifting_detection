# Shoplifting Anomaly Detection Results Summary

## Dataset Setup

- Shoplifting anomaly test videos from the UCF Crime feature set
- Normal test videos from the UCF Crime feature set
- 10-crop feature files grouped by base video name before evaluation
- 1024-dimensional pre-extracted features
- 32 temporal segments per video sample

## Training Method

- Multiple instance learning ranking loss
- Video-level weak supervision

## Evaluation Results

- Frame/temporal-level AUC: 0.544123
- Video-level AUC: 0.866213
- Video-level best threshold: 0.052090

## Interpretation

The model performs better at video-level anomaly detection than exact temporal localization. This is expected because training uses weak video-level labels rather than precise temporal supervision.

## Limitations

- Pre-extracted features only
- No raw video feature extractor yet
- Temporal annotations had to be projected to the feature temporal axis

## Future Work

- Add an I3D/C3D feature extractor for raw video
- Improve temporal localization
- Test on real CCTV/shop footage
