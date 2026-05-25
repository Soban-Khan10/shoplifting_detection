# Suspicious Clip Extraction Plan

## Scope

Add raw-video-only post-processing after `run_video_detection.py` completes S3D feature extraction and two-stage inference. Keep the existing `.npy` feature inference path intact and avoid changing model behavior.

## Files Needing Modification

- `run_video_detection.py`
  - Read the inference JSON produced by `final_two_stage_inference.py`.
  - Read the S3D metadata JSON produced by `extract_s3d_features_from_video.py`.
  - If `alert` is true and `suspicious_segment_index` is present, compute timestamps and save the clip/frame.
  - Update the result JSON with `timestamp_start`, `timestamp_end`, `clip_path`, and `key_frame_path`.
- New helper module, likely `suspicious_clip_utils.py`
  - Keep timestamp mapping, OpenCV clip extraction, key-frame extraction, and JSON update logic separate from the model pipeline.
  - This avoids rewriting `run_video_detection.py` and keeps the new stage testable.
- `final_two_stage_inference.py`
  - No planned model or pipeline changes.
  - Only modify if needed to make the output JSON path easier for `run_video_detection.py` to locate, but the current deterministic path can already be derived from the feature path stem.
- Optional focused test file or smoke script
  - Add only if practical with a tiny synthetic video, because real checkpoint/video tests are heavier.

## Segment Index To Timestamp Mapping

`final_two_stage_inference.py` always runs the models on `N_SEGMENTS = 32` temporal bins. For an alert, Stage 2 returns `suspicious_segment_index` in the range `0..31`.

The S3D feature extractor creates one feature per raw clip:

- clip `i` starts at source frame `i * stride`
- clip `i` ends around source frame `i * stride + clip_len - 1`
- timestamp seconds are frame index divided by `original_fps`

The model segment should first be mapped from the 32 model bins back to the extracted S3D feature index range:

```text
feature_start = floor(segment_idx / 32 * num_features)
feature_end = ceil((segment_idx + 1) / 32 * num_features)
```

Then map that feature window to raw video frames:

```text
frame_start = feature_start * stride
frame_end = min((feature_end - 1) * stride + clip_len, total_frames)
timestamp_start = frame_start / original_fps
timestamp_end = frame_end / original_fps
```

Clamp all indices to valid ranges. Ensure `timestamp_end > timestamp_start`; if the computed range collapses, expand to at least one clip length or a small minimum duration within the video bounds.

## Metadata Needed From S3D Extraction

The existing S3D metadata sidecar already contains the required fields:

- `video_path`
- `original_fps`
- `total_frames`
- `clip_len`
- `stride`
- `number_of_clips_features`
- `output_feature_path`

Use `feature_path.with_suffix(".metadata.json")` to locate it. Validate that `number_of_clips_features` matches the feature array length or the inference JSON `original_feature_lengths` when available.

## Saving Suspicious Clip And Key Frame

Use OpenCV in the helper module:

- Open the original video from metadata `video_path`.
- Seek to `frame_start`.
- Write frames through `frame_end` to a new `.mp4` file under the selected output directory, for example:
  - `<output-dir>/<video_stem>_suspicious_clip.mp4`
- Preserve source FPS and frame size from `cv2.VideoCapture`.
- Save a key suspicious frame near the center of the timestamp window:
  - `key_frame = floor((frame_start + frame_end) / 2)`
  - `<output-dir>/<video_stem>_suspicious_key_frame.jpg`
- Update the inference JSON in place with:
  - `timestamp_start`
  - `timestamp_end`
  - `clip_path`
  - `key_frame_path`

If `alert` is false, leave clip/frame paths as `null` or omit them and keep the current no-localization message.

## Risks And Limitations

- Stage 2 returns a 32-bin model segment, not a precise frame-level boundary.
- S3D features are generated from overlapping clips, so the mapped timestamp range is approximate.
- `final_two_stage_inference.py` may resample the S3D feature sequence to 32 bins using integer linspace; this loses exact one-to-one timing detail.
- The timestamp may be less accurate for very short videos or videos where `num_features < 32`.
- OpenCV-reported FPS and frame count can be imperfect for variable-frame-rate videos.
- The extracted suspicious clip should be presented as an approximate review window, not as a definitive event boundary.

## Exact Next Implementation Steps

1. Add a helper module for metadata loading, segment-to-time conversion, clip extraction, key-frame extraction, and JSON update.
2. Extend `run_video_detection.py` after successful inference to locate the result JSON and S3D metadata JSON.
3. When `alert` is true, compute timestamp/frame bounds from `suspicious_segment_index` and metadata.
4. Save the suspicious clip and key frame into the existing `--output-dir`.
5. Add the four requested fields to the JSON result: `timestamp_start`, `timestamp_end`, `clip_path`, `key_frame_path`.
6. Keep the existing feature extraction and `final_two_stage_inference.py` CLI unchanged.
7. Run `python -m py_compile run_video_detection.py final_two_stage_inference.py suspicious_clip_utils.py`.
8. If a small sample video is available, run a raw-video smoke test on CPU and verify that the JSON paths point to existing files when an alert is produced.
