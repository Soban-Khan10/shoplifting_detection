# Top-K Localization Update

The Stage 2 localization model can produce a single highest-scoring segment that is temporally wrong, even when Stage 1 correctly detects that the video is suspicious. This was observed on `Shoplifting001_x264.mp4`, where the top segment is near the end of the video while the ground-truth anomaly is earlier.

The pipeline now reports top-K suspicious Stage 2 segments in addition to the existing single peak:

- `suspicious_segment_index` remains the highest-scoring segment for backward compatibility.
- `top_suspicious_segments` records ranked segment indices, scores, and approximate feature windows.
- raw-video results add `top_suspicious_time_windows` with approximate timestamps for each ranked segment.

This does not fix the trained model's localization behavior or make the boundaries frame-accurate. It gives demo users and debugging tools more suspicious candidates to inspect, which is more useful than showing only one potentially wrong peak.

Ground truth is not used in production inference. Annotation overlap should remain part of separate debug scripts only.
