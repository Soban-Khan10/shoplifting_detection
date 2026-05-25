# I3D Timestamp Alignment Debug

The raw-video I3D pipeline produces suspicious timestamps by mapping the Stage 2 model's 32-segment index back onto the I3D clip timeline. This mapping is approximate because the model does not predict frame-level boundaries.

The debug script `debug_i3d_timestamp_alignment.py` compares:

- UCF-Crime ground-truth anomaly frame intervals
- ground-truth time intervals using the original FPS from I3D metadata
- model predicted suspicious segment index
- mapped suspicious timestamp interval
- overlap and distance between prediction and ground truth

Use it to validate whether the extracted suspicious clip/frame is temporally aligned with the dataset annotation before treating the clip as a correct localization.
