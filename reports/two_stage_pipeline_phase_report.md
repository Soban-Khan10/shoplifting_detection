# Retail Shoplifting Anomaly Detection System: Two-Stage Temporal Localization Phase Report

## 1. Project Goal

This project is a retail shoplifting anomaly detection system built using pre-extracted UCF Crime `.npy` feature files. The original goal was video-level anomaly detection: decide whether a video is suspicious or normal. During this phase, the goal evolved toward temporal suspicious-window localization: after detecting a suspicious video, identify the most suspicious temporal segment.

The current system is still feature-based. It does not yet perform raw-video frame extraction, does not connect to CCTV, and does not perform real face recognition. Later person matching should be treated as a simulated demo only, not as a real identity recognition system.

## 2. Original Pipeline

Original flow:

```text
.npy feature file -> AnomalyNet -> 32 anomaly scores
```

The original `AnomalyNet` uses:

| Layer | Details |
|---|---|
| Input | `input_dim=1024` |
| Linear | 1024 to 512 |
| Activation | ReLU |
| Dropout | 0.6 |
| Linear | 512 to 32 |
| Activation | ReLU |
| Dropout | 0.6 |
| Linear | 32 to 1 |
| Output | Sigmoid |

The output is 32 segment-level anomaly scores.

Original results:

| Metric | Value |
|---|---:|
| Temporal/frame-level AUC | 0.544123 |
| Video-level AUC | 0.866213 |

Interpretation: the original MIL-style model was useful for video-level detection, but weak for exact temporal localization. The training objective taught the model that anomaly videos should score high somewhere, but it did not strongly teach the exact anomalous segment position.

## 3. First Temporal Fine-Tuning Experiment

The first temporal fine-tuning experiment used temporal annotations from `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`. Frame intervals were converted into 32 segment labels. The model was fine-tuned using binary cross entropy loss plus a smoothness loss.

This experiment used anomaly and normal test feature files because temporal annotations were available for those videos.

Optimistic result:

| Metric | Value |
|---|---:|
| Temporal AUC | 0.851954 |
| Video AUC | 0.950113 |

This result was not fully honest because the model was fine-tuned and evaluated on the same annotated videos. It was useful as proof that the model can learn localization, but it was not reliable as a final generalization score.

## 4. Honest Split Evaluation

To avoid crop leakage, a base-video split was created. All crops from the same base video were kept in the same split.

| Split | Anomaly Base Videos | Normal Base Videos |
|---|---:|---:|
| Train | 14 | 14 |
| Eval | 7 | 7 |

Strict-label honest split result:

| Metric | Value |
|---|---:|
| Temporal AUC | 0.665318 |
| Video AUC | 0.938776 |

Video detection stayed strong, but temporal localization was still not good enough. Diagnostics showed that the model often detected suspicious videos but localized the wrong segment.

## 5. Diagnostics and Failed/Weak Experiments

Several diagnostic phases were created:

- `diagnose_temporal_split.py`
- `inspect_alignment.py`
- `diagnose_temporal_offset.py`
- `compare_label_mapping_strategies.py`
- expanded-label training

Main findings:

- Strict peak overlap was poor.
- Model peaks often did not match projected annotation segments.
- Average distance to the nearest positive segment was high during alignment inspection.
- Label expansion only slightly improved relaxed localization.

Expanded-label checkpoint result:

| Metric | Value |
|---|---:|
| Temporal AUC | 0.663768 |
| Video AUC | 0.938776 |
| Strict peak overlap ratio | 0.000000 |
| Expanded-label peak overlap ratio | 0.142857 |

Conclusion: label expansion alone was not enough. A larger pipeline change was needed.

## 6. Stronger Temporal Model Comparison

Four stronger temporal models were compared honestly on the same held-out split:

- `two_stage_tcn_refiner`
- `direct_tcn`
- `direct_bilstm`
- `small_transformer`

Results:

| Model | Temporal AUC | Video AUC | Peak Overlap | Avg Distance |
|---|---:|---:|---:|---:|
| direct_bilstm | 0.893811 | 0.775510 | 1.000000 | 0.000000 |
| small_transformer | 0.876686 | 0.693878 | 1.000000 | 0.000000 |
| two_stage_tcn_refiner | 0.795038 | 0.755102 | 1.000000 | 0.000000 |
| direct_tcn | 0.789644 | 0.755102 | 1.000000 | 0.000000 |

The Direct BiLSTM gave the best temporal localization. However, its video-level AUC was weaker than the original AnomalyNet. This suggested using different models for different jobs.

## 7. Finalized Two-Stage Pipeline

The finalized pipeline uses two models:

Stage 1: AnomalyNet for video-level alert.

Stage 2: Direct BiLSTM for temporal localization after Stage 1 alert.

Pipeline:

```text
.npy feature crops
-> group by base video
-> Stage 1 AnomalyNet scores each crop
-> average scores across crops
-> max score determines video alert
-> if alert=True, Stage 2 BiLSTM localizes suspicious segment
-> return most suspicious segment/window
```

This was chosen because AnomalyNet is strong for video-level detection, while BiLSTM is strong for temporal localization. Combining them gives the best match for the project goal: alert plus suspicious segment/window.

## 8. Final Two-Stage Evaluation Results

The final two-stage pipeline was evaluated only on the held-out eval split:

- 7 anomaly base videos
- 7 normal base videos
- no train videos used in evaluation

Stage 1 results:

| Metric | Value |
|---|---:|
| Video AUC | 0.979592 |
| Best threshold | 0.052090 |
| TP | 7 |
| TN | 6 |
| FP | 1 |
| FN | 0 |
| Accuracy | 0.928571 |
| Precision | 0.875000 |
| Recall | 1.000000 |
| F1 | 0.933333 |

Stage 2 results:

| Metric | Value |
|---|---:|
| Temporal AUC | 0.893811 |
| Peak overlap ratio | 1.000000 |
| Average distance to positive | 0.000000 |

Full pipeline results:

| Metric | Value |
|---|---:|
| Pipeline accuracy | 0.928571 |
| Anomaly localization success rate | 1.000000 |
| False alarm count on normal videos | 1 |
| Videos passed to Stage 2 | 8 |
| Percent videos passed to Stage 2 | 0.571429 |

## 9. Final Interpretation

The finalized pipeline is better aligned with the project goal than using one model for everything. It now has strong video-level alerting and strong suspicious-window localization on the held-out split.

The result should still be described carefully. The system is feature-based, not raw-video based. The held-out evaluation set is small, so the result is promising but not production-level proof.

## 10. Current Final Files Kept

Important files kept:

- `models/anomaly_net.py`
- `train.py`
- `evaluate.py`
- `inference.py`
- `evaluate_two_stage_pipeline.py`
- `anomaly_net_weights.pth`
- `temporal_model_direct_bilstm.pth`
- `outputs/two_stage_pipeline_summary.csv`
- `outputs/two_stage_pipeline_per_video.csv`
- `outputs/two_stage_stage1_video_roc.png`
- `outputs/two_stage_stage2_temporal_roc.png`
- `outputs/two_stage_stage1_score_histogram.png`
- `outputs/two_stage_stage2_score_histogram.png`
- `outputs/two_stage_pipeline_per_video_scores/`

## 11. Next Phase

The next recommended phase is to build a clean inference script for the finalized pipeline. Given one `.npy` feature file or grouped crops, it should return:

- video alert
- Stage 1 score
- suspicious segment index
- approximate feature window

Later phases:

- add raw video feature extraction
- map suspicious feature segment back to video frame range
- show suspicious clip/window in a dashboard
- simulate fake person metadata matching as a demo only
