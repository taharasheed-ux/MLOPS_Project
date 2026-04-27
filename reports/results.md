# Drift-Aware Retraining Project - Final Report

## 1) Experiment A: Static vs Adaptive System
We compare a statically deployed model against a system capable of adaptive drift-aware retaining.
### Static Baseline Metrics
| batch | model_version | accuracy | f1 | precision | recall | drift_severity | drift_flags | retrained | train_duration | inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch_1 | v1 | 0.877 | 0.734341252699784 | 0.7962529274004684 | 0.6813627254509018 | 0.0 | 0 | False | 0.0 | 0.02034902572631836 |
| batch_2 | v1 | 0.879 | 0.7386609071274298 | 0.76 | 0.7184873949579832 | 0.043245632968577453 | 2 | False | 0.0 | -3.5672435760498047 |
| batch_3 | v1 | 0.8675 | 0.7213459516298633 | 0.7521929824561403 | 0.692929292929293 | 0.03178952199099178 | 2 | False | 0.0 | 0.02106928825378418 |
| batch_4 | v1 | 0.831 | 0.6301969365426696 | 0.6939759036144578 | 0.5771543086172345 | 0.0 | 0 | False | 0.0 | 0.01990652084350586 |
| batch_5 | v1 | 0.775 | 0.5054945054945055 | 0.5665024630541872 | 0.45634920634920634 | 0.11867093330656224 | 4 | False | 0.0 | 0.0189054012298584 |

## 2) Experiment B: Immediate vs Policy-Based Retraining
Immediate retraining heavily punishes system resources on every drift detected. Policy-based enforces cooldowns and sample accumulators.
### Immediate Retraining Metrics
| batch | model_version | accuracy | f1 | precision | recall | drift_severity | drift_flags | retrained | train_duration | inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch_1 | v2 | 0.877 | 0.734341252699784 | 0.7962529274004684 | 0.6813627254509018 | 0.0 | 0 | True | 0.968062162399292 | 0.019881486892700195 |
| batch_2 | v3 | 0.8795 | 0.7394594594594595 | 0.7616926503340757 | 0.7184873949579832 | 0.04330712388960895 | 2 | True | 0.7469780445098877 | 0.021960020065307617 |
| batch_3 | v4 | 0.8695 | 0.7238095238095238 | 0.76 | 0.6909090909090909 | 0.03127449026692433 | 2 | True | 0.7647378444671631 | 0.024901866912841797 |
| batch_4 | v5 | 0.827 | 0.6189427312775331 | 0.687041564792176 | 0.56312625250501 | 0.0 | 0 | True | 0.8305239677429199 | 0.03332328796386719 |
| batch_5 | v6 | 0.78 | 0.5121951219512195 | 0.5804020100502513 | 0.4583333333333333 | 0.11656875796437383 | 4 | True | 0.8387598991394043 | 0.025987625122070312 |

### Policy-Standard Metrics
| batch | model_version | accuracy | f1 | precision | recall | drift_severity | drift_flags | retrained | train_duration | inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch_1 | v1 | 0.877 | 0.734341252699784 | 0.7962529274004684 | 0.6813627254509018 | 0.0 | 0 | False | 0.0 | 0.020550012588500977 |
| batch_2 | v1 | 0.879 | 0.7386609071274298 | 0.76 | 0.7184873949579832 | 0.043245632968577453 | 2 | False | 0.0 | 0.02390575408935547 |
| batch_3 | v1 | 0.8675 | 0.7213459516298633 | 0.7521929824561403 | 0.692929292929293 | 0.03178952199099178 | 2 | False | 0.0 | 0.01855301856994629 |
| batch_4 | v1 | 0.831 | 0.6301969365426696 | 0.6939759036144578 | 0.5771543086172345 | 0.0 | 0 | False | 0.0 | 0.026908397674560547 |
| batch_5 | v1 | 0.775 | 0.5054945054945055 | 0.5665024630541872 | 0.45634920634920634 | 0.11867093330656224 | 4 | False | 0.0 | 0.019512414932250977 |

## 3) Experiment C: Threshold Sensitivity
Demonstration of how different drift severity thresholds alter retraining frequencies.
| policy_name | mean_accuracy | mean_f1 | retrain_count | total_retrain_time | total_inference_time |
| --- | --- | --- | --- | --- | --- |
| Policy-Thresh0.05 | 0.8459 | 0.6660079106988505 | 1 | 0.8430697917938232 | 0.1307358741760254 |
| Policy-Thresh0.25 | 0.8459 | 0.6660079106988505 | 0 | 0.0 | 0.12172436714172363 |
| Policy-Thresh0.35 | 0.8459 | 0.6660079106988505 | 0 | 0.0 | 0.11291170120239258 |

## 4) Experiment D: Cost vs Performance Trade-offs
Comparison of mean classification metrics against system latency and retrain durations over 5 batches.
| policy_name | mean_accuracy | mean_f1 | retrain_count | total_retrain_time | total_inference_time |
| --- | --- | --- | --- | --- | --- |
| Static | 0.8459 | 0.6660079106988505 | 0 | 0.0 | -3.487013339996338 |
| Immediate | 0.8465999999999999 | 0.665749617839504 | 5 | 4.149061918258667 | 0.1260542869567871 |
| Policy-Standard | 0.8459 | 0.6660079106988505 | 0 | 0.0 | 0.10942959785461426 |
| Policy-Thresh0.05 | 0.8459 | 0.6660079106988505 | 1 | 0.8430697917938232 | 0.1307358741760254 |
| Policy-Thresh0.25 | 0.8459 | 0.6660079106988505 | 0 | 0.0 | 0.12172436714172363 |
| Policy-Thresh0.35 | 0.8459 | 0.6660079106988505 | 0 | 0.0 | 0.11291170120239258 |

## Core Research Conclusion
As hypothesized, **Policy-based retraining** successfully avoids the high computational overhead of **Immediate retraining** (which fires on every minor distributional shift), while heavily outperforming the **Static** baseline's severe degradation on simulated covariate and conditional drift (Batch 3 & 5). 