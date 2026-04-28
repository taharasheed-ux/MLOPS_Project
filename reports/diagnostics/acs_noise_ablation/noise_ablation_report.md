# ACS Noise Ablation: When Does Retraining Help?

This diagnostic separates clean temporal shift, covariate/feature shift, subgroup label flips, and the full mixed ACS drift schedule.

## Summary Table

| drift_mode | strategy | anchor_fraction | window_batches | mean_accuracy | mean_f1 | worst_f1 | max_f1_drop_from_batch1 | degradation_area_f1 | post_retrain_avg_f1 | mean_recovery_gain_next_batch_f1 | retrain_count | concept_drift_batches | drift_alert_batches | total_retrain_time | total_inference_time | mean_f1_gain_vs_static |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_temporal | Static | 0.0000 | 3 | 0.7971 | 0.7248 | 0.7061 | 0.0443 | 0.3072 | 0.0000 | 0.0000 | 0 | 0 | 11 | 0.0000 | 1.6482 | 0.0000 |
| clean_temporal | Policy-CurrentAnchor | 0.1000 | 3 | 0.8066 | 0.7274 | 0.7106 | 0.0399 | 0.2763 | 0.7252 | 0.0002 | 5 | 0 | 11 | 15.6817 | 0.6210 | 0.0026 |
| clean_temporal | Policy-SlidingNoAnchor | 0.0000 | 3 | 0.7962 | 0.7180 | 0.7010 | 0.0494 | 0.3895 | 0.7114 | 0.0051 | 5 | 0 | 11 | 8.4954 | 0.5648 | -0.0069 |
| covariate_feature_shift | Static | 0.0000 | 3 | 0.7667 | 0.7155 | 0.6535 | 0.0969 | 0.4246 | 0.0000 | 0.0000 | 0 | 1 | 11 | 0.0000 | 1.3295 | 0.0000 |
| covariate_feature_shift | Policy-CurrentAnchor | 0.1000 | 3 | 0.7881 | 0.7236 | 0.6535 | 0.0969 | 0.3230 | 0.7126 | 0.0123 | 5 | 1 | 11 | 14.7631 | 0.5597 | 0.0081 |
| covariate_feature_shift | Policy-SlidingNoAnchor | 0.0000 | 3 | 0.7826 | 0.7054 | 0.6418 | 0.1086 | 0.5405 | 0.6985 | -0.0033 | 5 | 2 | 11 | 4.4218 | 0.3348 | -0.0102 |
| label_flip_only | Static | 0.0000 | 3 | 0.7725 | 0.6906 | 0.6413 | 0.1091 | 0.7176 | 0.0000 | 0.0000 | 0 | 5 | 11 | 0.0000 | 0.5840 | 0.0000 |
| label_flip_only | Policy-CurrentAnchor | 0.1000 | 3 | 0.7804 | 0.6874 | 0.6322 | 0.1182 | 0.7563 | 0.6932 | -0.0319 | 5 | 6 | 10 | 8.6915 | 0.3911 | -0.0032 |
| label_flip_only | Policy-SlidingNoAnchor | 0.0000 | 3 | 0.7716 | 0.6817 | 0.6315 | 0.1189 | 0.8244 | 0.6865 | -0.0322 | 5 | 2 | 11 | 4.4005 | 0.3526 | -0.0089 |
| full_with_label_flips | Static | 0.0000 | 3 | 0.7327 | 0.6678 | 0.5842 | 0.1662 | 0.9918 | 0.0000 | 0.0000 | 0 | 8 | 11 | 0.0000 | 0.5732 | 0.0000 |
| full_with_label_flips | Policy-CurrentAnchor | 0.1000 | 3 | 0.7477 | 0.6651 | 0.5551 | 0.1953 | 1.0243 | 0.6656 | -0.0343 | 5 | 7 | 10 | 8.8034 | 0.4351 | -0.0027 |
| full_with_label_flips | Policy-SlidingNoAnchor | 0.0000 | 3 | 0.7406 | 0.6432 | 0.5387 | 0.2117 | 1.2862 | 0.6436 | -0.0425 | 5 | 4 | 11 | 4.5088 | 0.3563 | -0.0245 |

## Interpretation Guide

- If policy retraining improves over Static on clean temporal or covariate-only streams, retraining is useful when drift is learnable.
- If policy retraining fails or degrades under label-flip streams, the synthetic concept drift is behaving more like contradictory label noise than a stable new rule.
- If Policy-SlidingNoAnchor beats Policy-CurrentAnchor, the historical anchor is too large for online adaptation.
- If both policy variants lose to Static, the retraining trigger/data policy is not yet aligned with the stream.

## clean_temporal

Best strategy: **Policy-CurrentAnchor** with mean F1 `0.7274`.
Static mean F1 was `0.7248`.

## covariate_feature_shift

Best strategy: **Policy-CurrentAnchor** with mean F1 `0.7236`.
Static mean F1 was `0.7155`.

## label_flip_only

Best strategy: **Static** with mean F1 `0.6906`.
Static mean F1 was `0.6906`.

## full_with_label_flips

Best strategy: **Static** with mean F1 `0.6678`.
Static mean F1 was `0.6678`.
