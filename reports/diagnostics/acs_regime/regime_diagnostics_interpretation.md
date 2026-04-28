# ACS Temporal Regime Diagnostics

This diagnostic compares models trained on different ACS temporal regimes and evaluates each model on held-out 2018 data.

## Key Findings

- Training on 2017-2018 data improves 2018 F1 by `0.0239` over the 2016-only baseline.
- Adding a 10% stratified 2016 anchor changes F1 by only `0.0002` relative to 2017-2018-only training.
- The expanding-window model changes F1 by `-0.0021` relative to 2017-2018-only training and shows lower recall, suggesting mild historical dilution.
- Feature-importance L1 distance from the 2016-only model is `0.0703` for 2017-2018-only, `0.0778` for the anchored model, and `0.0748` for the expanding-window model.
- `SCHL` remains the top feature in every regime, so the observed temporal shift is learnable but not a complete rule reversal.

## Interpretation

The professor's sliding-window concern is directionally valid: later-year-only training performs best or effectively tied on F1, while the expanding window slightly reduces F1 and recall. However, the 10% anchor is not harmful in this full-data diagnostic because the 2017-2018 data dominates the training set. In smaller online retraining windows, the same anchor can still be too large relative to recent batches, so production retraining should cap the anchor by a fraction of recent samples or disable it for pure sliding-window experiments.
