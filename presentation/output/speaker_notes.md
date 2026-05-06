# Drift-Aware MLOps ACS Presentation - Speaker Notes

Slide 1: Introduce the project as both a research study and an end-to-end MLOps implementation.
Slide 2: Explain the problem: deployed ML models decay under dataset shift, and retraining must be justified.
Slide 3: Walk through the system components: data, training, drift policy, API, MLflow, Prometheus/Grafana, CI/CD.
Slide 4: Emphasize the methodological correction: label flips created noise; final experiments use learnable drift regimes.
Slide 5: Present the main result: policy retraining improves final batch accuracy/F1 while avoiding unnecessary retrains.
Slide 6: Explain the ablation result: retraining fails under synthetic label noise but helps under learnable shift.
Slide 7: Show the operational stack and close with the thesis: retraining is a signal-quality decision, not a magic button.

Suggested split: Taha covers slides 1, 2, 5, 7; Hamza covers slide 4 and experiment logic; Hasnat covers slide 3 and slide 6 or the live MLOps demo.