# PI-DLNN reproducibility package

This package supports the manuscript **“Evaluating Physics-Informed Lagrangian Regularization for Non-Technical Loss Detection: A Leakage-Controlled Ten-Seed Benchmark with Utility-Provided and Synthetic Attack Labels.”**

## Contents

- `pi_dlnn_reproduction.py`: deterministic data preparation, model training, ablations, and seed-level export.
- `analyze_final_results.py`: paired tests, bootstrap confidence intervals, attack-specific analysis, and publication figures.
- `lambda_sensitivity_diagnostics.py`: ten-value λ_EL sensitivity runs, per-epoch physics-loss and shared-encoder gradient diagnostics, paired inference, and Figure 3.
- `experiment_outputs/run_metrics.csv`: complete 180-run metric table (2 datasets × 9 variants × 10 seeds).
- `experiment_outputs/{SGCC,UCI}/...`: raw test probabilities, split manifests, histories, model checkpoints, checkpoint hashes, and run metadata.
- `experiment_outputs/final_analysis/`: final summary tables, paired tests, attack-specific metrics, and figures.
- `lambda_sensitivity_outputs/lambda_seed_metrics.csv`: complete 200-run sensitivity metric and checkpoint-diagnostic table (2 datasets × 10 weights × 10 seeds).
- `lambda_sensitivity_outputs/{SGCC,UCI}/...`: sensitivity checkpoints, histories, raw test predictions, and run metadata.
- `lambda_sensitivity_outputs/lambda_summary.csv`, `lambda_paired_comparisons.csv`, and `global_validation_selected_lambda_*.csv`: aggregate and matched-seed sensitivity analyses.
- `lambda_sensitivity_outputs/lambda_sensitivity_gradient_diagnostics.png`: manuscript Figure 3.

## Public input data

The raw datasets are intentionally not duplicated in this archive. Obtain:

1. SGCC data from `https://github.com/henryRDlab/ElectricityTheftDetection`.
2. UCI Electricity Load Diagrams 2011–2014 from `https://doi.org/10.24432/C58C86`.

## Final protocol

- Seeds: 11, 23, 37, 53, 71, 89, 107, 131, 149, 173.
- Consumer/client-grouped 70/15/15 split.
- Train-only clipping and standardization.
- Validation-selected MCC threshold, locked before test evaluation.
- SGCC: 20 epochs maximum, patience 4.
- UCI: 40 epochs maximum, patience 8.
- Sensitivity weights: 0, 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 0.1, 1, 10, and 100.
- Gradient diagnostics: separate BCE and raw/weighted physics gradients on the shared LSTM encoder, evaluated each epoch on a deterministic subset of at most 2,048 training samples.
- Sensitivity selection: highest mean validation MCC across the ten prespecified seeds; test comparisons against λ_EL = 0 use paired bootstrap confidence intervals, two-sided Wilcoxon tests, and Holm adjustment within dataset and endpoint.

The broad high-weight extension was gradient-informed rather than preregistered. The original λ_EL = 10⁻⁴ setting is reproduced within the same fixed-seed protocol.

The archive preserves the completed results; rerunning requires the public raw files and the Python dependencies imported by the program.
