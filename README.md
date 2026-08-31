# pi-dlnn-ntl-benchmark
Ten-seed reproducible benchmark testing physics-based Lagrangian regularization technique for non-technical loss detection using SGCC and UCI-ELD data sets.
PI-DLNN NTL Benchmark
This repository provides the code, experimental outputs, and analysis files supporting the manuscript:
> **Evaluating Physics-Informed Lagrangian Regularization for Non-Technical Loss Detection: A Leakage-Controlled Ten-Seed Benchmark with Utility-Provided and Synthetic Attack Labels**
The study evaluates whether Euler–Lagrange regularization improves non-technical loss detection under consumer-disjoint splitting, train-only preprocessing, validation-locked thresholds, ten fixed random seeds, matched ablations, and gradient-based sensitivity diagnostics.
Repository contents
`pi_dlnn_reproduction.py` — deterministic data preparation, model training, ablations, and seed-level export.
`analyze_final_results.py` — paired tests, bootstrap confidence intervals, attack-specific analysis, and publication figures.
`lambda_sensitivity_diagnostics.py` — ten-value λ_EL sensitivity analysis, per-epoch physics-loss and shared-encoder gradient diagnostics, paired inference, and Figure 3.
`experiment_outputs/run_metrics.csv` — complete 180-run metric table: 2 datasets × 9 variants × 10 seeds.
`experiment_outputs/{SGCC,UCI}/...` — raw test probabilities, split manifests, training histories, model checkpoints, checkpoint hashes, and run metadata.
`experiment_outputs/final_analysis/` — final summary tables, paired tests, attack-specific metrics, and publication figures.
`lambda_sensitivity_outputs/lambda_seed_metrics.csv` — complete 200-run sensitivity table: 2 datasets × 10 weights × 10 seeds.
`lambda_sensitivity_outputs/{SGCC,UCI}/...` — sensitivity checkpoints, training histories, raw test predictions, and run metadata.
`lambda_sensitivity_outputs/lambda_summary.csv` — aggregate sensitivity results.
`lambda_sensitivity_outputs/lambda_paired_comparisons.csv` — matched-seed sensitivity comparisons.
`lambda_sensitivity_outputs/global_validation_selected_lambda_*.csv` — validation-selected λ_EL metrics and comparisons.
`lambda_sensitivity_outputs/lambda_sensitivity_gradient_diagnostics.png` — manuscript Figure 3.
Public input data
The raw datasets are intentionally not duplicated in this repository.
SGCC electricity-theft dataset: https://github.com/henryRDlab/ElectricityTheftDetection
UCI Electricity Load Diagrams 2011–2014: https://doi.org/10.24432/C58C86
The SGCC input must retain the `CONS_NO` and `FLAG` columns and the dated consumption columns. The UCI input is the original semicolon-delimited Electricity Load Diagrams text file.
Place the downloaded files in a local data directory, for example:
```text
data/
├── sgcc.csv
└── LD2011_2014.txt
```
Software requirements
The scripts require Python and the following packages:
NumPy
pandas
PyTorch
SciPy
scikit-learn
Matplotlib
Install the required packages in a dedicated environment:
```bash
python -m pip install numpy pandas torch scipy scikit-learn matplotlib
```
Reproducing the experiments
Run all commands from the repository root. The benchmark should be run before the analysis and λ_EL sensitivity scripts because it creates the cached preprocessed datasets and the required seed-level outputs.
1. Run the SGCC benchmark
```bash
python pi_dlnn_reproduction.py --sgcc data/sgcc.csv --uci data/LD2011_2014.txt --datasets SGCC --epochs 20 --patience 4
```
2. Run the UCI-ELD benchmark
```bash
python pi_dlnn_reproduction.py --sgcc data/sgcc.csv --uci data/LD2011_2014.txt --datasets UCI --epochs 40 --patience 8
```
Both commands write to `experiment_outputs/`. Completed dataset–model–seed combinations are skipped unless `--force` is supplied.
3. Generate the final statistical analyses and figures
```bash
python analyze_final_results.py
```
This command reads the completed benchmark outputs and writes the final tables and figures to `experiment_outputs/final_analysis/`.
4. Run the λ_EL sensitivity analysis
```bash
python lambda_sensitivity_diagnostics.py
```
This command evaluates ten λ_EL values for both datasets and all ten seeds, produces the gradient diagnostics, and writes the results to `lambda_sensitivity_outputs/`.
Final experimental protocol
Seeds: 11, 23, 37, 53, 71, 89, 107, 131, 149, and 173.
Splitting: consumer/client-grouped 70%/15%/15% training, validation, and test partitions.
Preprocessing: 99.5th-percentile clipping and standardization fitted using the training partition only.
Threshold: selected by validation-set Matthews correlation coefficient and locked before test evaluation.
SGCC training: maximum 20 epochs with early-stopping patience of 4.
UCI-ELD training: maximum 40 epochs with early-stopping patience of 8.
Sensitivity weights: 0, 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 0.1, 1, 10, and 100.
Gradient diagnostics: separate binary cross-entropy and raw/weighted physics gradients on the shared LSTM encoder, evaluated each epoch using a deterministic subset of at most 2,048 training samples.
Sensitivity selection: the λ_EL value with the highest mean validation Matthews correlation coefficient across the ten prespecified seeds.
Inference: validation-selected λ_EL values are compared with λ_EL = 0 using paired bootstrap confidence intervals, two-sided Wilcoxon signed-rank tests, and Holm adjustment within each dataset and endpoint.
Transparency statement
The broad high-weight sensitivity extension was informed by gradient diagnostics and was not preregistered. The original λ_EL = 10⁻⁴ setting is reproduced using the same fixed-seed protocol. The repository preserves the completed results; rerunning the experiments requires the public raw data files and the Python dependencies listed above.
Citation
If you use this repository, please cite the associated manuscript. The complete journal citation and DOI will be added after publication.
Data and licensing
The raw SGCC and UCI-ELD datasets are not redistributed. Their respective providers retain ownership and licensing authority. Users are responsible for complying with the original dataset terms. Add a repository-level `LICENSE` file before public release to define the licence for the source code and generated materials.
