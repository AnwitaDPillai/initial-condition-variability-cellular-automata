# Initial-Condition Variability in Elementary Cellular Automata

**Author:** Anwita D Pillai (Independent Scholar)  
**Contact:** anwitadpillai007@gmail.com  
**ORCID:** [0009-0007-1499-4455](https://orcid.org/0009-0007-1499-4455)

## Overview
Elementary cellular automata (ECA) are deterministic systems whose global behavior depends on both local update rules and initial states. While local rules are simple, predicting global variability remains a challenge. 

This repository contains the complete, reproducible code and data for the study: *"Initial-Condition Variability in Elementary Cellular Automata: Local-Descriptor Prediction and an Exploratory Density Association."* 

The project quantifies this variation across all 256 Wolfram rules using 100 independent random initial states per rule (width 200, 200 updates) and tests whether local rule-table descriptors predict the variance. It also includes an exploratory analysis using Gaussian mixtures to investigate the association between trajectory complexity (LZ78) and mean density.

## Key Findings
1. **Local descriptors fail to predict global variance.** Under five-fold cross-validation grouped by symmetry equivalence classes, an enriched ridge model yielded an \(R^2\) of 0.0157, demonstrating that local rule-table descriptors provide weak predictive performance for global variance.
2. **Repeatable density association.** Across two lattice widths (200 and 400), held-out trajectories assigned to high- and low-complexity LZ78 components exhibited repeatable differences in mean density, though a secondary damage-spreading measure showed no significant effects.

## Repository Structure

```text
├── data/
│   ├── full_runs_100.csv                 # Raw trajectory-level data (25,600 trajectories)
│   ├── full_summary_100.csv              # Rule-level summary statistics
│   ├── grouped_cv_predictions_100.csv    # Cross-validation predictions
│   ├── grouped_cv_results_100.csv        # Cross-validation model metrics (RMSE, R2)
│   ├── width_specific_component_*.csv    # GMM results, tests, and robustness metrics
│   └── full_config_100.json              # Simulation configuration and seeds
├── figures/
│   ├── blinded_spacetime_gallery.png
│   ├── full_variability_histogram.png
│   ├── full_selected_distributions.png
│   └── variability_vs_lambda.png
├── src/
│   ├── automaton.py                      # NumPy ECA engine (periodic boundaries)
│   ├── features.py                       # Lambda, Boolean influences, sensitivity
│   ├── metrics.py                        # LZ78 complexity, density, damage curve
│   └── symmetry.py                       # 88 reflection/complement equivalence classes
├── tests/
│   └── test_engine.py                    # Unit tests for engine and metrics
├── analyze_full_100.py                   # GroupKFold CV analysis
├── replicate_width_specific_components.py# Post-hoc GMM analysis script
├── run_full_100.py                       # Main simulation script (all 256 rules)
├── requirements.txt                      # Python dependencies
└── README.md

```

## Reproducibility & Requirements

**Requirements:** Python 3.10+ is recommended. 

1. Install the required packages:
```bash
pip install -r requirements.txt
```
*(Note: `numba` is optional but highly recommended for accelerating the LZ78 parsing algorithm during the full run).*

2. Run the unit tests to ensure the engine is functioning correctly:
```bash
python -m pytest -q tests/
```

3. Run the main simulation (generates `data/full_runs_100.csv` and `data/full_summary_100.csv`):
```bash
python run_full_100.py
```
*(This script checkpoints every 8 rules to allow resuming if interrupted. The full run takes approximately 4 seconds on a modern CPU).*

4. Run the cross-validation analysis (generates `data/grouped_cv_results_100.csv`):
```bash
python analyze_full_100.py
```

5. Run the exploratory width-specific GMM analysis (generates the width-specific CSVs):
```bash
python replicate_width_specific_components.py
```

## Data & Code Availability
The code and data required to reproduce all results, figures, and tables in this manuscript are permanently archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23432754.svg)](https://doi.org/10.5281/zenodo.23432754)

## License
This project is licensed under the MIT License. The data and figures are available under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.