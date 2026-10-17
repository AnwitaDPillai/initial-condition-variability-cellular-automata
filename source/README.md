# Elementary Cellular Automata: Initial-Condition Variability

Reproducible NumPy implementation and experiments for testing whether local ECA rule-table descriptors predict the variability of global outcomes across random initial conditions.

## Current result in one sentence

Across 256 rules with 100 random initial conditions each, the chosen rule-table descriptors predict across-seed LZ78 variability poorly under five-fold validation grouped by the 88 exact reflection/complement classes. The enriched symmetry-invariant ridge model only slightly improves over a null predictor; this is a useful negative result for the chosen model, not proof that rule-table information is absent.

## Quick start

Requirements: Python 3.10+, NumPy, pandas, matplotlib, scikit-learn, SciPy, pytest; Numba is optional and accelerates LZ78 parsing when installed.

```bash
python -m pytest -q
python run_pilot.py
python run_full_100.py
python analyze_full_100.py
python run_size_robustness.py
```

The full-run script writes checkpoints every eight rules so it can resume after interruption. Run scripts from this directory.

## Files

- `src/automaton.py`: NumPy ECA engine, periodic boundaries, reproducible initial states.
- `src/features.py`: lambda, Boolean input influences, sensitivity, output entropy.
- `src/metrics.py`: LZ78 row-complexity score, density statistics, damage curve.
- `src/symmetry.py`: reflection/complement equivalence classes.
- `tests/`: engine, feature, metric, and symmetry tests.
- `run_pilot.py`: ten-rule pilot, 50 seeds per rule.
- `run_full.py`: all 256 rules, 50 seeds per rule.
- `run_full_100.py`: all 256 rules, 100 seeds per rule.
- `analyze_full_100.py`: grouped cross-validation analysis.
- `run_size_robustness.py`: exploratory width-100/400 check on selected high- and low-variability classes.
- `ANALYSIS_PLAN.md`: frozen definitions, models, and validation rules, including limitations of the current exploratory results.
- `data/`: per-run CSVs, rule-level summaries, configurations, model predictions, and robustness summaries.
- `figures/`: pilot and full-run plots.

## Main-run configuration

- 256 Wolfram rules; periodic boundary conditions.
- iid Bernoulli(0.5) initial states.
- Width 200; 200 updates; 100 seeds per rule.
- Primary trajectory outcome: mean over evolved rows of normalized LZ78 phrase count `C(n)*log2(n)/n`.
- Variability target: sample SD of the trajectory outcome across seeds.
- Validation: five-fold GroupKFold by the 88 reflection/complement equivalence classes.

See `ANALYSIS_PLAN.md` before changing metrics or running additional analyses. The current study is exploratory: the full scan and the final model choices were not externally preregistered before any pilot outcomes were examined.

## Literature context and novelty boundary

- Borriello & Walker (2017), *An Information-Based Classification of Elementary Cellular Automata*, explicitly varies initial conditions (including 20 random inputs per rule) and measures variation in average transfer entropy. DOI: https://doi.org/10.1155/2017/1280351
- Dhar et al. (1995), *Role of initial conditions in the classification of the rule space of cellular automata dynamics*, reports dependence of qualitative rule classification on initial conditions. DOI: https://doi.org/10.1103/PhysRevE.51.3032
- Pita (2026), *How local rules generate emergent structure in cellular automata*, preprint on causal tiles and dynamically decoupled regions: https://arxiv.org/abs/2604.00273
